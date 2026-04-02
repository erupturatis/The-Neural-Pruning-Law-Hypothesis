"""
Persistent-worker GPU experiment runner.

Unlike the subprocess-based runner in experiment_runner.py, this runner keeps
one worker process alive per GPU for the entire session.  CUDA is initialised
exactly once per worker and the context is never torn down between experiments.

Motivation
----------
Repeated CUDA context creation/destruction (one per subprocess in the default
runner) can destabilise some GPU drivers, causing the device to become
inaccessible and requiring a full machine restart.  By keeping each worker
process alive for all experiments the context init/teardown events are reduced
from N (one per experiment) to 1 (one per GPU per full run).

Between experiments the worker explicitly frees model memory with:
    gc.collect()  →  torch.cuda.empty_cache()  →  torch.cuda.synchronize()
without ever calling into the CUDA driver's context-destroy path.

Public interface
----------------
Identical to experiment_runner.py::

    ExperimentSpec(name, fn, description="", log_path=None)
    run_experiments(experiments, gpu_ids=None, log_dir="experiment_logs",
                    poll_interval=10.0, main_log=None, batch_dispatch=False)

The only addition is the ``batch_dispatch`` keyword argument (see docstring).
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import multiprocessing
import os
import queue as queue_module
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ── Public interface (same as experiment_runner.py) ───────────────────────────

@dataclass
class ExperimentSpec:
    """
    Describes one experiment.  Identical to the dataclass in experiment_runner.

    Attributes
    ----------
    name        : Unique identifier, used as the default log filename.
    fn          : Zero-arg callable.  Must be a module-level function
                  (picklable) — no lambdas.
    description : Optional human-readable summary.
    log_path    : Explicit path for stdout/stderr capture.
    """
    name:        str
    fn:          Callable[[], None]
    description: str        = ""
    log_path:    str | None = None


# ── Internal message types ─────────────────────────────────────────────────────

@dataclass
class _WorkerTask:
    """Picklable representation of one experiment sent over a Queue."""
    spec_name:        str
    spec_description: str
    spec_log_path:    str | None
    fn_module:        str           # e.g. 'runners.run_nplh_experiments'
    fn_name:          str           # e.g. 'nplh_lenet_mnist'
    fn_script_path:   str | None    # set only when fn.__module__ == '__main__'
    log_dir:          str           # fallback log directory


@dataclass
class _WorkerResult:
    """Result returned by a worker after completing (or failing) one task."""
    spec_name:     str
    gpu_id:        int | None
    success:       bool
    error:         str | None = None
    traceback_str: str | None = None


# Poison pill: putting this into a task queue tells the worker to exit.
_SHUTDOWN: None = None


# ── Worker process (runs in a separate OS process, one per GPU) ───────────────

def _gpu_worker(
    gpu_id:       int | None,
    task_queue:   multiprocessing.Queue,  # type: ignore[type-arg]
    result_queue: multiprocessing.Queue,  # type: ignore[type-arg]
) -> None:
    """
    Long-lived worker process.

    Lifecycle
    ---------
    1. Set ``CUDA_VISIBLE_DEVICES`` to *gpu_id*.
    2. Import torch and perform a tiny CUDA operation to warm the context.
    3. Loop:
       a. Block on *task_queue*.
       b. On shutdown sentinel → break.
       c. Resolve the experiment function via importlib.
       d. Redirect stdout/stderr to the per-experiment log file.
       e. Call ``fn()``.  Catch all exceptions; never let a single bad
          experiment kill the worker.
       f. Restore stdout/stderr, close log file.
       g. gc.collect() + empty_cache() + synchronize() without destroying
          the CUDA context.
       h. Put a ``_WorkerResult`` on *result_queue*.
    """
    # ── 1. Device visibility ──────────────────────────────────────────────────
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ── 2. One-time CUDA initialisation ──────────────────────────────────────
    import torch  # noqa: PLC0415 — import inside function intentional

    _has_cuda = (gpu_id is not None) and torch.cuda.is_available()
    if _has_cuda:
        _warmup = torch.zeros(1, device="cuda:0")
        del _warmup
        torch.cuda.synchronize()

    label = f"GPU {gpu_id}" if gpu_id is not None else "CPU"

    # ── 3. Task loop ──────────────────────────────────────────────────────────
    while True:
        task: _WorkerTask | None = task_queue.get()

        # ── Shutdown sentinel ─────────────────────────────────────────────────
        if task is _SHUTDOWN:
            break

        # ── Resolve experiment function ───────────────────────────────────────
        fn = None
        try:
            if task.fn_script_path is not None:
                # Function lives in the __main__ script; load it by file path.
                _imp_spec = importlib.util.spec_from_file_location(
                    "_exp_mod", task.fn_script_path
                )
                _mod = importlib.util.module_from_spec(_imp_spec)  # type: ignore[arg-type]
                _imp_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
            else:
                _mod = importlib.import_module(task.fn_module)
            fn = getattr(_mod, task.fn_name)
        except Exception:
            result_queue.put(_WorkerResult(
                spec_name=task.spec_name,
                gpu_id=gpu_id,
                success=False,
                error="Failed to resolve experiment function",
                traceback_str=traceback.format_exc(),
            ))
            continue

        # ── Per-experiment log file ───────────────────────────────────────────
        log_path = (
            task.spec_log_path
            or str(Path(task.log_dir) / f"{task.spec_name}.log")
        )
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        # ── Run experiment with stdout/stderr redirected to log ───────────────
        success  = True
        err_msg  = None
        tb_str   = None
        _log_fh  = None
        _old_out = sys.stdout
        _old_err = sys.stderr

        try:
            _log_fh    = open(log_path, "w", buffering=1)
            sys.stdout = _log_fh
            sys.stderr = _log_fh
            fn()
        except Exception:
            success = False
            tb_str  = traceback.format_exc()
            err_msg = tb_str.strip().splitlines()[-1]
            # Write traceback to the experiment log while we still hold the fh.
            try:
                print(
                    f"\n[WORKER ERROR — {label}]\n{tb_str}",
                    file=_log_fh,
                    flush=True,
                )
            except Exception:
                pass
        finally:
            sys.stdout = _old_out
            sys.stderr = _old_err
            if _log_fh is not None:
                try:
                    _log_fh.close()
                except Exception:
                    pass

        # ── Return result ─────────────────────────────────────────────────────
        result_queue.put(_WorkerResult(
            spec_name=task.spec_name,
            gpu_id=gpu_id,
            success=success,
            error=err_msg,
            traceback_str=tb_str,
        ))

        # ── Deterministic memory cleanup (no CUDA context teardown) ──────────
        gc.collect()
        if _has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_experiments(
    experiments:    list[ExperimentSpec],
    gpu_ids:        list[int] | None  = None,
    log_dir:        str | Path        = "experiment_logs",
    poll_interval:  float             = 10.0,
    main_log:       str | Path | None = None,
    batch_dispatch: bool              = False,
) -> None:
    """
    Dispatch experiments to persistent per-GPU workers.

    Each GPU gets exactly one long-lived worker process.  CUDA is initialised
    once per worker.  Experiments are delivered via per-GPU Queues.

    Parameters
    ----------
    experiments
        Ordered list of experiments to run.
    gpu_ids
        GPU IDs to use.  ``None`` → all detected CUDA devices, or CPU if none.
    log_dir
        Directory for per-experiment log files (used as fallback when
        ``ExperimentSpec.log_path`` is ``None``).
    poll_interval
        Seconds between result-drain / watchdog passes.
    main_log
        Path for the orchestrator's own timestamped log (stdout is also used).
    batch_dispatch
        ``False`` (default) — **sequential dispatch**: experiments are sent
        to a worker only when that worker has finished its previous task.
        Mirrors the slot-based behaviour of the subprocess runner; ordering
        is preserved.

        ``True`` — **fire-and-forget dispatch**: all experiments are placed
        in worker queues upfront (round-robin across GPUs) before any results
        are awaited.  Maximises queue fill; useful when you have many more
        experiments than GPUs and want to avoid any idle time.
    """
    import torch  # noqa: PLC0415

    # ── Resolve GPU list ──────────────────────────────────────────────────────
    if gpu_ids is None:
        n = torch.cuda.device_count()
        gpu_ids = list(range(n)) if n > 0 else []
    slots: list[int | None] = list(gpu_ids) if gpu_ids else [None]

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Main log ──────────────────────────────────────────────────────────────
    _mlf = None
    if main_log is not None:
        Path(main_log).parent.mkdir(parents=True, exist_ok=True)
        _mlf = open(main_log, "w", buffering=1)

    def _log(msg: str) -> None:
        ts   = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if _mlf is not None:
            print(line, file=_mlf, flush=True)

    # ── Convert ExperimentSpec → _WorkerTask (picklable) ─────────────────────
    tasks: list[_WorkerTask] = []
    for spec in experiments:
        fn_mod    = spec.fn.__module__
        fn_name   = spec.fn.__name__
        fn_script = os.path.abspath(sys.argv[0]) if fn_mod == "__main__" else None
        tasks.append(_WorkerTask(
            spec_name        = spec.name,
            spec_description = spec.description,
            spec_log_path    = spec.log_path,
            fn_module        = fn_mod,
            fn_name          = fn_name,
            fn_script_path   = fn_script,
            log_dir          = str(log_dir),
        ))

    # ── Spawn persistent workers (one per GPU slot) ───────────────────────────
    mp_ctx = multiprocessing.get_context("spawn")

    task_queues:  dict[int | None, multiprocessing.Queue] = {}  # type: ignore[type-arg]
    result_queue: multiprocessing.Queue = mp_ctx.Queue()         # type: ignore[type-arg]
    workers:      dict[int | None, multiprocessing.Process] = {}

    # Save the parent's CUDA_VISIBLE_DEVICES so we can restore it after spawning.
    _cvd_before = os.environ.get("CUDA_VISIBLE_DEVICES")

    for gpu_id in slots:
        # Set CUDA_VISIBLE_DEVICES in the PARENT before proc.start().
        # With multiprocessing.spawn, Python re-imports __main__ during
        # bootstrap in the child process (before _gpu_worker runs), which
        # triggers any module-level get_device() / CUDA calls in the imported
        # experiment code.  The child inherits the parent's environment at fork
        # time, so setting it here ensures the correct GPU is visible from the
        # very first import in the child.
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        tq = mp_ctx.Queue()
        task_queues[gpu_id] = tq
        proc = mp_ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, tq, result_queue),
            daemon=False,
        )
        proc.start()
        lbl = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
        _log(f"[runner] worker started  {lbl}  pid={proc.pid}")
        workers[gpu_id] = proc

    # Restore parent's original CUDA_VISIBLE_DEVICES (orchestrator doesn't
    # use CUDA itself, but restore for cleanliness).
    if _cvd_before is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _cvd_before
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    gpu_label = f"{len(gpu_ids)} GPU(s) {gpu_ids}" if gpu_ids else "CPU-only"
    mode_label = "batch/fire-and-forget" if batch_dispatch else "sequential"
    _log(
        f"[runner] {len(experiments)} experiment(s)  ·  {gpu_label}  ·  "
        f"{mode_label} dispatch"
    )

    # ── Dispatch state ────────────────────────────────────────────────────────
    # current_task[gpu_id]: the task currently being executed by that worker
    # (None = worker is idle / between tasks).
    current_task: dict[int | None, _WorkerTask | None] = {g: None for g in slots}

    # For watchdog in batch mode: which task names were assigned to each GPU.
    assigned_to:  dict[int | None, list[str]] = {g: [] for g in slots}

    pending_tasks   = list(tasks)
    free_slots      = list(slots)   # starts full; shrinks as tasks dispatched
    results_pending = 0
    n_done          = 0
    n_total         = len(tasks)

    # ── Batch dispatch: pre-fill all queues round-robin ───────────────────────
    if batch_dispatch:
        for i, task in enumerate(pending_tasks):
            gpu_id = slots[i % len(slots)]
            task_queues[gpu_id].put(task)
            assigned_to[gpu_id].append(task.spec_name)
            desc = f"  [{task.spec_description}]" if task.spec_description else ""
            lbl  = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
            _log(f"[{lbl}]  queued    {task.spec_name}{desc}")
            results_pending += 1
        pending_tasks.clear()
        free_slots.clear()

    # ── Main loop: dispatch (sequential mode) + collect results + watchdog ────
    while n_done < n_total:

        # -- Watchdog: detect dead worker processes ---------------------------
        dead = [gid for gid, p in list(workers.items()) if not p.is_alive()]
        for gpu_id in dead:
            proc = workers.pop(gpu_id)
            lbl  = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
            running = current_task.get(gpu_id)
            if running is not None:
                _log(
                    f"[WATCHDOG] worker {lbl} (pid={proc.pid}) died unexpectedly "
                    f"while running '{running.spec_name}'"
                )
            elif batch_dispatch and assigned_to[gpu_id]:
                remaining = assigned_to[gpu_id]
                _log(
                    f"[WATCHDOG] worker {lbl} (pid={proc.pid}) died unexpectedly. "
                    f"Tasks assigned to it (not necessarily all failed): "
                    f"{remaining}"
                )
            else:
                _log(
                    f"[WATCHDOG] worker {lbl} (pid={proc.pid}) died unexpectedly "
                    f"(idle — no current task)"
                )

        # -- Sequential dispatch: send next task to each free worker ----------
        if not batch_dispatch:
            while free_slots and pending_tasks:
                gpu_id = free_slots.pop(0)
                task   = pending_tasks.pop(0)
                task_queues[gpu_id].put(task)
                current_task[gpu_id] = task
                results_pending     += 1
                lbl  = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
                desc = f"  [{task.spec_description}]" if task.spec_description else ""
                _log(f"[{lbl}]  dispatched  {task.spec_name}{desc}")

        # -- Drain all available results --------------------------------------
        drained = False
        while True:
            try:
                result: _WorkerResult = result_queue.get_nowait()
            except queue_module.Empty:
                break

            drained  = True
            n_done  += 1
            results_pending -= 1
            lbl    = f"GPU {result.gpu_id}" if result.gpu_id is not None else "CPU"
            status = "DONE" if result.success else f"FAILED — {result.error}"
            _log(
                f"[{lbl}]  {status:<55}  {result.spec_name}  "
                f"({n_done}/{n_total})"
            )

            if not batch_dispatch:
                # Free the slot so the next pending task can be dispatched.
                current_task[result.gpu_id] = None
                free_slots.append(result.gpu_id)

            if batch_dispatch and result.gpu_id in assigned_to:
                # Remove from watchdog's task-name list.
                try:
                    assigned_to[result.gpu_id].remove(result.spec_name)
                except ValueError:
                    pass

        if not drained:
            time.sleep(poll_interval)

    # ── Send shutdown signals and wait for workers to exit cleanly ────────────
    for gpu_id, tq in task_queues.items():
        tq.put(_SHUTDOWN)

    for gpu_id, proc in list(workers.items()):
        proc.join(timeout=30)
        if proc.is_alive():
            lbl = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
            _log(f"[runner] worker {lbl} did not exit within 30 s — terminating")
            proc.terminate()

    _log("\n[runner] All experiments complete.")

    if _mlf is not None:
        _mlf.close()


# ── CLI (optional standalone sanity check) ────────────────────────────────────

def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Persistent-worker GPU experiment runner (sanity check)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs, e.g. '0,1'.  Defaults to all detected.",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use batch/fire-and-forget dispatch instead of sequential.",
    )
    args = parser.parse_args()

    gpu_ids: list[int] | None = None
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]

    def _dummy() -> None:
        import time as _t, os as _os
        print(
            f"pid={_os.getpid()}  "
            f"CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
        _t.sleep(1)
        print("done")

    EXPERIMENTS = [
        ExperimentSpec(name=f"dummy_{i}", fn=_dummy, description="no-op sanity")
        for i in range(6)
    ]

    run_experiments(
        EXPERIMENTS,
        gpu_ids=gpu_ids,
        batch_dispatch=args.batch,
    )


if __name__ == "__main__":
    _cli_main()
