"""
General-purpose GPU experiment runner using subprocess.

This runner uses `subprocess.Popen` instead of `multiprocessing.spawn`.
This guarantees `CUDA_VISIBLE_DEVICES` is set at the OS level before the
Python interpreter boots, completely preventing module-level CUDA
initializations (e.g., global tensors) from breaking device masking.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ── Experiment descriptor ──────────────────────────────────────────────────────

@dataclass
class ExperimentSpec:
    """
    Describes one experiment to run in an isolated GPU worker process.

    Attributes
    ----------
    name        : Unique identifier — used as the default log filename
                  ({log_dir}/{name}.log) when *log_path* is None.
    fn          : Zero-arg callable. Extracted by the runner to execute
                  in an isolated subprocess environment.
    description : Optional human-readable summary.
    log_path    : Explicit path for stdout/stderr capture.
    """
    name:        str
    fn:          Callable[[], None]
    description: str        = ""
    log_path:    str | None = None


# ── Scheduler / dispatcher ─────────────────────────────────────────────────────

def run_experiments(
    experiments:   list[ExperimentSpec],
    gpu_ids:       list[int] | None = None,
    log_dir:       str | Path       = "experiment_logs",
    poll_interval: float            = 10.0,
    main_log:      str | Path | None = None,
) -> None:
    """
    Dispatch a list of experiments across GPU slots using isolated subprocesses.
    """
    import torch  # Imported here so callers can import this module without torch

    # ── Resolve GPU list ──────────────────────────────────────────────────────
    if gpu_ids is None:
        n = torch.cuda.device_count()
        gpu_ids = list(range(n)) if n > 0 else []

    # A slot value of None means "run on CPU"
    slots: list[int | None] = list(gpu_ids) if gpu_ids else [None]

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Main log (optional) ───────────────────────────────────────────────────
    main_log_file = None
    if main_log is not None:
        Path(main_log).parent.mkdir(parents=True, exist_ok=True)
        main_log_file = open(main_log, "w", buffering=1)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if main_log_file is not None:
            print(msg, file=main_log_file, flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    gpu_label = f"{len(gpu_ids)} GPU(s) {gpu_ids}" if gpu_ids else "CPU"
    _log(f"[runner] {len(experiments)} experiment(s)  ·  {gpu_label}")
    _log(f"[runner] fallback log dir  →  {log_dir}")
    if main_log:
        _log(f"[runner] main log         →  {main_log}")

    # ── Dispatch loop ─────────────────────────────────────────────────────────
    pending = list(experiments)
    running_procs: list[tuple[subprocess.Popen, int | None, ExperimentSpec, Any]] = []

    while pending or running_procs:
        # Fill all free slots
        while slots and pending:
            gpu_id = slots.pop(0)
            spec   = pending.pop(0)

            # 1. Setup OS Environment
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            # 2. Resolve log path and redirect stdout/stderr
            resolved_log = spec.log_path if spec.log_path else str(log_dir / f"{spec.name}.log")
            Path(resolved_log).parent.mkdir(parents=True, exist_ok=True)
            log_file = open(resolved_log, "w", buffering=1)

            # 3. Construct subprocess execution command
            fn_name = spec.fn.__name__
            module = spec.fn.__module__

            # Handle standard imports vs execution from __main__
            if module == "__main__":
                script_path = os.path.abspath(sys.argv[0])
                py_code = (
                    f"import sys\n"
                    f"import importlib.util\n"
                    f"spec = importlib.util.spec_from_file_location('run_mod', r'{script_path}')\n"
                    f"run_mod = importlib.util.module_from_spec(spec)\n"
                    f"sys.modules['run_mod'] = run_mod\n"
                    f"spec.loader.exec_module(run_mod)\n"
                    f"run_mod.{fn_name}()\n"
                )
            else:
                py_code = f"from {module} import {fn_name}; {fn_name}()"

            cmd = [sys.executable, "-c", py_code]

            # 4. Dispatch
            label = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
            desc  = f"  [{spec.description}]" if spec.description else ""

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )

            running_procs.append((proc, gpu_id, spec, log_file))
            _log(f"[{label}]  started   {spec.name}{desc}  pid={proc.pid}  log={resolved_log}")

        # Reap finished processes
        still_running = []
        for proc, gpu_id, spec, log_file in running_procs:
            retcode = proc.poll()
            if retcode is not None:
                # Process finished
                log_file.close()
                label  = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
                status = "DONE" if retcode == 0 else f"FAILED (exit={retcode})"
                _log(f"[{label}]  {status:<20}  {spec.name}")
                slots.append(gpu_id)
            else:
                still_running.append((proc, gpu_id, spec, log_file))
        running_procs = still_running

        if running_procs and not slots:
            time.sleep(poll_interval)

    _log("\n[runner] All experiments complete.")

    if main_log_file is not None:
        main_log_file.close()


# ── CLI (optional standalone use) ─────────────────────────────────────────────

def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs, e.g. '0,1,2'.  Defaults to all detected GPUs.",
    )
    args = parser.parse_args()

    gpu_ids: list[int] | None = None
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        print(f"[runner] using GPUs: {gpu_ids}", flush=True)

    def _dummy() -> None:
        import time, os
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
        time.sleep(2)
        print("done")

    EXPERIMENTS = [
        ExperimentSpec(name=f"dummy_{i}", fn=_dummy, description="no-op sanity check")
        for i in range(4)
    ]

    run_experiments(EXPERIMENTS, gpu_ids=gpu_ids)


if __name__ == "__main__":
    _cli_main()
