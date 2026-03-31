"""
Parallel experiment runner — assigns GPUs automatically.

Usage:
    python smoke_test_parallel.py                  # uses all detected GPUs
    python smoke_test_parallel.py --gpus 0,1       # restrict to specific GPUs
    python smoke_test_parallel.py --gpus 0,1,2     # use 3 GPUs

Each experiment is launched as an independent subprocess with CUDA_VISIBLE_DEVICES
set before any Python code runs, which is the only reliable way to pin a process
to a specific GPU. Output for each experiment is written to <exp_name>.log.

If there are more experiments than GPUs, experiments are queued and dispatched as
GPUs become free.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

ROOT = Path(__file__).parent

# ── Experiment registry ────────────────────────────────────────────────────────
# Each entry: (log_name, script_filename)
EXPERIMENTS = [
    ("resnet50_cifar10",  "run_resnet50_cifar10_dense.py"),
    ("resnet50_cifar100", "run_resnet50_cifar100_dense.py"),
    ("vgg19_cifar10",     "run_vgg19_cifar10_dense.py"),
    ("vgg19_cifar100",    "run_vgg19_cifar100_dense.py"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def preload_datasets() -> None:
    """Download datasets once in the main process to avoid race conditions."""
    print("Pre-downloading datasets...", flush=True)
    t = ToTensor()
    for cls, name in [(datasets.CIFAR10, "CIFAR-10"), (datasets.CIFAR100, "CIFAR-100")]:
        for split in (True, False):
            cls(root=str(ROOT / "data"), train=split, download=True, transform=t)
    print("Datasets ready.\n", flush=True)


def detect_gpus() -> list[int]:
    n = torch.cuda.device_count()
    if n == 0:
        print("No CUDA GPUs found — experiments will run on CPU.", flush=True)
        return []
    print(f"Detected {n} GPU(s): {list(range(n))}", flush=True)
    return list(range(n))


def launch(gpu_id: int | None, exp_name: str, script: str) -> tuple:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"   # flush print() immediately to log file
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        label = f"GPU {gpu_id}"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""   # force CPU
        label = "CPU"

    log_path = ROOT / f"{exp_name}.log"
    log_file = log_path.open("w")

    proc = subprocess.Popen(
        [sys.executable, str(ROOT / script)],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    print(f"[{label}] started {exp_name}  →  {log_path.name}  (pid {proc.pid})", flush=True)
    return proc, gpu_id, exp_name, log_file


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_all(gpu_ids: list[int]) -> None:
    # CPU-only fallback: run sequentially on one "slot"
    available = list(gpu_ids) if gpu_ids else [None]

    pending = list(EXPERIMENTS)
    running: list[tuple] = []   # (proc, gpu_id, exp_name, log_file)

    while pending or running:
        # Fill free GPU slots
        while available and pending:
            gpu_id = available.pop(0)
            exp_name, script = pending.pop(0)
            running.append(launch(gpu_id, exp_name, script))

        # Reap finished processes
        still_running = []
        for proc, gpu_id, exp_name, log_file in running:
            if proc.poll() is not None:
                log_file.close()
                ok = proc.returncode == 0
                label = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
                status = "DONE" if ok else f"FAILED (exit={proc.returncode})"
                print(f"[{label}] {exp_name}: {status}", flush=True)
                available.append(gpu_id)
            else:
                still_running.append((proc, gpu_id, exp_name, log_file))
        running = still_running

        if running and not available:
            time.sleep(10)

    print("\nAll experiments finished.", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs to use, e.g. '0,1,2'. Defaults to all detected GPUs.",
    )
    args = parser.parse_args()

    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        print(f"Using GPUs: {gpu_ids}", flush=True)
    else:
        gpu_ids = detect_gpus()

    preload_datasets()
    run_all(gpu_ids)


if __name__ == "__main__":
    main()
