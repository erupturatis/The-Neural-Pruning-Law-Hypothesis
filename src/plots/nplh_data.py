from __future__ import annotations

import csv
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Optional

from src.infrastructure.others import prefix_path_with_root


# ── Process-level identifier ──────────────────────────────────────────────────
# If NPLH_PROCESS_ID is set in the environment (e.g. by run_nplh_experiments.py
# before spawning child processes) all experiments in the same run share one
# output folder. Otherwise a fresh ID is generated — safe for standalone use.

_PROCESS_ID = os.environ.get("NPLH_PROCESS_ID") or (
    time.strftime('%Y%m%d_%H%M_') + ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
)

NPLH_DATA_FOLDER = 'nplh_data'


def get_process_folder() -> str:
    return os.path.join(prefix_path_with_root(NPLH_DATA_FOLDER), _PROCESS_ID)


def write_experiment_details(experiment_folder: str, content: str) -> None:
    """
    Writes a details.txt file into the experiment's subfolder.
    Creates the folder if it does not yet exist.
    """
    folder = os.path.join(get_process_folder(), experiment_folder)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "details.txt"), "w") as f:
        f.write(content)


# ── Sample ────────────────────────────────────────────────────────────────────

@dataclass
class NplhSample:
    density:                    float
    contributing:               float
    avg_saliency:               float
    avg_saliency_contributing:  float
    min_saliency:               Optional[float] = None
    min_saliency_contributing:  Optional[float] = None
    accuracy:                   Optional[float] = None
    test_loss:                  Optional[float] = None
    train_loss:                 Optional[float] = None
    epoch:                      Optional[int]   = None


# ── Series ────────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    'density', 'contributing',
    'avg_saliency', 'avg_saliency_contributing',
    'min_saliency', 'min_saliency_contributing',
    'accuracy', 'test_loss', 'train_loss', 'epoch',
]


class NplhSeries:
    """
    Records NPLH snapshots for one experimental series and persists them
    incrementally to a CSV file inside the experiment's subfolder.

    Usage:
        series = NplhSeries("lenet_magnitude", experiment_folder="lenet_magnitude_retrain")
        # inside the pruning loop:
        state  = compute_network_state(ctx)
        result = saliency_policy.measure_saliency(ctx, state)
        series.record(density, contributing, result.avg_saliency, ..., accuracy=acc, epoch=epoch)
        series.save()   # safe to call after every snapshot
    """

    def __init__(self, name: str, experiment_folder: str = ""):
        self.name = name
        self._samples: list[NplhSample] = []

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        short_id  = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        filename  = f"{name}_{timestamp}_{short_id}.csv"

        if experiment_folder:
            self._filepath = os.path.join(get_process_folder(), experiment_folder, filename)
        else:
            self._filepath = os.path.join(get_process_folder(), filename)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        density:                   float,
        contributing:              float,
        avg_saliency:              float,
        avg_saliency_contributing: float,
        min_saliency:              Optional[float] = None,
        min_saliency_contributing: Optional[float] = None,
        accuracy:                  Optional[float] = None,
        test_loss:                 Optional[float] = None,
        train_loss:                Optional[float] = None,
        epoch:                     Optional[int]   = None,
    ) -> None:
        self._samples.append(NplhSample(
            density=density,
            contributing=contributing,
            avg_saliency=avg_saliency,
            avg_saliency_contributing=avg_saliency_contributing,
            min_saliency=min_saliency,
            min_saliency_contributing=min_saliency_contributing,
            accuracy=accuracy,
            test_loss=test_loss,
            train_loss=train_loss,
            epoch=epoch,
        ))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Writes all recorded samples to the series CSV file, creating it (and
        any parent directories) if needed. Safe to call after every record()
        call — the file always reflects the full in-memory state so a mid-run
        crash loses at most the last unfinished sample.
        """
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        with open(self._filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_COLUMNS)
            for s in self._samples:
                writer.writerow([
                    s.density,
                    s.contributing,
                    s.avg_saliency,
                    s.avg_saliency_contributing,
                    s.min_saliency,
                    s.min_saliency_contributing,
                    '' if s.accuracy   is None else s.accuracy,
                    '' if s.test_loss  is None else s.test_loss,
                    '' if s.train_loss is None else s.train_loss,
                    '' if s.epoch      is None else s.epoch,
                ])

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def filepath(self) -> str:
        return self._filepath

    def __len__(self) -> int:
        return len(self._samples)
