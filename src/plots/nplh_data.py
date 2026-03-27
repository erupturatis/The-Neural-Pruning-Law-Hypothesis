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
# Generated once at import time. All series created in this process are saved
# under nplh_data/{_PROCESS_ID}/ so runs never collide and are easy to group.

_PROCESS_ID = time.strftime('%Y%m%d_%H%M_') + ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

NPLH_DATA_FOLDER = 'nplh_data'


def get_process_folder() -> str:
    return os.path.join(prefix_path_with_root(NPLH_DATA_FOLDER), _PROCESS_ID)


# ── Sample ────────────────────────────────────────────────────────────────────

@dataclass
class NplhSample:
    density:      float
    avg_saliency: float
    min_saliency: Optional[float] = None
    accuracy:     Optional[float] = None
    epoch:        Optional[int]   = None


# ── Series ────────────────────────────────────────────────────────────────────

_CSV_COLUMNS = ['density', 'min_saliency', 'avg_saliency', 'accuracy', 'epoch']


class NplhSeries:
    """
    Records NPLH snapshots for one experimental series and persists them
    incrementally to a CSV file.

    Usage:
        series = NplhSeries("lenet_magnitude")
        # inside the pruning loop:
        min_sal, avg_sal = saliency_policy.measure_saliency(ctx)
        series.record(density, min_sal, avg_sal, accuracy=acc, epoch=epoch)
        series.save()   # safe to call after every snapshot
    """

    def __init__(self, name: str):
        self.name = name
        self._samples: list[NplhSample] = []

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        short_id  = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        filename  = f"{name}_{timestamp}_{short_id}.csv"
        self._filepath = os.path.join(get_process_folder(), filename)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        density:      float,
        avg_saliency: float,
        min_saliency: Optional[float] = None,
        accuracy:     Optional[float] = None,
        epoch:        Optional[int]   = None,
    ) -> None:
        self._samples.append(NplhSample(density, avg_saliency, min_saliency, accuracy, epoch))

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
                    s.min_saliency,
                    s.avg_saliency,
                    '' if s.accuracy is None else s.accuracy,
                    '' if s.epoch    is None else s.epoch,
                ])

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def filepath(self) -> str:
        return self._filepath

    def __len__(self) -> int:
        return len(self._samples)
