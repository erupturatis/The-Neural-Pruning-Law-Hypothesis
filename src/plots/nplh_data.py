from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NplhSeries:
    density: np.ndarray
    saliency:  np.ndarray
    accuracy:  np.ndarray | None = field(default=None)  

COL_NAME_DENSITY = "density"
COL_NAME_ACCURACY = "accuracy"
EXPERIMENT_NAME = "..."
