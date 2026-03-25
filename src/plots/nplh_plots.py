
"""
nplh_plots.py
=============
Two canonical NPLH plotting functions used across all experiments.

plot_panels_with_accuracy(series_list, out_path, ...)
    One panel per series.
    X (log, dense → sparse) = remaining weights %.
    Left  Y (log) = saliency  (scatter + line, coloured).
    Right Y (lin) = accuracy  (dashed line + scatter, semi-transparent grey).
    The accuracy axis is omitted for series that have no accuracy data.

plot_joint(series_list, out_path, ...)
    All series overlaid on a single log-log plot.
    X (log, dense → sparse) = remaining weights %.
    Y (log) = saliency.
    No accuracy shown.

Both accept a list of NplhSeries instances.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

AXIS_LABEL_SIZE  = 13
TICK_LABEL_SIZE  = 11
LEGEND_FONT_SIZE = 9
TITLE_FONT_SIZE  = 13
MARKER_SIZE      = 30

COLOURS = plt.get_cmap("tab10").colors

_X_TICKS = [
    0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8,
    1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95,
]
