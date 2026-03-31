
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

import csv as _csv
import os
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


# ---------------------------------------------------------------------------
# Saliency vs Density — log-log overlay plot
# ---------------------------------------------------------------------------

@dataclass
class SeriesSpec:
    """One CSV series to overlay on a saliency log-log plot.

    Args:
        csv_path:     Absolute or relative path to the NPLH CSV file.
        label:        Legend label. If None, the CSV filename stem is used.
        saliency_col: Which saliency column to plot — 'avg_saliency',
                      'avg_saliency_contributing', 'min_saliency', etc.
        x_col:        Which column to use as X axis — 'density' or
                      'contributing'.
    """
    csv_path: str
    label: str | None = None
    saliency_col: str = 'avg_saliency'
    x_col: str = 'density'


def _read_series(spec: SeriesSpec) -> tuple[list[float], list[float]]:
    """Read (x, saliency) pairs from one CSV.

    Points where x or saliency is <= 0 are silently dropped — they are
    incompatible with a log scale.  A warning is printed only if the entire
    series is empty after filtering, along with a count of how many rows were
    skipped, so the caller can tell the difference between a missing file and
    an all-zero column (which happens for min_saliency of Taylor/Gradient).
    """
    xs, ys = [], []
    skipped_zero = 0
    with open(spec.csv_path, newline='') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            x_raw = row.get(spec.x_col, '').strip()
            y_raw = row.get(spec.saliency_col, '').strip()
            if not x_raw or not y_raw:
                continue
            try:
                x, y = float(x_raw), float(y_raw)
            except ValueError:
                continue
            if x > 0 and y > 0:
                xs.append(x)
                ys.append(y)
            else:
                skipped_zero += 1

    if not xs and skipped_zero > 0:
        print(
            f"  [plot] WARNING: '{os.path.basename(spec.csv_path)}' col='{spec.saliency_col}' "
            f"— all {skipped_zero} rows have value <= 0 (not plottable on log scale). "
            f"This is expected for min_saliency of Taylor/Gradient metrics. "
            f"Try saliency_col='avg_saliency' instead."
        )
    return xs, ys


def plot_saliency_loglog(
    series: list[SeriesSpec],
    out_path: str | None = None,
    title: str = 'NPLH: Saliency vs Density',
    x_label: str = 'Remaining weights (%)',
    figsize: tuple[int, int] = (9, 6),
) -> None:
    """Log-log plot of saliency vs a density column, one line per CSV.

    X axis (log, left=dense → right=sparse): density or contributing %.
    Y axis (log): saliency value (avg or min, as specified per series).

    Args:
        series:   List of SeriesSpec — each specifies a CSV, its label,
                  which saliency column to use, and which x column to use.
        out_path: If given, saves the figure to this path instead of showing.
        title:    Plot title.
        x_label:  X axis label.
        figsize:  Figure size in inches (width, height).
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, spec in enumerate(series):
        colour = COLOURS[i % len(COLOURS)]
        label = spec.label if spec.label is not None else os.path.splitext(os.path.basename(spec.csv_path))[0]

        xs, ys = _read_series(spec)
        if not xs:
            print(f"  [plot] WARNING: no valid data in {spec.csv_path!r}, skipping.")
            continue

        xs_np = np.array(xs)
        ys_np = np.array(ys)
        order  = np.argsort(xs_np)[::-1]   # dense → sparse (high x first)
        xs_np  = xs_np[order]
        ys_np  = ys_np[order]

        ax.plot(xs_np, ys_np, color=colour, linewidth=1.5, alpha=0.85)
        ax.scatter(xs_np, ys_np, color=colour, s=MARKER_SIZE, zorder=3, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # X axis: dense (left) → sparse (right)
    ax.invert_xaxis()

    # Tick positions
    visible_x_ticks = [t for t in _X_TICKS if t <= 100]
    ax.set_xticks(visible_x_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}%'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Saliency', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE, framealpha=0.85)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"  [plot] saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)
