
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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter(series: NplhSeries) -> NplhSeries:
    """Drop rows where saliency ≤ 0 or remaining ≤ 0."""
    keep = (series.saliency > 0) & (series.remaining > 0)
    acc  = series.accuracy[keep] if series.accuracy is not None else None
    return NplhSeries(
        label=series.label,
        remaining=series.remaining[keep],
        saliency=series.saliency[keep],
        accuracy=acc,
    )


def _style_x_axis(ax, xlim_right: float, xlim_left: float) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.FixedLocator(_X_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Remaining weights (%, dense → sparse)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, rotation=35)
    ax.set_xlim(xlim_left, xlim_right)


# ---------------------------------------------------------------------------
# Plot 1: one panel per series, with optional accuracy on twin axis
# ---------------------------------------------------------------------------

def plot_panels_with_accuracy(
    series_list: list[NplhSeries],
    out_path:     str,
    saliency_label: str = "Saliency",
    title:          str = "NPLH – Remaining Weights vs Saliency & Accuracy",
    ncols:          int = 3,
) -> None:
    """
    One panel per series.

    X (log, dense → sparse) = remaining weights %.
    Left  Y (log) = saliency  (scatter + line).
    Right Y (lin) = accuracy  (dashed + scatter, semi-transparent grey).
    Accuracy axis omitted when series.accuracy is None.
    """
    filtered = [_filter(s) for s in series_list]
    filtered = [s for s in filtered if len(s.remaining) >= 2]
    n = len(filtered)
    if n == 0:
        print("[nplh_plots] No usable series for panel plot.")
        return

    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    for i, series in enumerate(filtered):
        ax_sal = axes[i]
        colour = COLOURS[i % len(COLOURS)]

        rem = series.remaining
        sal = series.saliency
        order = np.argsort(rem)[::-1]   # dense → sparse
        rem_s = rem[order]
        sal_s = sal[order]

        ax_sal.scatter(rem_s, sal_s, s=MARKER_SIZE, color=colour, alpha=0.85, zorder=5)
        ax_sal.plot(rem_s, sal_s, color=colour, linewidth=1.2, alpha=0.45, zorder=4)

        ax_sal.set_yscale("log")
        ax_sal.set_ylabel(saliency_label, fontsize=AXIS_LABEL_SIZE)
        ax_sal.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax_sal.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax_sal.set_title(series.label, fontsize=TITLE_FONT_SIZE)

        xlim_left  = rem_s.max() * 1.1
        xlim_right = max(rem_s.min() * 0.8, 0.04)
        _style_x_axis(ax_sal, xlim_right, xlim_left)

        legend_handles = [
            Line2D([0], [0], color=colour, marker="o", markersize=5,
                   linestyle="-", linewidth=1.2, label="Saliency (left)"),
        ]

        if series.accuracy is not None:
            ax_acc = ax_sal.twinx()
            acc_s  = series.accuracy[order]
            ax_acc.plot(rem_s, acc_s, color="dimgray", linewidth=1.6,
                        linestyle="--", alpha=0.30, zorder=6)
            ax_acc.scatter(rem_s, acc_s, s=MARKER_SIZE * 0.5, color="dimgray",
                           marker="^", alpha=0.25, zorder=7)
            ax_acc.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_SIZE - 1,
                              color="dimgray")
            ax_acc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE - 1,
                               labelcolor="dimgray")
            ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))
            legend_handles.append(
                Line2D([0], [0], color="dimgray", marker="^", markersize=5,
                       linestyle="--", linewidth=1.4, alpha=0.4,
                       label="Accuracy (right)"),
            )

        ax_sal.legend(handles=legend_handles, fontsize=LEGEND_FONT_SIZE,
                      loc="lower right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE + 2, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[nplh_plots] Saved panels → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: all series on one log-log plot, saliency only
# ---------------------------------------------------------------------------

def plot_joint(
    series_list: list[NplhSeries],
    out_path:     str,
    saliency_label: str = "Saliency",
    title:          str = "NPLH – Remaining Weights vs Saliency (all series)",
) -> None:
    """
    All series overlaid on one log-log plot.

    X (log, dense → sparse) = remaining weights %.
    Y (log) = saliency.
    No accuracy shown.
    """
    filtered = [_filter(s) for s in series_list]
    filtered = [s for s in filtered if len(s.remaining) >= 2]
    if not filtered:
        print("[nplh_plots] No usable series for joint plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, series in enumerate(filtered):
        colour = COLOURS[i % len(COLOURS)]
        rem = series.remaining
        sal = series.saliency
        order = np.argsort(rem)[::-1]
        rem_s = rem[order]
        sal_s = sal[order]

        ax.scatter(rem_s, sal_s, s=MARKER_SIZE, color=colour,
                   alpha=0.80, zorder=5, label=series.label)
        ax.plot(rem_s, sal_s, color=colour, linewidth=1.2, alpha=0.45, zorder=4)

    ax.set_yscale("log")
    ax.set_ylabel(saliency_label, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE + 1)

    all_rem    = np.concatenate([s.remaining for s in filtered])
    xlim_left  = all_rem.max() * 1.1
    xlim_right = max(all_rem.min() * 0.8, 0.04)
    _style_x_axis(ax, xlim_right, xlim_left)

    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right",
              title="Series", title_fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[nplh_plots] Saved joint  → {out_path}")
    plt.close()
