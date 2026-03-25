"""
plot_lenet_variable.py
======================
Generates 5 figures for the variable-size LeNet IMP experiment:

  1. NPLH subplots        – one panel per architecture.
  2. NPLH combined        – all architectures overlaid on one log-log plot.
  3. NPLH + accuracy      – combined plot with accuracy overlaid on a twin axis.
  4. Baseline accuracy vs parameter count.
  5. End-of-pruning accuracy vs parameter count.

Outlier filtering: any row where saliency < 1e-6 is dropped.

Run from the project root:
    python -m src.mnist_lenet300.plot_lenet_variable
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

from src.infrastructure.nplh_run_context import COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY
from src.infrastructure.others import prefix_path_with_root
from src.plots.nplh_plots import NplhSeries, plot_panels_with_accuracy, plot_joint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default legacy data directory (old runs without a run ID)
_LEGACY_DATA_DIR = "neural_pruning_law/final_data/lenet_variable"

ARCHITECTURES = [
    {"name": "lenet_4_2",       "h1":    4, "h2":    2},
    {"name": "lenet_10_5",      "h1":   10, "h2":    5},
    {"name": "lenet_20_10",     "h1":   20, "h2":   10},
    {"name": "lenet_50_25",     "h1":   50, "h2":   25},
    {"name": "lenet_100_50",    "h1":  100, "h2":   50},
    {"name": "lenet_300_100",   "h1":  300, "h2":  100},
    {"name": "lenet_600_200",   "h1":  600, "h2":  200},
    {"name": "lenet_1200_400",  "h1": 1200, "h2":  400},
    {"name": "lenet_2500_800",  "h1": 2500, "h2":  800},
    {"name": "lenet_5000_1500", "h1": 5000, "h2": 1500},
]

# Subset up to and including the canonical (300, 100) network
ARCHITECTURES_SMALL = ARCHITECTURES[:6]

AXIS_LABEL_SIZE  = 13
TICK_LABEL_SIZE  = 11
LEGEND_FONT_SIZE = 9
TITLE_FONT_SIZE  = 12
MARKER_SIZE      = 30

COLOURS = plt.get_cmap("tab10").colors

# 15 fixed y-tick positions (%) covering the full pruning range
_Y_TICKS = [0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]

# 20 fixed x-tick positions for remaining-weights plots (log scale, %)
_X_TICKS = [0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prunable_params(h1: int, h2: int) -> int:
    return 784 * h1 + h1 * h2 + h2 * 10


def _read_imp_csv(path: str, saliency_col: str = None):
    """
    Returns dict with numpy arrays or None if file missing / too short.

    saliency_col: column name to read saliency from.  If None, tries the
    standardised name (COL_SALIENCY = "Saliency") first, then falls back to
    legacy names ("SaliencyIMP", "SaliencyAvg") for backward compatibility.
    """
    if not os.path.isfile(path):
        return None
    saliency, remaining, accuracy = [], [], []
    _LEGACY_SAL_COLS = ["SaliencyIMP", "SaliencyAvg"]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Resolve which saliency column to use
        if saliency_col is None:
            if COL_SALIENCY in fieldnames:
                saliency_col = COL_SALIENCY
            else:
                saliency_col = next(
                    (c for c in _LEGACY_SAL_COLS if c in fieldnames), None
                )
        if saliency_col is None:
            return None

        for row in reader:
            try:
                saliency.append(float(row[saliency_col]))
                remaining.append(float(row[COL_REMAINING]))
                # Accuracy column is optional
                acc_val = row.get(COL_ACCURACY, "")
                accuracy.append(float(acc_val) if acc_val else float("nan"))
            except (ValueError, KeyError):
                continue
    if len(saliency) < 2:
        return None
    acc_arr = np.array(accuracy)
    return {
        "saliency":  np.array(saliency),
        "remaining": np.array(remaining),
        "accuracy":  acc_arr if not np.all(np.isnan(acc_arr)) else None,
    }


def _filter_outliers(data: dict) -> dict:
    """Drop any row where saliency < 1e-6."""
    keep = data["saliency"] >= 1e-6
    return {
        "saliency":  data["saliency"][keep],
        "remaining": data["remaining"][keep],
        "accuracy":  data["accuracy"][keep],
    }


def _style_nplh_axes(ax, title: str, saliency_label: str = "Min Saliency"):
    """Apply shared styling to an NPLH scatter axis (left / primary)."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.FixedLocator(_Y_TICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_xlabel(f"{saliency_label} (magnitude threshold)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Remaining weights (%)",                   fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)


# ---------------------------------------------------------------------------
# Figure 1 – NPLH subplots
# ---------------------------------------------------------------------------

def plot_nplh_subplots(data_dir: str, out_path: str,
                       arch_list=None, csv_suffix="_imp",
                       saliency_col="SaliencyIMP", saliency_label="Min Saliency"):
    arch_list = arch_list or ARCHITECTURES
    available = []
    for arch in arch_list:
        path = os.path.join(data_dir, f"{arch['name']}{csv_suffix}.csv")
        data = _read_imp_csv(path, saliency_col)
        if data is not None:
            available.append((arch, _filter_outliers(data)))

    n = len(available)
    if n == 0:
        print("No data found for NPLH subplots.")
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for i, (arch, data) in enumerate(available):
        ax = axes[i]
        h1, h2 = arch["h1"], arch["h2"]
        params = _prunable_params(h1, h2)
        ax.scatter(
            data["saliency"], data["remaining"],
            s=MARKER_SIZE, color=COLOURS[i % len(COLOURS)], alpha=0.80, zorder=5,
        )
        _style_nplh_axes(ax, f"LeNet ({h1}, {h2})  –  {params:,} params", saliency_label)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"NPLH: {saliency_label} vs Remaining Weights  –  Variable LeNet on MNIST",
        fontsize=TITLE_FONT_SIZE + 2, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved NPLH subplots        → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 – NPLH combined
# ---------------------------------------------------------------------------

def plot_nplh_combined(data_dir: str, out_path: str,
                       arch_list=None, csv_suffix="_imp",
                       saliency_col="SaliencyIMP", saliency_label="Min Saliency"):
    arch_list = arch_list or ARCHITECTURES
    fig, ax = plt.subplots(figsize=(10, 7))

    plotted = 0
    for i, arch in enumerate(arch_list):
        path = os.path.join(data_dir, f"{arch['name']}{csv_suffix}.csv")
        data = _read_imp_csv(path, saliency_col)
        if data is None:
            continue
        data = _filter_outliers(data)
        h1, h2 = arch["h1"], arch["h2"]
        label = f"({h1}, {h2})  –  {_prunable_params(h1, h2):,} params"
        ax.scatter(
            data["saliency"], data["remaining"],
            s=MARKER_SIZE, color=COLOURS[i % len(COLOURS)],
            alpha=0.75, zorder=5, label=label,
        )
        plotted += 1

    if plotted == 0:
        print("No data found for combined NPLH plot.")
        plt.close()
        return

    _style_nplh_axes(ax, f"NPLH: All Architectures  –  Variable LeNet on MNIST",
                     saliency_label)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left",
              title="Architecture", title_fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved NPLH combined        → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 – NPLH + accuracy subplots (one panel per architecture)
# ---------------------------------------------------------------------------

def plot_nplh_combined_with_accuracy(data_dir: str, out_path: str,
                                     arch_list=None, csv_suffix="_imp",
                                     saliency_col="SaliencyIMP", saliency_label="Min Saliency"):
    from matplotlib.lines import Line2D

    arch_list = arch_list or ARCHITECTURES
    available = []
    for arch in arch_list:
        path = os.path.join(data_dir, f"{arch['name']}{csv_suffix}.csv")
        data = _read_imp_csv(path, saliency_col)
        if data is not None:
            available.append((arch, _filter_outliers(data)))

    n = len(available)
    if n == 0:
        print("No data found for NPLH + accuracy subplots.")
        return

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, (arch, data) in enumerate(available):
        ax_rem = axes[i]
        ax_acc = ax_rem.twinx()
        colour = COLOURS[i % len(COLOURS)]
        h1, h2 = arch["h1"], arch["h2"]
        params = _prunable_params(h1, h2)

        # NPLH scatter on left axis
        ax_rem.scatter(
            data["saliency"], data["remaining"],
            s=MARKER_SIZE, color=colour, alpha=0.80, zorder=5,
            label="Remaining %",
        )

        # Accuracy on right axis — shaded area in the background, line on top
        ax_acc.fill_between(
            data["saliency"], data["accuracy"],
            alpha=0.12, color=colour, zorder=2,
        )
        ax_acc.plot(
            data["saliency"], data["accuracy"],
            color=colour, linewidth=1.6, linestyle="--", alpha=0.75, zorder=3,
        )
        ax_acc.scatter(
            data["saliency"], data["accuracy"],
            s=MARKER_SIZE * 0.5, color=colour, marker="^",
            alpha=0.75, zorder=4, label="Accuracy",
        )

        # Style left axis
        _style_nplh_axes(ax_rem, f"LeNet ({h1}, {h2})  –  {params:,} params",
                         saliency_label)

        # Style right axis
        ax_acc.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_SIZE - 1,
                          color="dimgray")
        ax_acc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE - 1,
                           labelcolor="dimgray")
        ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))

        # Small legend inside each panel
        legend_handles = [
            Line2D([0], [0], color=colour, linestyle="none",
                   marker="o", markersize=5, label="Remaining % (left)"),
            Line2D([0], [0], color=colour, linestyle="--", linewidth=1.4,
                   marker="^", markersize=5, label="Accuracy (right)"),
        ]
        ax_rem.legend(handles=legend_handles, fontsize=LEGEND_FONT_SIZE - 1,
                      loc="upper left")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"NPLH + Accuracy vs {saliency_label}  –  Variable LeNet on MNIST",
        fontsize=TITLE_FONT_SIZE + 2, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved NPLH + accuracy      → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4 – NPLH subplots with accuracy-drop regions highlighted
# ---------------------------------------------------------------------------

def plot_nplh_with_accuracy_regions(data_dir: str, out_path: str,
                                    drop_threshold_pct: float = 1.0,
                                    arch_list=None, csv_suffix="_imp",
                                    saliency_col="SaliencyIMP", saliency_label="Min Saliency"):
    """
    9-panel grid.  Each panel:
      • NPLH scatter points coloured by accuracy (RdYlGn colormap,
        normalised per-architecture so within-run variation is visible).
      • Light red background shading for the saliency region where accuracy
        has dropped more than `drop_threshold_pct` below that network's peak.
      • Vertical dashed line marking the boundary of the drop.
      • Accuracy line on a twin right axis for precise reading.
    """
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    arch_list = arch_list or ARCHITECTURES
    available = []
    for arch in arch_list:
        path = os.path.join(data_dir, f"{arch['name']}{csv_suffix}.csv")
        data = _read_imp_csv(path, saliency_col)
        if data is not None:
            available.append((arch, _filter_outliers(data)))

    n = len(available)
    if n == 0:
        print("No data found for accuracy-region plot.")
        return

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    cmap = plt.get_cmap("RdYlGn")

    for i, (arch, data) in enumerate(available):
        ax_rem = axes[i]
        ax_acc = ax_rem.twinx()
        h1, h2  = arch["h1"], arch["h2"]
        params  = _prunable_params(h1, h2)
        sal     = data["saliency"]
        rem     = data["remaining"]
        acc     = data["accuracy"]

        # Per-architecture normalisation so colour variation is always visible
        acc_min, acc_max = acc.min(), acc.max()
        # Avoid degenerate case where accuracy is perfectly flat
        if acc_max - acc_min < 0.01:
            acc_min = acc_max - 0.5
        norm = Normalize(vmin=acc_min, vmax=acc_max)

        # ── NPLH scatter, coloured by accuracy ──────────────────────────────
        sc = ax_rem.scatter(
            sal, rem,
            c=acc, cmap=cmap, norm=norm,
            s=MARKER_SIZE + 5, zorder=5, alpha=0.90,
        )

        # ── Identify accuracy-drop region ───────────────────────────────────
        peak_acc   = acc.max()
        drop_level = peak_acc - drop_threshold_pct
        dropped    = acc < drop_level

        if dropped.any():
            # First saliency value where accuracy goes below threshold
            first_drop_sal = sal[dropped][0]

            # Shade the degraded region (from that saliency to the right)
            ax_rem.axvspan(
                first_drop_sal, sal.max() * 1.5,
                alpha=0.12, color="red", zorder=1,
            )
            # Vertical marker at the boundary
            ax_rem.axvline(
                first_drop_sal,
                color="red", linewidth=1.4, linestyle="--",
                alpha=0.75, zorder=6,
                label=f"Drop > {drop_threshold_pct}%",
            )
            # Annotate with the drop saliency
            ax_rem.text(
                first_drop_sal, rem.max() * 0.7,
                f" γ={first_drop_sal:.2e}",
                color="red", fontsize=7.5, va="top", rotation=90, zorder=7,
            )
        else:
            # No significant drop — green background to indicate stability
            ax_rem.set_facecolor("#f0fff0")

        # ── Accuracy line on right axis ──────────────────────────────────────
        ax_acc.plot(sal, acc, color="dimgray", linewidth=1.4,
                    linestyle="--", alpha=0.65, zorder=3)
        ax_acc.axhline(
            drop_level, color="red", linewidth=0.9,
            linestyle=":", alpha=0.60, zorder=2,
        )
        ax_acc.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_SIZE - 1,
                          color="dimgray")
        ax_acc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE - 1,
                           labelcolor="dimgray")
        ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))

        # ── Style left axis ──────────────────────────────────────────────────
        _style_nplh_axes(ax_rem, f"LeNet ({h1}, {h2})  –  {params:,} params",
                         saliency_label)

        # ── Per-panel colourbar ──────────────────────────────────────────────
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_rem, pad=0.14, fraction=0.035)
        cb.set_label("Accuracy (%)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # ── Legend ───────────────────────────────────────────────────────────
        legend_handles = [
            Line2D([0], [0], color="red", linestyle="--", linewidth=1.2,
                   label=f"Acc drop > {drop_threshold_pct}%"),
            Line2D([0], [0], color="dimgray", linestyle="--", linewidth=1.2,
                   label="Accuracy (right)"),
        ]
        ax_rem.legend(handles=legend_handles, fontsize=LEGEND_FONT_SIZE - 1,
                      loc="lower left")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"NPLH coloured by Accuracy  –  Variable LeNet on MNIST\n"
        f"(red shading = accuracy dropped > {drop_threshold_pct}% below peak)",
        fontsize=TITLE_FONT_SIZE + 1, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved NPLH accuracy regions → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5 – Remaining weights (x) vs Saliency + Accuracy (twin y)
# ---------------------------------------------------------------------------

def plot_remaining_vs_saliency_accuracy(data_dir: str, out_path: str,
                                        arch_list=None, csv_suffix="_imp",
                                        saliency_col="SaliencyIMP", saliency_label="Min Saliency"):
    """
    9-panel grid.  X axis = remaining weights % (log, dense → sparse).
    Left  Y = saliency (log scale).
    Right Y = accuracy (linear, semi-transparent).
    No drop highlighting.
    """
    from matplotlib.lines import Line2D

    arch_list = arch_list or ARCHITECTURES
    available = []
    for arch in arch_list:
        path = os.path.join(data_dir, f"{arch['name']}{csv_suffix}.csv")
        data = _read_imp_csv(path, saliency_col)
        if data is not None:
            available.append((arch, _filter_outliers(data)))

    n = len(available)
    if n == 0:
        print("No data found for remaining-vs-saliency-accuracy plot.")
        return

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    for i, (arch, data) in enumerate(available):
        ax_sal = axes[i]
        ax_acc = ax_sal.twinx()
        colour = COLOURS[i % len(COLOURS)]
        h1, h2 = arch["h1"], arch["h2"]
        params  = _prunable_params(h1, h2)

        rem = data["remaining"]
        sal = data["saliency"]
        acc = data["accuracy"]

        order = np.argsort(rem)[::-1]
        rem_s, sal_s, acc_s = rem[order], sal[order], acc[order]

        # ── Saliency on left axis ────────────────────────────────────────────
        ax_sal.scatter(rem_s, sal_s, s=MARKER_SIZE, color=colour,
                       alpha=0.80, zorder=5)
        ax_sal.plot(rem_s, sal_s, color=colour, linewidth=1.2,
                    alpha=0.50, zorder=4)
        ax_sal.set_yscale("log")
        ax_sal.set_ylabel(f"{saliency_label} (threshold)", fontsize=AXIS_LABEL_SIZE)
        ax_sal.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax_sal.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)

        # ── Accuracy on right axis (semi-transparent) ────────────────────────
        ax_acc.plot(rem_s, acc_s, color="dimgray", linewidth=1.6,
                    linestyle="--", alpha=0.30, zorder=6)
        ax_acc.scatter(rem_s, acc_s, s=MARKER_SIZE * 0.5, color="dimgray",
                       marker="^", alpha=0.25, zorder=7)
        ax_acc.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_SIZE - 1,
                          color="dimgray")
        ax_acc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE - 1,
                           labelcolor="dimgray")
        ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))

        # ── X axis ──────────────────────────────────────────────────────────
        ax_sal.set_xscale("log")
        ax_sal.set_xlim(rem_s.max() * 1.05, max(rem_s.min() * 0.8, 0.1))
        ax_sal.xaxis.set_major_locator(mticker.FixedLocator(_X_TICKS))
        ax_sal.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
        ax_sal.xaxis.set_minor_locator(mticker.NullLocator())
        ax_sal.set_xlabel("Remaining weights (%,  dense → sparse)",
                          fontsize=AXIS_LABEL_SIZE)
        ax_sal.set_title(f"LeNet ({h1}, {h2})  –  {params:,} params",
                         fontsize=TITLE_FONT_SIZE)
        ax_sal.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, rotation=45)

        legend_handles = [
            Line2D([0], [0], color=colour, marker="o", markersize=5,
                   linestyle="-", linewidth=1.2, label="Saliency (left)"),
            Line2D([0], [0], color="dimgray", marker="^", markersize=5,
                   linestyle="--", linewidth=1.4, alpha=0.4, label="Accuracy (right)"),
        ]
        ax_sal.legend(handles=legend_handles, fontsize=LEGEND_FONT_SIZE - 1,
                      loc="lower right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Remaining Weights vs {saliency_label} & Accuracy  –  Variable LeNet on MNIST",
        fontsize=TITLE_FONT_SIZE + 1, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved remaining vs sal+acc  → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6 – Combined: all architectures, remaining (log x) vs saliency + accuracy
# ---------------------------------------------------------------------------

def plot_remaining_vs_saliency_accuracy_combined(data_dir: str, out_path: str,
                                                 arch_list=None, csv_suffix="_imp",
                                                 saliency_col="SaliencyIMP", saliency_label="Min Saliency"):
    """
    All architectures on one plot.
    X = remaining weights % (log, dense → sparse).
    Y = saliency (log).  No accuracy overlay.
    """
    arch_list = arch_list or ARCHITECTURES
    fig, ax_sal = plt.subplots(figsize=(12, 7))

    plotted = 0
    for i, arch in enumerate(arch_list):
        path = os.path.join(data_dir, f"{arch['name']}{csv_suffix}.csv")
        data = _read_imp_csv(path, saliency_col)
        if data is None:
            continue
        data = _filter_outliers(data)
        colour = COLOURS[i % len(COLOURS)]
        h1, h2 = arch["h1"], arch["h2"]
        label  = f"({h1}, {h2})  –  {_prunable_params(h1, h2):,} params"

        rem = data["remaining"]
        sal = data["saliency"]

        order = np.argsort(rem)[::-1]
        rem_s, sal_s = rem[order], sal[order]

        ax_sal.scatter(rem_s, sal_s, s=MARKER_SIZE, color=colour,
                       alpha=0.80, zorder=5, label=label)
        ax_sal.plot(rem_s, sal_s, color=colour, linewidth=1.2,
                    alpha=0.50, zorder=4)
        plotted += 1

    if plotted == 0:
        print("No data found for combined remaining-vs-saliency plot.")
        plt.close()
        return

    ax_sal.set_xscale("log")
    ax_sal.set_yscale("log")
    ax_sal.invert_xaxis()
    ax_sal.xaxis.set_major_locator(mticker.FixedLocator(_X_TICKS))
    ax_sal.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
    ax_sal.xaxis.set_minor_locator(mticker.NullLocator())
    ax_sal.set_xlabel("Remaining weights (%,  dense → sparse)", fontsize=AXIS_LABEL_SIZE)
    ax_sal.set_ylabel(f"{saliency_label} (threshold)", fontsize=AXIS_LABEL_SIZE)
    ax_sal.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, rotation=45)
    ax_sal.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax_sal.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_sal.set_title(
        f"Remaining Weights vs {saliency_label}  –  All Architectures",
        fontsize=TITLE_FONT_SIZE + 1,
    )
    ax_sal.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right",
                  title="Architecture", title_fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved remaining vs saliency combined → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figures 7 & 8 – accuracy vs parameter count
# ---------------------------------------------------------------------------

def plot_accuracy_vs_params(data_dir: str, out_path_baseline: str, out_path_end: str):
    param_counts  = []
    baseline_accs = []
    end_accs      = []
    labels        = []

    for arch in ARCHITECTURES:
        path = os.path.join(data_dir, f"{arch['name']}_imp.csv")
        data = _read_imp_csv(path)
        if data is None:
            continue
        data = _filter_outliers(data)
        h1, h2 = arch["h1"], arch["h2"]
        param_counts.append(_prunable_params(h1, h2))
        baseline_accs.append(data["accuracy"][0])
        end_accs.append(data["accuracy"][-1])
        labels.append(f"({h1},{h2})")

    if not param_counts:
        print("No data found for accuracy plots.")
        return

    param_counts  = np.array(param_counts)
    baseline_accs = np.array(baseline_accs)
    end_accs      = np.array(end_accs)

    def _make_acc_plot(accs, title, ylabel, out_path):
        fig, ax = plt.subplots(figsize=(9, 5))
        colour = "steelblue"
        ax.scatter(param_counts, accs, s=MARKER_SIZE + 30, color=colour, zorder=5, alpha=0.9)
        ax.plot(param_counts, accs, color=colour, linewidth=1.3, alpha=0.5, zorder=4)
        for x, y, lbl in zip(param_counts, accs, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color="dimgray")
        ax.set_xscale("log")
        ax.set_xlabel("Prunable parameters (log scale)", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE + 1, pad=12)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved accuracy plot        → {out_path}")
        plt.close()

    _make_acc_plot(
        baseline_accs,
        title="Baseline Accuracy vs Network Size  –  Variable LeNet on MNIST",
        ylabel="Accuracy (%) — before pruning",
        out_path=out_path_baseline,
    )
    _make_acc_plot(
        end_accs,
        title="End-of-Pruning Accuracy vs Network Size  –  Variable LeNet on MNIST",
        ylabel="Accuracy (%) — after full IMP",
        out_path=out_path_end,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_series_list(data_dir, arch_list, csv_suffix, saliency_col=None):
    """
    Build a list of NplhSeries from the given architecture list.
    Used to feed the canonical nplh_plots functions.
    """
    series = []
    for arch in arch_list:
        path = os.path.join(data_dir, f"{arch['name']}{csv_suffix}.csv")
        data = _read_imp_csv(path, saliency_col)
        if data is None:
            continue
        data = _filter_outliers(data)
        h1, h2 = arch["h1"], arch["h2"]
        label  = f"LeNet ({h1}, {h2})  –  {_prunable_params(h1, h2):,} params"
        series.append(NplhSeries(
            label=label,
            remaining=data["remaining"],
            saliency=data["saliency"],
            accuracy=data.get("accuracy"),
        ))
    return series


def main(data_dir: str = None):
    """
    Generate all NPLH plots for the variable LeNet experiment.

    Parameters
    ----------
    data_dir : str, optional
        Absolute path to the run folder.  If omitted, defaults to the
        legacy ``neural_pruning_law/final_data/lenet_variable/`` directory
        so old data is still plottable.  For new runs pass
        ``run_ctx.folder_path``.

    Can also be invoked from the command line:
        python -m src.mnist_lenet300.plot_lenet_variable /path/to/run/folder
    """
    if data_dir is None:
        if len(sys.argv) > 1:
            data_dir = sys.argv[1]
        else:
            data_dir = prefix_path_with_root(_LEGACY_DATA_DIR)

    os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Two canonical plots (nplh_plots) – min saliency, all architectures
    # ------------------------------------------------------------------
    min_series = _build_series_list(
        data_dir, ARCHITECTURES, csv_suffix="_imp", saliency_col=None,
    )
    if min_series:
        plot_panels_with_accuracy(
            series_list=min_series,
            out_path=os.path.join(data_dir, "panels_min_saliency.pdf"),
            saliency_label="Min Saliency (threshold)",
            title="NPLH – Min Saliency vs Remaining Weights  (Variable LeNet / MNIST)",
        )
        plot_joint(
            series_list=min_series,
            out_path=os.path.join(data_dir, "joint_min_saliency.pdf"),
            saliency_label="Min Saliency (threshold)",
            title="NPLH – Min Saliency vs Remaining Weights  (all architectures)",
        )

    avg_series = _build_series_list(
        data_dir, ARCHITECTURES_SMALL, csv_suffix="_avg_saliency", saliency_col=None,
    )
    if avg_series:
        plot_panels_with_accuracy(
            series_list=avg_series,
            out_path=os.path.join(data_dir, "panels_avg_saliency.pdf"),
            saliency_label="Avg Saliency",
            title="NPLH – Avg Saliency vs Remaining Weights  (Variable LeNet / MNIST)",
        )
        plot_joint(
            series_list=avg_series,
            out_path=os.path.join(data_dir, "joint_avg_saliency.pdf"),
            saliency_label="Avg Saliency",
            title="NPLH – Avg Saliency vs Remaining Weights  (all architectures)",
        )

    # ------------------------------------------------------------------
    # Specialised per-architecture plots (kept from original script)
    # ------------------------------------------------------------------

    plot_nplh_subplots(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "nplh_all_architectures.pdf"),
    )
    plot_nplh_combined(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "nplh_combined.pdf"),
    )
    plot_nplh_combined_with_accuracy(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "nplh_combined_with_accuracy.pdf"),
    )
    plot_nplh_with_accuracy_regions(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "nplh_accuracy_regions.pdf"),
    )
    plot_remaining_vs_saliency_accuracy(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "remaining_vs_saliency_accuracy.pdf"),
    )
    plot_remaining_vs_saliency_accuracy_combined(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "remaining_vs_saliency_accuracy_combined.pdf"),
    )
    plot_accuracy_vs_params(
        data_dir=data_dir,
        out_path_baseline=os.path.join(data_dir, "accuracy_vs_params_baseline.pdf"),
        out_path_end=os.path.join(data_dir, "accuracy_vs_params_end_pruning.pdf"),
    )

    # ------------------------------------------------------------------
    # Avg-saliency variants  (architectures up to lenet_300_100)
    # ------------------------------------------------------------------
    avg_kwargs = dict(
        arch_list=ARCHITECTURES_SMALL,
        csv_suffix="_avg_saliency",
        saliency_col="SaliencyAvg",
        saliency_label="Avg Saliency",
    )

    plot_nplh_subplots(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "avg_nplh_all_architectures.pdf"),
        **avg_kwargs,
    )
    plot_nplh_combined(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "avg_nplh_combined.pdf"),
        **avg_kwargs,
    )
    plot_nplh_combined_with_accuracy(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "avg_nplh_combined_with_accuracy.pdf"),
        **avg_kwargs,
    )
    plot_nplh_with_accuracy_regions(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "avg_nplh_accuracy_regions.pdf"),
        **avg_kwargs,
    )
    plot_remaining_vs_saliency_accuracy(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "avg_remaining_vs_saliency_accuracy.pdf"),
        **avg_kwargs,
    )
    plot_remaining_vs_saliency_accuracy_combined(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "avg_remaining_vs_saliency_accuracy_combined.pdf"),
        **avg_kwargs,
    )

    # ------------------------------------------------------------------
    # Control group — Min saliency  (all architectures)
    # ------------------------------------------------------------------
    ctrl_min_kwargs = dict(
        arch_list=ARCHITECTURES,
        csv_suffix="_control_imp",
        saliency_col="SaliencyIMP",
        saliency_label="Min Saliency [Control]",
    )

    plot_nplh_subplots(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_min_nplh_all_architectures.pdf"),
        **ctrl_min_kwargs,
    )
    plot_nplh_combined(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_min_nplh_combined.pdf"),
        **ctrl_min_kwargs,
    )
    plot_nplh_combined_with_accuracy(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_min_nplh_combined_with_accuracy.pdf"),
        **ctrl_min_kwargs,
    )
    plot_nplh_with_accuracy_regions(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_min_nplh_accuracy_regions.pdf"),
        **ctrl_min_kwargs,
    )
    plot_remaining_vs_saliency_accuracy(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_min_remaining_vs_saliency_accuracy.pdf"),
        **ctrl_min_kwargs,
    )
    plot_remaining_vs_saliency_accuracy_combined(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_min_remaining_vs_saliency_accuracy_combined.pdf"),
        **ctrl_min_kwargs,
    )

    # ------------------------------------------------------------------
    # Control group — Avg saliency  (all architectures)
    # ------------------------------------------------------------------
    ctrl_avg_kwargs = dict(
        arch_list=ARCHITECTURES,
        csv_suffix="_control_avg",
        saliency_col="SaliencyAvg",
        saliency_label="Avg Saliency [Control]",
    )

    plot_nplh_subplots(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_avg_nplh_all_architectures.pdf"),
        **ctrl_avg_kwargs,
    )
    plot_nplh_combined(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_avg_nplh_combined.pdf"),
        **ctrl_avg_kwargs,
    )
    plot_nplh_combined_with_accuracy(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_avg_nplh_combined_with_accuracy.pdf"),
        **ctrl_avg_kwargs,
    )
    plot_nplh_with_accuracy_regions(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_avg_nplh_accuracy_regions.pdf"),
        **ctrl_avg_kwargs,
    )
    plot_remaining_vs_saliency_accuracy(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_avg_remaining_vs_saliency_accuracy.pdf"),
        **ctrl_avg_kwargs,
    )
    plot_remaining_vs_saliency_accuracy_combined(
        data_dir=data_dir,
        out_path=os.path.join(data_dir, "ctrl_avg_remaining_vs_saliency_accuracy_combined.pdf"),
        **ctrl_avg_kwargs,
    )


if __name__ == "__main__":
    main()
