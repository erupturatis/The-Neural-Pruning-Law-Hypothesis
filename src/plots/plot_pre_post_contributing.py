"""
plot_pre_post_contributing.py
=============================
Plots saliency vs contributing and loss/accuracy vs contributing for a single
experiment folder.

Produces plots saved in <folder>/plots/:
  - 5 saliency plots: one per saliency policy
      <slug>_saliency_contributing.png
  - 2 loss plots:
      test_loss_contributing.png
      train_loss_contributing.png
  - 1 accuracy plot:
      accuracy_contributing.png

Each saliency plot: x=contributing %, y=avg_saliency_contributing (log-log).
Each loss/accuracy plot: x=contributing %, y=metric (log-log).

Usage
-----
    python src/plots/plot_pre_post_contributing.py \\
        --folder nplh_data/<run_id>/lenet_random_retrain

    # --out defaults to <folder>/plots/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Style constants
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

SALIENCY_CLASSES = [
    "MagnitudeSaliencyMeasurementPolicy",
    "GradientSaliencyMeasurementPolicy",
    "TaylorSaliencyMeasurementPolicy",
    "HessianSaliencyMeasurementPolicy",
    "NeuronActivationFrequencyPolicy",
]

SHORT_NAME = {
    "MagnitudeSaliencyMeasurementPolicy": "Magnitude",
    "GradientSaliencyMeasurementPolicy":  "Gradient",
    "TaylorSaliencyMeasurementPolicy":    "Taylor",
    "HessianSaliencyMeasurementPolicy":   "Hessian",
    "NeuronActivationFrequencyPolicy":    "NeuronActivFreq",
}

FILE_SLUG = {
    "MagnitudeSaliencyMeasurementPolicy": "magnitude",
    "GradientSaliencyMeasurementPolicy":  "gradient",
    "TaylorSaliencyMeasurementPolicy":    "taylor",
    "HessianSaliencyMeasurementPolicy":   "hessian",
    "NeuronActivationFrequencyPolicy":    "neuron",
}

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _identify_saliency(filename: str) -> str | None:
    for cls in SALIENCY_CLASSES:
        if cls in filename:
            return cls
    return None


def _scan_folder(folder: str) -> dict[str, str]:
    """Return {saliency_class: csv_path} for each CSV found in folder."""
    csv_map: dict[str, str] = {}
    for csv_file in sorted(Path(folder).glob("*.csv")):
        cls = _identify_saliency(csv_file.name)
        if cls is None:
            continue
        csv_map[cls] = str(csv_file)
    return csv_map


def _read_xy(csv_path: str, x_col: str, y_col: str,
             x_min: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read (x, y) pairs, filtering out non-positive values, sorted dense→sparse."""
    xs, ys = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            x_s = row.get(x_col, "").strip()
            y_s = row.get(y_col, "").strip()
            if not x_s or not y_s:
                continue
            try:
                x, y = float(x_s), float(y_s)
            except ValueError:
                continue
            if x > 0 and y > 0 and (x_min is None or x >= x_min):
                xs.append(x)
                ys.append(y)
    if not xs:
        return np.array([]), np.array([])
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    order = np.argsort(xs_np)[::-1]   # dense → sparse
    return xs_np[order], ys_np[order]


# ---------------------------------------------------------------------------
# Shared axis styling
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes, title: str, x_label: str, y_label: str,
              x_right: float | None = None) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    visible = [t for t in _X_TICKS if t <= 100]
    ax.set_xticks(visible)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    if x_right is not None:
        ax.set_xlim(right=x_right)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_one_saliency(
    csv_path: str,
    cls: str,
    out_path: str,
    experiment_label: str,
    x_min: float | None = None,
) -> None:
    xs, ys = _read_xy(csv_path, x_col="contributing", y_col="avg_saliency_contributing", x_min=x_min)
    if xs.size == 0:
        print(f"  [plot] WARNING: no valid data for {SHORT_NAME[cls]}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colour = COLOURS[0]
    ax.plot(xs, ys, color=colour, linewidth=1.5, alpha=0.85)
    ax.scatter(xs, ys, color=colour, s=MARKER_SIZE, zorder=3, label=SHORT_NAME[cls])

    _style_ax(ax,
              title=f"{experiment_label}  |  {SHORT_NAME[cls]} saliency vs Contributing",
              x_label="Contributing weights (%)",
              y_label="Avg saliency (contributing)",
              x_right=x_min)
    ax.legend(fontsize=LEGEND_FONT_SIZE, framealpha=0.85)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  [plot] saved → {out_path}")
    plt.close(fig)


def _plot_one_scalar(
    csv_path: str,
    y_col: str,
    y_label: str,
    out_path: str,
    experiment_label: str,
    x_min: float | None = None,
) -> None:
    xs, ys = _read_xy(csv_path, x_col="contributing", y_col=y_col, x_min=x_min)
    if xs.size == 0:
        print(f"  [plot] WARNING: no valid data for {y_col}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colour = COLOURS[0]
    ax.plot(xs, ys, color=colour, linewidth=1.5, alpha=0.85)
    ax.scatter(xs, ys, color=colour, s=MARKER_SIZE, zorder=3, label=y_label)

    _style_ax(ax,
              title=f"{experiment_label}  |  {y_label} vs Contributing",
              x_label="Contributing weights (%)",
              y_label=y_label,
              x_right=x_min)
    ax.legend(fontsize=LEGEND_FONT_SIZE, framealpha=0.85)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  [plot] saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folder", required=True,
                        help="Experiment folder containing CSVs")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <folder>/plots/)")
    parser.add_argument("--x_min", type=float, default=None,
                        help="Only plot points where contributing >= this value (e.g. 2.0)")
    args = parser.parse_args()

    folder = args.folder
    out_dir = args.out or str(Path(folder) / "plots")
    experiment_label = Path(folder).name
    x_min = args.x_min

    print(f"\n[plot] folder         : {folder}")
    print(f"[plot] output dir     : {out_dir}")
    print(f"[plot] experiment     : {experiment_label}")
    if x_min is not None:
        print(f"[plot] x_min          : {x_min}%")
    print()

    csv_map = _scan_folder(folder)
    print(f"[plot] found {len(csv_map)} CSV(s): {[FILE_SLUG[c] for c in csv_map]}\n")

    if not csv_map:
        print("[plot] ERROR: no CSVs found. Exiting.")
        sys.exit(1)

    # pick any CSV for scalar metrics (all share the same density/contributing/accuracy/loss columns)
    rep_csv = next(iter(csv_map.values()))

    # ── saliency plots ────────────────────────────────────────────────────────
    print("[plot] Generating saliency vs contributing plots...")
    for cls in SALIENCY_CLASSES:
        if cls not in csv_map:
            continue
        slug = FILE_SLUG[cls]
        _plot_one_saliency(
            csv_map[cls], cls,
            out_path=os.path.join(out_dir, f"{slug}_saliency_contributing.png"),
            experiment_label=experiment_label,
            x_min=x_min,
        )

    # ── loss plots ────────────────────────────────────────────────────────────
    print("\n[plot] Generating loss vs contributing plots...")
    for loss_col, loss_label in [("test_loss", "Test loss"), ("train_loss", "Train loss")]:
        _plot_one_scalar(
            rep_csv, loss_col, loss_label,
            out_path=os.path.join(out_dir, f"{loss_col}_contributing.png"),
            experiment_label=experiment_label,
            x_min=x_min,
        )

    # ── accuracy plot ─────────────────────────────────────────────────────────
    print("\n[plot] Generating accuracy vs contributing plot...")
    _plot_one_scalar(
        rep_csv, "accuracy", "Accuracy (%)",
        out_path=os.path.join(out_dir, "accuracy_contributing.png"),
        experiment_label=experiment_label,
        x_min=x_min,
    )

    n = len(csv_map) + 3  # saliency + test_loss + train_loss + accuracy
    print(f"\n[plot] done. {n} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
