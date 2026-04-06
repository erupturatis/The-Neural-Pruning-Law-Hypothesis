"""
Plot retrain-only results using contributing weights as x-axis.

Produces 8 images per run:
  - 5 per-saliency plots (avg_saliency_contributing vs contributing density)
  - 3 metric plots: accuracy, test_loss, train_loss vs contributing density

X-axis is cut at 1% (data below 1% contributing density is not shown).
No static comparison, no background metrics.

Usage
-----
    python src/plots/plot_static_vs_retrain.py \\
        --retrain nplh_data/run_id/lenet_random_retrain \\
        --out     plots_output/lenet_random_retrain

    # --out defaults to a 'plots/' subfolder inside --retrain.
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


# ── Saliency class name → display / file slug ─────────────────────────────────

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

_COLOURS = plt.get_cmap("tab10").colors
SALIENCY_COLOUR = {cls: _COLOURS[i] for i, cls in enumerate(SALIENCY_CLASSES)}

_METRIC_LABEL = {
    "accuracy":   "Accuracy (%)",
    "test_loss":  "Test loss",
    "train_loss": "Train loss",
}
_METRIC_COLOUR = {
    "accuracy":   "forestgreen",
    "test_loss":  "mediumpurple",
    "train_loss": "chocolate",
}

# X-axis ticks shown.
_X_TICKS = [0.05, 0.1, 0.2, 0.4, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]

X_MIN = 0.0   # no cut-off: show full range


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _identify_saliency(filename: str) -> str | None:
    for cls in SALIENCY_CLASSES:
        if cls in filename:
            return cls
    return None


def _scan_folder(folder: str) -> dict[str, str]:
    result: dict[str, str] = {}
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"  [plot] ERROR: folder not found: {folder}")
        return result
    for csv_file in sorted(folder_path.glob("*.csv")):
        cls = _identify_saliency(csv_file.name)
        if cls:
            result[cls] = str(csv_file)
    return result


def _read_csv(csv_path: str,
              x_col: str = "contributing",
              y_col: str = "avg_saliency_contributing") -> dict:
    """
    Read (x, y) and metric columns from a CSV, filtering x < X_MIN.
    Returns:
      xs, ys          — saliency series (x >= X_MIN, y > 0, dense→sparse)
      bg_xs           — x values for metrics (x >= X_MIN, dense→sparse)
      accuracy, test_loss, train_loss — parallel to bg_xs
    """
    def _f(s: str) -> float:
        try:
            return float(s) if s else float("nan")
        except ValueError:
            return float("nan")

    xs_raw, ys_raw = [], []
    bg_xs_raw: list[float] = []
    acc_raw, tl_raw, trl_raw = [], [], []

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            x_str = row.get(x_col, "").strip()
            if not x_str:
                continue
            try:
                x = float(x_str)
            except ValueError:
                continue
            if x <= 0:
                continue

            bg_xs_raw.append(x)
            acc_raw.append(_f(row.get("accuracy", "")))
            tl_raw.append(_f(row.get("test_loss", "")))
            trl_raw.append(_f(row.get("train_loss", "")))

            y_str = row.get(y_col, "").strip()
            try:
                y = float(y_str)
                if y > 0:
                    xs_raw.append(x)
                    ys_raw.append(y)
            except ValueError:
                pass

    if bg_xs_raw:
        order = np.argsort(np.array(bg_xs_raw))[::-1]
        bg_xs   = np.array(bg_xs_raw)[order]
        acc_np  = np.array(acc_raw)[order]
        tl_np   = np.array(tl_raw)[order]
        trl_np  = np.array(trl_raw)[order]
    else:
        bg_xs = acc_np = tl_np = trl_np = np.array([])

    if xs_raw:
        order = np.argsort(np.array(xs_raw))[::-1]
        xs = np.array(xs_raw)[order]
        ys = np.array(ys_raw)[order]
    else:
        xs = ys = np.array([])

    return {
        "xs": xs, "ys": ys,
        "bg_xs": bg_xs,
        "accuracy": acc_np, "test_loss": tl_np, "train_loss": trl_np,
    }


# ── Shared axis styling ────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str, y_label: str, x_label: str,
              log_y: bool = True) -> None:
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlim(left=100)
    ax.set_xticks(_X_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)


# ── Per-saliency plot ─────────────────────────────────────────────────────────

def _plot_one_saliency(
    cls: str,
    retrain_csv: str,
    out_path: str,
    pruning_label: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> None:
    short  = SHORT_NAME[cls]
    colour = SALIENCY_COLOUR[cls]

    data = _read_csv(retrain_csv, x_col=x_col, y_col=y_col)
    if not data["xs"].size:
        print(f"  [plot] skipping {short} — no valid data")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(data["xs"], data["ys"], color=colour, linewidth=1.8)
    ax.scatter(data["xs"], data["ys"], color=colour, s=28, zorder=3)
    _style_ax(ax, f"{pruning_label}  |  {short} saliency",
              y_label=y_label, x_label=x_label)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  [plot] saved → {out_path}")
    plt.close(fig)


# ── Per-metric plot ───────────────────────────────────────────────────────────

def _plot_one_metric(
    metric: str,
    retrain_map: dict[str, str],
    out_path: str,
    pruning_label: str,
    x_col: str,
    x_label: str,
) -> None:
    rep_csv = next(iter(retrain_map.values()), None)
    if rep_csv is None:
        print(f"  [plot] skipping {metric} — no CSV available")
        return

    data = _read_csv(rep_csv, x_col=x_col)
    ys_all = data[metric]
    mask = ~np.isnan(ys_all)
    if not mask.any():
        print(f"  [plot] skipping {metric} — all NaN")
        return

    xs = data["bg_xs"][mask]
    ys = ys_all[mask]
    colour = _METRIC_COLOUR[metric]
    y_label = _METRIC_LABEL[metric]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs, ys, color=colour, linewidth=1.8)
    ax.scatter(xs, ys, color=colour, s=28, zorder=3)
    _style_ax(ax, f"{pruning_label}  |  {y_label}",
              y_label=y_label, x_label=x_label, log_y=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  [plot] saved → {out_path}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--retrain", required=True,
                        help="Folder containing the retrain experiment CSVs")
    parser.add_argument("--out", default=None,
                        help="Output directory for images (default: plots/ inside --retrain)")
    parser.add_argument("--mode", choices=["contributing", "density"],
                        default="contributing",
                        help="Which x/y columns to use (default: contributing)")
    args = parser.parse_args()

    retrain_folder = args.retrain
    out_dir = args.out if args.out else str(Path(retrain_folder) / "plots")

    if args.mode == "contributing":
        x_col   = "contributing"
        y_col   = "avg_saliency_contributing"
        x_label = "Contributing weights (%)"
        y_label = "Avg saliency (contributing)"
        suffix  = ""
    else:
        x_col   = "density"
        y_col   = "avg_saliency"
        x_label = "Remaining weights (%)"
        y_label = "Avg saliency"
        suffix  = "_density"

    pruning_label = Path(retrain_folder).name.removesuffix("_retrain")

    print(f"\n[plot] retrain folder : {retrain_folder}")
    print(f"[plot] output dir     : {out_dir}")
    print(f"[plot] pruning label  : {pruning_label}")
    print(f"[plot] mode           : {args.mode}\n")

    retrain_map = _scan_folder(retrain_folder)
    if not retrain_map:
        print("[plot] ERROR: no CSVs found. Exiting.")
        sys.exit(1)

    print(f"[plot] found {len(retrain_map)} retrain CSV(s)\n")

    for cls in SALIENCY_CLASSES:
        csv_path = retrain_map.get(cls)
        if not csv_path:
            print(f"  [plot] WARNING: no CSV for {SHORT_NAME[cls]}, skipping")
            continue
        slug = FILE_SLUG[cls]
        _plot_one_saliency(
            cls=cls,
            retrain_csv=csv_path,
            out_path=os.path.join(out_dir, f"{slug}{suffix}.png"),
            pruning_label=pruning_label,
            x_col=x_col,
            y_col=y_col,
            x_label=x_label,
            y_label=y_label,
        )

    for metric in ("accuracy", "test_loss", "train_loss"):
        _plot_one_metric(
            metric=metric,
            retrain_map=retrain_map,
            out_path=os.path.join(out_dir, f"{metric}{suffix}.png"),
            pruning_label=pruning_label,
            x_col=x_col,
            x_label=x_label,
        )

    print("\n[plot] done. 8 plots saved.")


if __name__ == "__main__":
    main()
