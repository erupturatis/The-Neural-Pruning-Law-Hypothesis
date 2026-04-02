"""
Plot static vs retrain comparison for one pruning method.

Produces 6 images:
  - One per saliency measurement (5 total): static line vs retrain line.
  - One combined plot with all 10 lines (5 saliencies × 2 conditions).

Usage
-----
    python src/plots/plot_static_vs_retrain.py \\
        --static  nplh_data/run_id/lenet_random_static \\
        --retrain nplh_data/run_id/lenet_random_retrain \\
        --out     plots_output/random_pruning

    # --out defaults to a 'plots/' subfolder next to the static folder.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# ── Saliency class name → short display name ──────────────────────────────────

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

# One colour per saliency type (used in the combined plot).
_COLOURS = plt.get_cmap("tab10").colors
SALIENCY_COLOUR = {cls: _COLOURS[i] for i, cls in enumerate(SALIENCY_CLASSES)}

# X-axis tick positions (%).
_X_TICKS = [
    0.05, 0.1, 0.2, 0.4, 0.5,
    1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95,
]


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _identify_saliency(filename: str) -> str | None:
    """Return the saliency class name embedded in *filename*, or None."""
    for cls in SALIENCY_CLASSES:
        if cls in filename:
            return cls
    return None


def _scan_folder(folder: str) -> dict[str, str]:
    """
    Return {saliency_class: csv_path} for all recognised CSVs in *folder*.
    If multiple CSVs share the same saliency class, the last one found is used
    (filenames are sorted so the most recent timestamp wins).
    """
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


def _read_xy(csv_path: str, x_col: str = "density",
             y_col: str = "avg_saliency") -> tuple[np.ndarray, np.ndarray]:
    """
    Read (x, y) pairs from a CSV.  Rows where either value is <= 0 are dropped
    (incompatible with log scale).  Returns arrays sorted dense → sparse
    (descending x).
    """
    xs, ys = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            x_raw = row.get(x_col, "").strip()
            y_raw = row.get(y_col, "").strip()
            if not x_raw or not y_raw:
                continue
            try:
                x, y = float(x_raw), float(y_raw)
            except ValueError:
                continue
            if x > 0 and y > 0:
                xs.append(x)
                ys.append(y)
    if not xs:
        return np.array([]), np.array([])
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    order = np.argsort(xs_np)[::-1]   # dense → sparse
    return xs_np[order], ys_np[order]


# ── Shared axis styling ────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str, x_label: str = "Remaining weights (%)") -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    visible = [t for t in _X_TICKS if t <= 100]
    ax.set_xticks(visible)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Avg saliency", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)


# ── Per-saliency plot (static vs retrain) ────────────────────────────────────

def _plot_one_saliency(
    cls: str,
    static_csv: str | None,
    retrain_csv: str | None,
    out_path: str,
    pruning_label: str,
) -> None:
    short = SHORT_NAME[cls]
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted = False

    if static_csv:
        xs, ys = _read_xy(static_csv)
        if xs.size:
            ax.plot(xs, ys, color="steelblue", linewidth=1.8, linestyle="-")
            ax.scatter(xs, ys, color="steelblue", s=28, zorder=3, label="Static")
            plotted = True
        else:
            print(f"  [plot] WARNING: no plottable data in static CSV for {short}")

    if retrain_csv:
        xs, ys = _read_xy(retrain_csv)
        if xs.size:
            ax.plot(xs, ys, color="tomato", linewidth=1.8, linestyle="--")
            ax.scatter(xs, ys, color="tomato", s=28, zorder=3,
                       marker="^", label="Retrain")
            plotted = True
        else:
            print(f"  [plot] WARNING: no plottable data in retrain CSV for {short}")

    if not plotted:
        print(f"  [plot] skipping {short} — no valid data in either CSV")
        plt.close(fig)
        return

    _style_ax(ax, f"{pruning_label}  |  {short} saliency")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  [plot] saved → {out_path}")
    plt.close(fig)


# ── Combined plot (all saliencies, static + retrain) ─────────────────────────

def _plot_combined(
    static_map:  dict[str, str],
    retrain_map: dict[str, str],
    out_path: str,
    pruning_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    legend_handles: list[Line2D] = []
    plotted_any = False

    for cls in SALIENCY_CLASSES:
        short  = SHORT_NAME[cls]
        colour = SALIENCY_COLOUR[cls]

        static_csv  = static_map.get(cls)
        retrain_csv = retrain_map.get(cls)

        has_static = has_retrain = False

        if static_csv:
            xs, ys = _read_xy(static_csv)
            if xs.size:
                ax.plot(xs, ys, color=colour, linewidth=1.5, linestyle="-",  alpha=0.9)
                ax.scatter(xs, ys, color=colour, s=22, zorder=3)
                has_static = True
                plotted_any = True

        if retrain_csv:
            xs, ys = _read_xy(retrain_csv)
            if xs.size:
                ax.plot(xs, ys, color=colour, linewidth=1.5, linestyle="--", alpha=0.9)
                ax.scatter(xs, ys, color=colour, s=22, zorder=3, marker="^")
                has_retrain = True
                plotted_any = True

        if has_static or has_retrain:
            conditions = []
            if has_static:  conditions.append("static")
            if has_retrain: conditions.append("retrain")
            legend_handles.append(
                Line2D([0], [0], color=colour, linewidth=2,
                       label=f"{short}  ({', '.join(conditions)})")
            )

    if not plotted_any:
        print("  [plot] combined: no valid data found, skipping")
        plt.close(fig)
        return

    # Extra legend entries to explain line style
    style_handles = [
        Line2D([0], [0], color="grey", linewidth=2, linestyle="-",  label="— static"),
        Line2D([0], [0], color="grey", linewidth=2, linestyle="--", label="-- retrain"),
    ]
    ax.legend(handles=legend_handles + style_handles,
              fontsize=8.5, framealpha=0.85, ncol=2)

    _style_ax(ax, f"{pruning_label}  |  all saliency metrics")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  [plot] saved → {out_path}")
    plt.close(fig)


# ── Combined plot — each saliency normalised to its own max ──────────────────

def _plot_combined_normalised(
    static_map:  dict[str, str],
    retrain_map: dict[str, str],
    out_path: str,
    pruning_label: str,
) -> None:
    """
    Same as _plot_combined but each saliency type's Y values are divided by
    the maximum value across both static and retrain for that type.  All curves
    therefore end up in (0, 1] regardless of their absolute scale, making the
    shapes directly comparable on the same axes.  Legend is placed above the plot.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    legend_handles: list[Line2D] = []
    plotted_any = False

    for cls in SALIENCY_CLASSES:
        short  = SHORT_NAME[cls]
        colour = SALIENCY_COLOUR[cls]

        static_csv  = static_map.get(cls)
        retrain_csv = retrain_map.get(cls)

        # Read both series first so we can normalise by a shared max.
        static_xs,  static_ys  = _read_xy(static_csv)  if static_csv  else (np.array([]), np.array([]))
        retrain_xs, retrain_ys = _read_xy(retrain_csv) if retrain_csv else (np.array([]), np.array([]))

        all_ys = np.concatenate([static_ys, retrain_ys])
        if all_ys.size == 0:
            continue
        norm = all_ys.max()
        if norm == 0:
            continue

        has_static = has_retrain = False

        if static_xs.size:
            ys_norm = static_ys / norm
            ax.plot(static_xs, ys_norm, color=colour, linewidth=1.5,
                    linestyle="-", alpha=0.9)
            ax.scatter(static_xs, ys_norm, color=colour, s=22, zorder=3)
            has_static = True
            plotted_any = True

        if retrain_xs.size:
            ys_norm = retrain_ys / norm
            ax.plot(retrain_xs, ys_norm, color=colour, linewidth=1.5,
                    linestyle="--", alpha=0.9)
            ax.scatter(retrain_xs, ys_norm, color=colour, s=22, zorder=3,
                       marker="^")
            has_retrain = True
            plotted_any = True

        if has_static or has_retrain:
            conditions = []
            if has_static:  conditions.append("static")
            if has_retrain: conditions.append("retrain")
            legend_handles.append(
                Line2D([0], [0], color=colour, linewidth=2,
                       label=f"{short}  ({', '.join(conditions)})")
            )

    if not plotted_any:
        print("  [plot] combined-normalised: no valid data found, skipping")
        plt.close(fig)
        return

    style_handles = [
        Line2D([0], [0], color="grey", linewidth=2, linestyle="-",  label="— static"),
        Line2D([0], [0], color="grey", linewidth=2, linestyle="--", label="-- retrain"),
    ]
    all_handles = legend_handles + style_handles

    # Legend above the plot area
    ax.legend(
        handles=all_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        fontsize=8.5,
        framealpha=0.85,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    visible = [t for t in _X_TICKS if t <= 100]
    ax.set_xticks(visible)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Remaining weights (%)", fontsize=12)
    ax.set_ylabel("Normalised avg saliency  (each metric scaled to its own max)",
                  fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_title(f"{pruning_label}  |  all saliency metrics  (normalised)",
                 fontsize=13, pad=60)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.subplots_adjust(top=0.78)   # extra headroom for the top legend
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  [plot] saved → {out_path}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--static",  required=True,
                        help="Folder containing the static experiment CSVs")
    parser.add_argument("--retrain", required=True,
                        help="Folder containing the retrain experiment CSVs")
    parser.add_argument("--out", default=None,
                        help="Output directory for images (default: plots/ next to --static)")
    args = parser.parse_args()

    static_folder  = args.static
    retrain_folder = args.retrain

    # Default output directory: plots/ sibling of the static folder
    if args.out is None:
        out_dir = str(Path(static_folder).parent / "plots")
    else:
        out_dir = args.out

    # Derive a human-readable pruning label from the folder names
    static_name  = Path(static_folder).name   # e.g. "lenet_random_static"
    retrain_name = Path(retrain_folder).name  # e.g. "lenet_random_retrain"
    # Strip trailing _static / _retrain to get pruning method label
    pruning_label = static_name.removesuffix("_static").removesuffix("_retrain")

    print(f"\n[plot] static  folder : {static_folder}")
    print(f"[plot] retrain folder : {retrain_folder}")
    print(f"[plot] output dir     : {out_dir}")
    print(f"[plot] pruning label  : {pruning_label}\n")

    static_map  = _scan_folder(static_folder)
    retrain_map = _scan_folder(retrain_folder)

    if not static_map and not retrain_map:
        print("[plot] ERROR: no CSVs found in either folder. Exiting.")
        sys.exit(1)

    print(f"[plot] found {len(static_map)} static CSV(s), "
          f"{len(retrain_map)} retrain CSV(s)\n")

    # ── 5 individual per-saliency plots ───────────────────────────────────────
    for cls in SALIENCY_CLASSES:
        slug = FILE_SLUG[cls]
        out_path = os.path.join(out_dir, f"{slug}_static_vs_retrain.png")
        _plot_one_saliency(
            cls=cls,
            static_csv=static_map.get(cls),
            retrain_csv=retrain_map.get(cls),
            out_path=out_path,
            pruning_label=pruning_label,
        )

    # ── 1 combined plot ───────────────────────────────────────────────────────
    _plot_combined(
        static_map=static_map,
        retrain_map=retrain_map,
        out_path=os.path.join(out_dir, "all_combined.png"),
        pruning_label=pruning_label,
    )

    # ── 1 combined plot — normalised per saliency ─────────────────────────────
    _plot_combined_normalised(
        static_map=static_map,
        retrain_map=retrain_map,
        out_path=os.path.join(out_dir, "all_combined_normalised.png"),
        pruning_label=pruning_label,
    )

    print("\n[plot] done.")


if __name__ == "__main__":
    main()
