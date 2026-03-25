"""
plot_static_imp_resnet50_imagenet.py
=====================================
Plots min and avg saliency vs remaining parameters for the static IMP
experiment on pretrained ResNet50 (ImageNet1k).

Uses the two canonical nplh_plots functions:
  - plot_panels_with_accuracy  (one panel per series, no accuracy here)
  - plot_joint                 (both metrics on one log-log plot)

Run from the project root, passing the run folder as an argument:
    python -m src.resnet50_imagenet1k.plot_static_imp_resnet50_imagenet <run_folder>

If no argument is given it looks for the most recent run folder matching
``*_resnet50_imagenet_static`` under neural_pruning_law/final_data/.
"""

import os
import sys
import csv
import glob
import numpy as np

from src.infrastructure.nplh_run_context import (
    COL_STEP, COL_REMAINING, COL_SALIENCY,
    SAL_MIN, SAL_AVG, METHOD_IMP_STATIC,
)
from src.infrastructure.others import prefix_path_with_root
from src.plots.nplh_plots import NplhSeries, plot_panels_with_accuracy, plot_joint


def _read_csv(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read (remaining, saliency) arrays from a standardised NPLH CSV."""
    if not os.path.isfile(path):
        return None
    remaining, saliency = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sal = float(row[COL_SALIENCY])
                rem = float(row[COL_REMAINING])
            except (ValueError, KeyError):
                continue
            if sal <= 0 or rem <= 0:
                continue
            saliency.append(sal)
            remaining.append(rem)
    if len(remaining) < 2:
        return None
    return np.array(remaining), np.array(saliency)


def _find_latest_run_folder() -> str | None:
    base = prefix_path_with_root("neural_pruning_law/final_data")
    matches = sorted(glob.glob(os.path.join(base, "*_resnet50_imagenet_static")))
    return matches[-1] if matches else None


def plot(run_folder: str) -> None:
    min_path = os.path.join(run_folder, f"resnet50_imagenet_{SAL_MIN}_{METHOD_IMP_STATIC}.csv")
    avg_path = os.path.join(run_folder, f"resnet50_imagenet_{SAL_AVG}_{METHOD_IMP_STATIC}.csv")

    min_data = _read_csv(min_path)
    avg_data = _read_csv(avg_path)

    if min_data is None and avg_data is None:
        print(f"No CSV data found in {run_folder}")
        return

    series_list = []
    if min_data is not None:
        rem, sal = min_data
        series_list.append(NplhSeries(
            label="Min saliency (threshold)",
            remaining=rem, saliency=sal, accuracy=None,
        ))
    if avg_data is not None:
        rem, sal = avg_data
        series_list.append(NplhSeries(
            label="Avg saliency",
            remaining=rem, saliency=sal, accuracy=None,
        ))

    # Panel plot (one panel per metric)
    plot_panels_with_accuracy(
        series_list=series_list,
        out_path=os.path.join(run_folder, "panels_saliency.pdf"),
        saliency_label="Saliency",
        title="ResNet50 ImageNet – static IMP",
        ncols=2,
    )

    # Joint plot (both metrics overlaid)
    plot_joint(
        series_list=series_list,
        out_path=os.path.join(run_folder, "joint_saliency.pdf"),
        saliency_label="Saliency",
        title="ResNet50 ImageNet – static IMP: Min vs Avg Saliency",
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = _find_latest_run_folder()
        if folder is None:
            print("No run folder found. Pass the run folder path as an argument.")
            sys.exit(1)
        print(f"Using most recent run folder: {folder}")

    plot(folder)
