"""
plot_lenet300_experiment.py
===========================
Plots IMP-magnitude and IMP-Taylor results for the LeNet300 experiment.

Reads the two min-saliency CSVs and two avg-saliency CSVs from a run folder
and produces four PDF plots using the canonical nplh_plots functions.

Run from the project root, passing the run folder as an argument:
    python -m src.mnist_lenet300.plot_lenet300_experiment <run_folder>

If no argument is given it looks for the most recent
``*_lenet300_all_policies`` folder under neural_pruning_law/final_data/.
"""

import os
import sys
import csv
import glob
import numpy as np

from src.infrastructure.nplh_run_context import (
    COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
    SAL_MIN, SAL_AVG,
    METHOD_IMP_MAGNITUDE, METHOD_IMP_TAYLOR,
)
from src.infrastructure.others import prefix_path_with_root
from src.plots.nplh_plots import NplhSeries, plot_panels_with_accuracy, plot_joint

ARCH = "lenet_300_100"
DATASET = "mnist"


def _read_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Read (remaining, saliency, accuracy) arrays from a standardised NPLH CSV."""
    if not os.path.isfile(path):
        return None
    remaining, saliency, accuracy = [], [], []
    has_acc = False
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rem = float(row[COL_REMAINING])
                sal = float(row[COL_SALIENCY])
            except (ValueError, KeyError):
                continue
            if sal <= 0 or rem <= 0:
                continue
            remaining.append(rem)
            saliency.append(sal)
            if COL_ACCURACY in row:
                try:
                    accuracy.append(float(row[COL_ACCURACY]))
                    has_acc = True
                except ValueError:
                    accuracy.append(float("nan"))
    if len(remaining) < 2:
        return None
    acc_arr = np.array(accuracy) if has_acc else None
    return np.array(remaining), np.array(saliency), acc_arr


def _find_latest_run_folder() -> str | None:
    base = prefix_path_with_root("neural_pruning_law/final_data")
    matches = sorted(glob.glob(os.path.join(base, "*_lenet300_all_policies")))
    return matches[-1] if matches else None


def plot(run_folder: str) -> None:
    methods = [
        (METHOD_IMP_MAGNITUDE, "IMP – Magnitude"),
        (METHOD_IMP_TAYLOR,    "IMP – Taylor"),
    ]

    # Build series for min and avg saliency separately
    min_series = []
    avg_series = []

    for method_tag, label in methods:
        min_path = os.path.join(run_folder, f"{ARCH}_{DATASET}_{SAL_MIN}_{method_tag}.csv")
        avg_path = os.path.join(run_folder, f"{ARCH}_{DATASET}_{SAL_AVG}_{method_tag}.csv")

        min_data = _read_csv(min_path)
        avg_data = _read_csv(avg_path)

        if min_data is not None:
            rem, sal, acc = min_data
            min_series.append(NplhSeries(label=label, remaining=rem, saliency=sal, accuracy=acc))
        else:
            print(f"[plot] Missing min-saliency CSV for {method_tag}: {min_path}")

        if avg_data is not None:
            rem, sal, acc = avg_data
            avg_series.append(NplhSeries(label=label, remaining=rem, saliency=sal, accuracy=acc))
        else:
            print(f"[plot] Missing avg-saliency CSV for {method_tag}: {avg_path}")

    # --- Min saliency (threshold) plots ---
    if min_series:
        plot_panels_with_accuracy(
            series_list=min_series,
            out_path=os.path.join(run_folder, "lenet300_min_saliency_panels.pdf"),
            saliency_label="Min Saliency (threshold)",
            title="LeNet300 MNIST – IMP: Min Saliency vs Remaining Weights",
            ncols=2,
        )
        plot_joint(
            series_list=min_series,
            out_path=os.path.join(run_folder, "lenet300_min_saliency_joint.pdf"),
            saliency_label="Min Saliency (threshold)",
            title="LeNet300 MNIST – IMP: Min Saliency (Magnitude vs Taylor)",
        )

    # --- Avg saliency plots ---
    if avg_series:
        plot_panels_with_accuracy(
            series_list=avg_series,
            out_path=os.path.join(run_folder, "lenet300_avg_saliency_panels.pdf"),
            saliency_label="Avg Saliency",
            title="LeNet300 MNIST – IMP: Avg Saliency vs Remaining Weights",
            ncols=2,
        )
        plot_joint(
            series_list=avg_series,
            out_path=os.path.join(run_folder, "lenet300_avg_saliency_joint.pdf"),
            saliency_label="Avg Saliency",
            title="LeNet300 MNIST – IMP: Avg Saliency (Magnitude vs Taylor)",
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
