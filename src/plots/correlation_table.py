"""
Compute a 5×5 Pearson correlation table between avg_saliency_contributing
for all saliency policies in one experiment folder.

Threshold: only rows where accuracy >= 75% of the peak accuracy in the
experiment are included (i.e. network has not yet degraded significantly).

Usage
-----
    python src/plots/correlation_table.py --folder nplh_data/.../lenet_random_retrain
    # saves correlation_table.txt inside the folder
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SALIENCY_CLASSES = [
    "MagnitudeSaliencyMeasurementPolicy",
    "GradientSaliencyMeasurementPolicy",
    "TaylorSaliencyMeasurementPolicy",
    "HessianSaliencyMeasurementPolicy",
    "NeuronActivationFrequencyPolicy",
]

SHORT = {
    "MagnitudeSaliencyMeasurementPolicy": "Magnitude",
    "GradientSaliencyMeasurementPolicy":  "Gradient",
    "TaylorSaliencyMeasurementPolicy":    "Taylor",
    "HessianSaliencyMeasurementPolicy":   "Hessian",
    "NeuronActivationFrequencyPolicy":    "NeuronAF",
}

ACCURACY_RETENTION = 0.75   # keep rows where accuracy >= this fraction of peak


def _find_csv(folder: Path, saliency_cls: str) -> Path | None:
    matches = sorted(folder.glob(f"*{saliency_cls}*.csv"))
    return matches[-1] if matches else None


def _load_full(csv_path: Path) -> list[dict]:
    """Return list of {contributing, avg_saliency_contributing, accuracy} dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                contrib = float(row["contributing"])
                val     = float(row["avg_saliency_contributing"])
                acc     = float(row["accuracy"])
            except (KeyError, ValueError):
                continue
            if val > 0:
                rows.append({"contributing": contrib, "saliency": val, "accuracy": acc})
    return rows


def compute_correlation_table(folder: str) -> None:
    folder_path = Path(folder)

    # Load raw rows for each saliency class
    raw: dict[str, list[dict]] = {}
    for cls in SALIENCY_CLASSES:
        csv_path = _find_csv(folder_path, cls)
        if csv_path is None:
            print(f"  [corr] WARNING: no CSV for {SHORT[cls]}, skipping folder")
            return
        raw[cls] = _load_full(csv_path)

    # Peak accuracy: max across any saliency CSV (all share the same accuracy column)
    all_accs = [r["accuracy"] for r in raw[SALIENCY_CLASSES[0]]]
    if not all_accs:
        print("  [corr] WARNING: no accuracy data found, skipping")
        return
    peak_acc = max(all_accs)
    acc_threshold = ACCURACY_RETENTION * peak_acc

    # Build per-class lookup: contributing -> saliency, filtered by accuracy threshold
    series: dict[str, dict[float, float]] = {}
    for cls in SALIENCY_CLASSES:
        series[cls] = {
            round(r["contributing"], 6): r["saliency"]
            for r in raw[cls]
            if r["accuracy"] >= acc_threshold
        }

    # Intersect contributing keys present in ALL series
    common_keys = set(series[SALIENCY_CLASSES[0]].keys())
    for cls in SALIENCY_CLASSES[1:]:
        common_keys &= set(series[cls].keys())

    common_keys_sorted = sorted(common_keys, reverse=True)
    n = len(common_keys_sorted)

    if n < 3:
        print(f"  [corr] WARNING: only {n} data points after accuracy filter — skipping")
        return

    print(f"  [corr] peak_acc={peak_acc:.2f}%  threshold={acc_threshold:.2f}%  →  {n} data points retained")

    # Build matrix and compute Pearson correlation
    mat = np.array([
        [series[cls][k] for cls in SALIENCY_CLASSES]
        for k in common_keys_sorted
    ])  # shape: (n, 5)

    corr = np.corrcoef(mat.T)  # shape: (5, 5)

    # Format table
    names = [SHORT[cls] for cls in SALIENCY_CLASSES]
    col_w = 10

    lines: list[str] = []
    lines.append("Pearson correlation of avg_saliency_contributing")
    lines.append(f"Folder    : {folder_path.name}")
    lines.append(f"Filter    : accuracy >= {ACCURACY_RETENTION*100:.0f}% of peak  "
                 f"(peak={peak_acc:.2f}%,  threshold={acc_threshold:.2f}%,  n={n})")
    lines.append("")

    header = " " * 10 + "".join(f"{name:>{col_w}}" for name in names)
    lines.append(header)
    lines.append("-" * len(header))

    for i, row_name in enumerate(names):
        row_str = f"{row_name:<10}" + "".join(f"{corr[i, j]:>{col_w}.4f}" for j in range(5))
        lines.append(row_str)

    table_text = "\n".join(lines) + "\n"

    out_path = folder_path / "correlation_table.txt"
    out_path.write_text(table_text)
    print(f"  [corr] saved → {out_path}")
    print()
    print(table_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True,
                        help="Experiment folder containing saliency CSVs")
    args = parser.parse_args()
    compute_correlation_table(args.folder)


if __name__ == "__main__":
    main()
