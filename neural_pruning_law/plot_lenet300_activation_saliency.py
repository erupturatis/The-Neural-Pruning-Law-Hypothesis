"""
plot_lenet300_activation_saliency.py
======================================
Plots magnitude saliency and inverted APoZ (firing rate = 1 - APoZ) for
the LeNet-300-100 random-pruning experiment on a log-log scale.

Both metrics are measured before each pruning step:
  - Magnitude saliency : mean |w| of active weights (left y-axis, blue)
  - Firing rate        : 1 - APoZ = fraction of batch where active neurons
                         fire (right y-axis, orange)

X-axis: Remaining Parameters (%), log scale, decreasing left → right.
Y-axes: log scale.

Auto-picks the latest lenet300_activation_saliency run folder.

Output: lenet300_magnitude_vs_firing_rate.pdf  (inside the run folder)

Run from project root:
    python -m neural_pruning_law.plot_lenet300_activation_saliency
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── locate run folder ──────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")

_runs = sorted(
    d for d in os.listdir(_FINAL_DATA)
    if "lenet300_activation_saliency" in d
)
if not _runs:
    raise FileNotFoundError(
        f"No 'lenet300_activation_saliency' run folder found in {_FINAL_DATA}.\n"
        "Run the experiment first:\n"
        "  python -m src.mnist_lenet300.run_lenet300_activation_saliency"
    )
RUN_DIR = os.path.join(_FINAL_DATA, _runs[-1])
print(f"Using run: {_runs[-1]}")

# ── CSV paths ──────────────────────────────────────────────────────────────────
ARCH      = "lenet_300_100"
DATASET   = "mnist"
_sal_type = "avg"

csv_mag  = os.path.join(RUN_DIR, f"{ARCH}_{DATASET}_{_sal_type}_random_retrain_magnitude.csv")
csv_apoz = os.path.join(RUN_DIR, f"{ARCH}_{DATASET}_{_sal_type}_random_retrain_apoz.csv")

for path in (csv_mag, csv_apoz):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

df_mag  = pd.read_csv(csv_mag)
df_apoz = pd.read_csv(csv_apoz)

# Inverted APoZ = firing rate (fraction of batch where neuron fires)
df_apoz["FiringRate"] = 1.0 - df_apoz["Saliency"]

# ── plot ───────────────────────────────────────────────────────────────────────
_PCT_FMT = ticker.FuncFormatter(lambda x, _: f"{x:g}")

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis — magnitude saliency
color_mag = "#1f77b4"
ax1.plot(
    df_mag["RemainingParams"], df_mag["Saliency"],
    color=color_mag, linewidth=2.0,
    label="Magnitude saliency  (mean |w|)",
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.invert_xaxis()
ax1.xaxis.set_major_formatter(_PCT_FMT)
ax1.set_xlabel("Remaining Parameters (%)", fontsize=11)
ax1.set_ylabel("Mean |w|  (log scale)", fontsize=11, color=color_mag)
ax1.tick_params(axis="y", labelcolor=color_mag, labelsize=9)
ax1.tick_params(axis="x", labelsize=9)
ax1.grid(True, which="both", alpha=0.25)

# Right axis — firing rate (1 - APoZ)
color_fr = "#ff7f0e"
ax2 = ax1.twinx()
ax2.plot(
    df_apoz["RemainingParams"], df_apoz["FiringRate"],
    color=color_fr, linewidth=2.0, linestyle="--",
    label="Firing rate  (1 − APoZ)",
)
ax2.set_yscale("log")
ax2.set_ylabel("Firing rate  (1 − APoZ,  log scale)", fontsize=11, color=color_fr)
ax2.tick_params(axis="y", labelcolor=color_fr, labelsize=9)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")

ax1.set_title(
    "LeNet-300-100  —  Magnitude vs Firing Rate (1 − APoZ)\n"
    "Random pruning + fine-tuning  |  log-log scale",
    fontsize=11,
)

fig.tight_layout()
out_path = os.path.join(RUN_DIR, "lenet300_magnitude_vs_firing_rate.pdf")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
