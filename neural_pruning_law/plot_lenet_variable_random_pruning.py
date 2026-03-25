"""
plot_lenet_variable_random_pruning.py
=======================================
Plots for the variable-LeNet random-pruning intrinsic-saliency experiment.

Per-architecture PDFs (10 total):
  Each shows avg |w| for the trained experiment (blue) and the static
  control (green) on a shared log-log axis, with accuracy on a twin axis
  for the trained experiment.

Joint PDF (1):
  All 10 trained avg |w| curves on one axis, coloured from cool (small /
  under-parameterized) to warm (large / over-parameterized), with the
  parameter count in the legend.

Outputs (inside the run folder):
  per_arch_lenet_{h1}_{h2}.pdf      × 10
  joint_trained_avg_saliency.pdf

Usage (from project root):
    python -m neural_pruning_law.plot_lenet_variable_random_pruning
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np

# ── locate run folder ─────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")

_runs = sorted(d for d in os.listdir(_FINAL_DATA) if "lenet_variable_random_pruning" in d)
if not _runs:
    raise FileNotFoundError(
        f"No 'lenet_variable_random_pruning' run folder found in {_FINAL_DATA}."
    )
RUN_DIR = os.path.join(_FINAL_DATA, _runs[-1])
print(f"Using run: {_runs[-1]}")

# ── architecture list (same order as the runner) ──────────────────────────────
ARCHITECTURES = [
    {"h1":    4, "h2":    2},
    {"h1":   10, "h2":    5},
    {"h1":   20, "h2":   10},
    {"h1":   50, "h2":   25},
    {"h1":  100, "h2":   50},
    {"h1":  300, "h2":  100},
    {"h1":  600, "h2":  200},
    {"h1": 1200, "h2":  400},
    {"h1": 2500, "h2":  800},
    {"h1": 5000, "h2": 1500},
]

# ── helpers ───────────────────────────────────────────────────────────────────
_PCT_FMT          = ticker.FuncFormatter(lambda x, _: f"{x:g}")
_FIG_W, _FIG_H    = 8, 5
_COLLAPSE_THRESHOLD = 15.0
_COLOR_TRAINED    = "#1f77b4"   # blue
_COLOR_STATIC     = "#2ca02c"   # green


def out(name: str) -> str:
    return os.path.join(RUN_DIR, name)


def _csv(arch_name: str, method: str) -> str:
    return os.path.join(RUN_DIR, f"{arch_name}_mnist_avg_{method}.csv")


def _find_collapse(df: pd.DataFrame) -> float | None:
    if "Accuracy" not in df.columns:
        return None
    hit = df[df["Accuracy"] < _COLLAPSE_THRESHOLD]
    return float(hit.iloc[0]["RemainingParams"]) if not hit.empty else None


# ── 1. Per-architecture PDFs ──────────────────────────────────────────────────

for arch in ARCHITECTURES:
    h1, h2    = arch["h1"], arch["h2"]
    arch_name = f"lenet_{h1}_{h2}"
    prunable  = 784 * h1 + h1 * h2 + h2 * 10

    p_trained = _csv(arch_name, "random_prune_trained")
    p_static  = _csv(arch_name, "random_prune_static")

    if not os.path.exists(p_trained):
        print(f"  [skip] {arch_name}: trained CSV not found")
        continue

    df_trained = pd.read_csv(p_trained)
    df_static  = pd.read_csv(p_static) if os.path.exists(p_static) else None

    fig, ax1 = plt.subplots(figsize=(_FIG_W, _FIG_H))

    # Trained saliency
    ax1.plot(df_trained["RemainingParams"], df_trained["Saliency"],
             color=_COLOR_TRAINED, linewidth=1.8,
             label="Avg |w| — random prune + training")

    # Static saliency
    if df_static is not None:
        ax1.plot(df_static["RemainingParams"], df_static["Saliency"],
                 color=_COLOR_STATIC, linewidth=1.8, ls="--",
                 label="Avg |w| — static (no retraining)")

    # Collapse vline (from trained experiment)
    cd = _find_collapse(df_trained)
    if cd is not None:
        ax1.axvline(x=cd, color="red", ls=":", linewidth=1.4,
                    label=f"Collapse at {cd:.4f}%")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.xaxis.set_major_formatter(_PCT_FMT)
    ax1.set_xlabel("Remaining Parameters (%)", fontsize=11)
    ax1.set_ylabel("Avg |w|", fontsize=11)
    ax1.tick_params(axis="both", labelsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # Accuracy twin axis (trained only)
    if "Accuracy" in df_trained.columns:
        ax2 = ax1.twinx()
        ax2.plot(df_trained["RemainingParams"], df_trained["Accuracy"],
                 color="darkorange", linewidth=1.8, ls=":",
                 label="Accuracy (trained)")
        ax2.axhline(y=10.0, color="gray", ls=":", linewidth=1.1)
        ax2.axhline(y=_COLLAPSE_THRESHOLD, color="salmon", ls=":", linewidth=1.1)
        ax2.set_ylabel("Test Accuracy (%)", fontsize=11, color="darkorange")
        ax2.tick_params(axis="y", labelsize=9)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        ax1.legend(fontsize=9)

    ax1.set_title(
        f"LeNet ({h1}, {h2}) – {prunable:,} params – Trained vs Static Random Pruning",
        fontsize=11,
    )
    fig.tight_layout()
    pdf_path = out(f"per_arch_{arch_name}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


# ── 2. Joint plot — all trained saliencies ────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

# Colour ramp: cool (small) → warm (large)
cmap   = cm.colormaps["coolwarm"].resampled(len(ARCHITECTURES))
colors = [cmap(i) for i in range(len(ARCHITECTURES))]

for i, arch in enumerate(ARCHITECTURES):
    h1, h2    = arch["h1"], arch["h2"]
    arch_name = f"lenet_{h1}_{h2}"
    prunable  = 784 * h1 + h1 * h2 + h2 * 10

    p_trained = _csv(arch_name, "random_prune_trained")
    if not os.path.exists(p_trained):
        continue

    df = pd.read_csv(p_trained)
    label = f"({h1},{h2})  {prunable:>11,} params"
    ax.plot(df["RemainingParams"], df["Saliency"],
            color=colors[i], linewidth=1.6, label=label)

ax.set_title(
    "Variable LeNet – Avg |w| during Random Pruning + Training (all architectures)",
    fontsize=12,
)
ax.set_xlabel("Remaining Parameters (%)", fontsize=11)
ax.set_ylabel("Avg |w|", fontsize=11)
ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_xaxis()
ax.xaxis.set_major_formatter(_PCT_FMT)
ax.tick_params(axis="both", labelsize=9)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper left", title="Architecture   Params",
          title_fontsize=8)
fig.tight_layout()
joint_path = out("joint_trained_avg_saliency.pdf")
fig.savefig(joint_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {joint_path}")

print("\nAll plots generated.")
