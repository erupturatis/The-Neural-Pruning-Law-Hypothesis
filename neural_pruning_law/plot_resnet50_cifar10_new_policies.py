"""
plot_resnet50_cifar10_new_policies.py
======================================
Generate NPLH diagrams for the gradient and random_regrowth experiments
(ResNet50 CIFAR-10, new-policies run).

Outputs (saved next to the CSVs in the run folder):
  per_method_gradient.pdf          – min & avg saliency for gradient
  per_method_random_regrowth.pdf   – min & avg saliency for random_regrowth
  joint_min_saliency.pdf           – both methods, min saliency
  joint_avg_saliency.pdf           – both methods, avg saliency
  accuracy_comparison.pdf          – test accuracy vs remaining params, both methods

Usage (from project root):
    python -m neural_pruning_law.plot_resnet50_cifar10_new_policies
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── locate run folder ────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")

_runs = sorted(d for d in os.listdir(_FINAL_DATA) if "new_policies" in d)
if not _runs:
    raise FileNotFoundError(
        f"No 'new_policies' run folder found in {_FINAL_DATA}. "
        "Run run_resnet50_cifar10_new_policies.py first."
    )
RUN_DIR = os.path.join(_FINAL_DATA, _runs[-1])
print(f"Using run: {_runs[-1]}")

# ── helpers ──────────────────────────────────────────────────────────────────

def load(saliency_type: str, method: str) -> pd.DataFrame:
    fname = f"resnet50_cifar10_{saliency_type}_{method}.csv"
    return pd.read_csv(os.path.join(RUN_DIR, fname))


def out(name: str) -> str:
    return os.path.join(RUN_DIR, name)


METHODS = ["gradient", "random_regrowth"]
METHOD_LABELS = {
    "gradient":       "Gradient (|∂L/∂w|)",
    "random_regrowth": "Random Regrowth",
}
COLORS = {
    "gradient":        "#9467bd",   # purple
    "random_regrowth": "#d62728",   # red
}
SAL_STYLES = {
    "min": {"ls": "-",  "label_suffix": " – min saliency"},
    "avg": {"ls": "--", "label_suffix": " – avg saliency"},
}

_FIG_W, _FIG_H = 8, 5


def _setup_saliency_axes(ax, title: str):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Remaining Parameters (%)", fontsize=11)
    ax.set_ylabel("Saliency", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


def _setup_accuracy_axes(ax, title: str):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Remaining Parameters (%)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.axhline(y=10.0, color="gray", ls=":", linewidth=1.2, label="Random chance (10%)")
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


# ── 1. Per-method plots (min + avg saliency) ─────────────────────────────────

for method in METHODS:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for sal in ("min", "avg"):
        df = load(sal, method)
        style = SAL_STYLES[sal]
        ax.plot(
            df["RemainingParams"], df["Saliency"],
            color=COLORS[method], ls=style["ls"], linewidth=1.8,
            label=METHOD_LABELS[method] + style["label_suffix"],
        )
    _setup_saliency_axes(ax, f"ResNet50 CIFAR-10 – {METHOD_LABELS[method]}")
    fig.tight_layout()
    save_path = out(f"per_method_{method}.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── 2. Joint min saliency ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load("min", method)
    ax.plot(
        df["RemainingParams"], df["Saliency"],
        color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method],
    )
_setup_saliency_axes(ax, "ResNet50 CIFAR-10 – Min Saliency (gradient vs random_regrowth)")
fig.tight_layout()
save_path = out("joint_min_saliency.pdf")
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")


# ── 3. Joint avg saliency ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load("avg", method)
    ax.plot(
        df["RemainingParams"], df["Saliency"],
        color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method],
    )
_setup_saliency_axes(ax, "ResNet50 CIFAR-10 – Avg Saliency (gradient vs random_regrowth)")
fig.tight_layout()
save_path = out("joint_avg_saliency.pdf")
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")


# ── 4. Accuracy comparison ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load("min", method)   # accuracy is identical in min/avg CSVs
    if "Accuracy" in df.columns:
        ax.plot(
            df["RemainingParams"], df["Accuracy"],
            color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method],
        )
_setup_accuracy_axes(ax, "ResNet50 CIFAR-10 – Accuracy vs Density (gradient vs random_regrowth)")
fig.tight_layout()
save_path = out("accuracy_comparison.pdf")
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")

print("\nAll plots generated.")
