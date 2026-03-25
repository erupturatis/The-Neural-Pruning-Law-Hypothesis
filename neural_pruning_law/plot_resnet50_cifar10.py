"""
Generate NPLH plots for ResNet50 CIFAR-10 experiment.

Produces (saved next to the CSVs):
  - per_method_magnitude.pdf   – min & avg saliency for IMP_magnitude
  - per_method_taylor.pdf      – min & avg saliency for IMP_taylor
  - per_method_static.pdf      – min & avg saliency for IMP_static
  - joint_min_saliency.pdf     – all 3 methods, min saliency on one axes
  - joint_avg_saliency.pdf     – all 3 methods, avg saliency on one axes
  - overlap_static_magnitude.pdf – static vs magnitude (both min and avg)

Usage (from project root):
    python -m neural_pruning_law.plot_resnet50_cifar10
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── locate the run folder ────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")

# Pick the latest resnet50_cifar10 run
_runs = sorted(
    d for d in os.listdir(_FINAL_DATA) if "resnet50_cifar10" in d
)
RUN_DIR = os.path.join(_FINAL_DATA, _runs[-1])
print(f"Using run: {_runs[-1]}")

# ── helpers ──────────────────────────────────────────────────────────────────

def load(saliency_type: str, method: str) -> pd.DataFrame:
    fname = f"resnet50_cifar10_{saliency_type}_{method}.csv"
    return pd.read_csv(os.path.join(RUN_DIR, fname))


def out(name: str) -> str:
    path = os.path.join(RUN_DIR, name)
    return path


METHODS = ["IMP_magnitude", "IMP_taylor", "IMP_static"]
METHOD_LABELS = {
    "IMP_magnitude": "IMP Magnitude",
    "IMP_taylor":    "IMP Taylor",
    "IMP_static":    "IMP Static",
}
COLORS = {
    "IMP_magnitude": "#1f77b4",   # blue
    "IMP_taylor":    "#ff7f0e",   # orange
    "IMP_static":    "#2ca02c",   # green
}
SAL_STYLES = {
    "min": {"ls": "-",  "label_suffix": " (min saliency)"},
    "avg": {"ls": "--", "label_suffix": " (avg saliency)"},
}

_FIG_W, _FIG_H = 8, 5


def _setup_axes(ax, title: str):
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


# ── 1. Per-method plots ───────────────────────────────────────────────────────

for method in METHODS:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for sal in ("min", "avg"):
        df = load(sal, method)
        style = SAL_STYLES[sal]
        ax.plot(
            df["RemainingParams"],
            df["Saliency"],
            color=COLORS[method],
            ls=style["ls"],
            linewidth=1.8,
            label=METHOD_LABELS[method] + style["label_suffix"],
        )
    _setup_axes(ax, f"ResNet50 CIFAR-10 – {METHOD_LABELS[method]}")
    fig.tight_layout()
    save_path = out(f"per_method_{method.lower().replace('imp_', '')}.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── 2. Joint min saliency (all methods) ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load("min", method)
    ax.plot(
        df["RemainingParams"],
        df["Saliency"],
        color=COLORS[method],
        linewidth=1.8,
        label=METHOD_LABELS[method],
    )
_setup_axes(ax, "ResNet50 CIFAR-10 – Min Saliency (all methods)")
fig.tight_layout()
save_path = out("joint_min_saliency.pdf")
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")


# ── 3. Joint avg saliency (all methods) ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load("avg", method)
    ax.plot(
        df["RemainingParams"],
        df["Saliency"],
        color=COLORS[method],
        linewidth=1.8,
        label=METHOD_LABELS[method],
    )
_setup_axes(ax, "ResNet50 CIFAR-10 – Avg Saliency (all methods)")
fig.tight_layout()
save_path = out("joint_avg_saliency.pdf")
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")


# ── 4. Overlap: static vs magnitude (both min and avg) ──────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in ("IMP_static", "IMP_magnitude"):
    for sal in ("min", "avg"):
        df = load(sal, method)
        style = SAL_STYLES[sal]
        ax.plot(
            df["RemainingParams"],
            df["Saliency"],
            color=COLORS[method],
            ls=style["ls"],
            linewidth=1.8,
            label=METHOD_LABELS[method] + style["label_suffix"],
        )
_setup_axes(ax, "ResNet50 CIFAR-10 – Static vs Magnitude Saliency Overlap")
fig.tight_layout()
save_path = out("overlap_static_magnitude.pdf")
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")

print("\nAll plots generated.")
