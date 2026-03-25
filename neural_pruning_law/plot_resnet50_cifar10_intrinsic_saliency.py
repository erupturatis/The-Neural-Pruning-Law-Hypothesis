"""
plot_resnet50_cifar10_intrinsic_saliency.py
============================================
Plots for the intrinsic saliency experiment (ResNet50 CIFAR-10).

Outputs:
  per_method_random_prune_trained.pdf  – avg |w| saliency + accuracy for trained
  per_method_random_prune_static.pdf   – avg |w| saliency + accuracy for static
  joint_avg_saliency.pdf               – both methods, avg |w| on one axes
  saliency_vs_accuracy.pdf             – twin-axis: avg |w| and accuracy together
                                         for trained experiment

Usage (from project root):
    python -m neural_pruning_law.plot_resnet50_cifar10_intrinsic_saliency
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── locate run folder ────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")

_runs = sorted(d for d in os.listdir(_FINAL_DATA) if "intrinsic_saliency" in d)
if not _runs:
    raise FileNotFoundError(
        f"No 'intrinsic_saliency' run folder found in {_FINAL_DATA}."
    )
RUN_DIR = os.path.join(_FINAL_DATA, _runs[-1])
print(f"Using run: {_runs[-1]}")

# ── helpers ──────────────────────────────────────────────────────────────────

def load(method: str) -> pd.DataFrame:
    fname = f"resnet50_cifar10_avg_{method}.csv"
    return pd.read_csv(os.path.join(RUN_DIR, fname))


def out(name: str) -> str:
    return os.path.join(RUN_DIR, name)


METHODS = ["random_prune_trained", "random_prune_static"]
METHOD_LABELS = {
    "random_prune_trained": "Random pruning + training",
    "random_prune_static":  "Random pruning static (no retraining)",
}
COLORS = {
    "random_prune_trained": "#1f77b4",  # blue
    "random_prune_static":  "#2ca02c",  # green
}

_FIG_W, _FIG_H = 8, 5
_COLLAPSE_THRESHOLD = 15.0
_PCT_FMT = ticker.FuncFormatter(lambda x, _: f"{x:g}")


def _find_collapse_density(df: pd.DataFrame) -> float | None:
    if "Accuracy" not in df.columns:
        return None
    hit = df[df["Accuracy"] < _COLLAPSE_THRESHOLD]
    return float(hit.iloc[0]["RemainingParams"]) if not hit.empty else None


def _add_collapse_vline(ax, df):
    d = _find_collapse_density(df)
    if d is not None:
        ax.axvline(x=d, color="red", ls=":", linewidth=1.4,
                   label=f"Collapse at {d:.4f}%")


# ── 1. Per-method: avg saliency + accuracy on twin axes ──────────────────────

for method in METHODS:
    df = load(method)
    has_acc = "Accuracy" in df.columns

    fig, ax1 = plt.subplots(figsize=(_FIG_W, _FIG_H))

    ax1.plot(df["RemainingParams"], df["Saliency"],
             color=COLORS[method], linewidth=1.8, label="Avg |w| saliency")
    _add_collapse_vline(ax1, df)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.xaxis.set_major_formatter(_PCT_FMT)
    ax1.set_xlabel("Remaining Parameters (%)", fontsize=11)
    ax1.set_ylabel("Avg |w|", fontsize=11, color=COLORS[method])
    ax1.tick_params(axis="both", labelsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    if has_acc:
        ax2 = ax1.twinx()
        ax2.plot(df["RemainingParams"], df["Accuracy"],
                 color="darkorange", linewidth=1.8, ls="--", label="Accuracy")
        ax2.axhline(y=10.0, color="gray", ls=":", linewidth=1.1)
        ax2.axhline(y=_COLLAPSE_THRESHOLD, color="salmon", ls=":", linewidth=1.1)
        ax2.set_ylabel("Test Accuracy (%)", fontsize=11, color="darkorange")
        ax2.tick_params(axis="y", labelsize=9)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        ax1.legend(fontsize=9)

    ax1.set_title(f"ResNet50 CIFAR-10 – {METHOD_LABELS[method]}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out(f"per_method_{method}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out(f'per_method_{method}.pdf')}")


# ── 2. Joint avg saliency — both methods on one axes ─────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load(method)
    ax.plot(df["RemainingParams"], df["Saliency"],
            color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])

ax.set_title("ResNet50 CIFAR-10 – Avg |w| Saliency: trained vs static random pruning",
             fontsize=12)
ax.set_xlabel("Remaining Parameters (%)", fontsize=11)
ax.set_ylabel("Avg |w|", fontsize=11)
ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_xaxis()
ax.xaxis.set_major_formatter(_PCT_FMT)
ax.tick_params(axis="both", labelsize=9)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(out("joint_avg_saliency.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out('joint_avg_saliency.pdf')}")


# ── 3. Accuracy comparison ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load(method)
    if "Accuracy" in df.columns:
        ax.plot(df["RemainingParams"], df["Accuracy"],
                color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])

ax.axhline(y=10.0, color="gray", ls=":", linewidth=1.2, label="Random chance (10%)")
ax.axhline(y=_COLLAPSE_THRESHOLD, color="salmon", ls=":", linewidth=1.2,
           label=f"Collapse threshold ({_COLLAPSE_THRESHOLD}%)")
ax.set_title("ResNet50 CIFAR-10 – Accuracy: trained vs static random pruning", fontsize=12)
ax.set_xlabel("Remaining Parameters (%)", fontsize=11)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_xscale("log")
ax.invert_xaxis()
ax.xaxis.set_major_formatter(_PCT_FMT)
ax.tick_params(axis="both", labelsize=9)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(out("accuracy_comparison.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out('accuracy_comparison.pdf')}")

print("\nAll plots generated.")
