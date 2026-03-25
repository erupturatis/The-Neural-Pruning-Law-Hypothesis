"""
plot_resnet50_cifar10_collapse.py
==================================
Generate diagrams for the collapse experiments on ResNet50 CIFAR-10.

Outputs (saved next to the CSVs in the run folder):
  per_method_collapse_imp.pdf      – min & avg saliency for collapse IMP
  per_method_collapse_static.pdf   – min & avg saliency for collapse static
  joint_min_saliency.pdf           – both methods, min saliency
  joint_avg_saliency.pdf           – both methods, avg saliency
  accuracy_comparison.pdf          – accuracy vs density, both methods
  saliency_vs_accuracy.pdf         – saliency (min) and accuracy on twin axes
                                     to show where collapse occurs

Usage (from project root):
    python -m neural_pruning_law.plot_resnet50_cifar10_collapse
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── locate run folder ────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")

_runs = sorted(d for d in os.listdir(_FINAL_DATA) if "collapse" in d)
if not _runs:
    raise FileNotFoundError(
        f"No 'collapse' run folder found in {_FINAL_DATA}. "
        "Run run_resnet50_cifar10_collapse_experiment.py first."
    )
RUN_DIR = os.path.join(_FINAL_DATA, _runs[-1])
print(f"Using run: {_runs[-1]}")

# ── helpers ──────────────────────────────────────────────────────────────────

def load(saliency_type: str, method: str) -> pd.DataFrame:
    fname = f"resnet50_cifar10_{saliency_type}_{method}.csv"
    return pd.read_csv(os.path.join(RUN_DIR, fname))


def out(name: str) -> str:
    return os.path.join(RUN_DIR, name)


METHODS = ["collapse_imp", "collapse_static"]
METHOD_LABELS = {
    "collapse_imp":    "Collapse IMP (with fine-tuning)",
    "collapse_static": "Collapse Static (no retraining)",
}
COLORS = {
    "collapse_imp":    "#1f77b4",  # blue
    "collapse_static": "#2ca02c",  # green
}
SAL_STYLES = {
    "min": {"ls": "-",  "label_suffix": " – min saliency"},
    "avg": {"ls": "--", "label_suffix": " – avg saliency"},
}

_FIG_W, _FIG_H = 8, 5


def _find_collapse_density(df: pd.DataFrame, threshold: float = 15.0) -> float | None:
    """Return the RemainingParams value where accuracy first drops below threshold."""
    collapsed = df[df["Accuracy"] < threshold]
    if collapsed.empty:
        return None
    return float(collapsed.iloc[0]["RemainingParams"])


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


def _add_collapse_vline(ax, density: float | None):
    if density is not None:
        ax.axvline(x=density, color="red", ls=":", linewidth=1.4,
                   label=f"Collapse at {density:.4f}%")


# ── 1. Per-method saliency plots ─────────────────────────────────────────────

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
    # Mark collapse point from min csv (has accuracy)
    df_min = load("min", method)
    if "Accuracy" in df_min.columns:
        collapse_d = _find_collapse_density(df_min)
        _add_collapse_vline(ax, collapse_d)
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
_setup_saliency_axes(ax, "ResNet50 CIFAR-10 – Min Saliency: IMP vs Static (to collapse)")
fig.tight_layout()
fig.savefig(out("joint_min_saliency.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out('joint_min_saliency.pdf')}")


# ── 3. Joint avg saliency ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load("avg", method)
    ax.plot(
        df["RemainingParams"], df["Saliency"],
        color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method],
    )
_setup_saliency_axes(ax, "ResNet50 CIFAR-10 – Avg Saliency: IMP vs Static (to collapse)")
fig.tight_layout()
fig.savefig(out("joint_avg_saliency.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out('joint_avg_saliency.pdf')}")


# ── 4. Accuracy comparison ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
for method in METHODS:
    df = load("min", method)
    if "Accuracy" in df.columns:
        ax.plot(
            df["RemainingParams"], df["Accuracy"],
            color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method],
        )
ax.set_title("ResNet50 CIFAR-10 – Accuracy vs Density (collapse experiments)", fontsize=13)
ax.set_xlabel("Remaining Parameters (%)", fontsize=11)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_xscale("log")
ax.invert_xaxis()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.axhline(y=10.0, color="gray", ls=":", linewidth=1.2, label="Random chance (10%)")
ax.axhline(y=15.0, color="salmon", ls=":", linewidth=1.2, label="Collapse threshold (15%)")
ax.tick_params(axis="both", labelsize=9)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(out("accuracy_comparison.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out('accuracy_comparison.pdf')}")


# ── 5. Twin-axis: min saliency + accuracy for collapse IMP ───────────────────
# Shows the saliency trajectory and accuracy on the same x-axis so the
# relationship between saliency divergence and collapse is visible.

df_imp = load("min", "collapse_imp")
if "Accuracy" in df_imp.columns:
    fig, ax1 = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax2 = ax1.twinx()

    ax1.plot(df_imp["RemainingParams"], df_imp["Saliency"],
             color=COLORS["collapse_imp"], linewidth=1.8, label="Min saliency")
    ax2.plot(df_imp["RemainingParams"], df_imp["Accuracy"],
             color="darkorange", linewidth=1.8, ls="--", label="Accuracy")

    collapse_d = _find_collapse_density(df_imp)
    if collapse_d is not None:
        ax1.axvline(x=collapse_d, color="red", ls=":", linewidth=1.4,
                    label=f"Collapse at {collapse_d:.4f}%")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax1.set_xlabel("Remaining Parameters (%)", fontsize=11)
    ax1.set_ylabel("Min Saliency", fontsize=11, color=COLORS["collapse_imp"])
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11, color="darkorange")
    ax1.tick_params(axis="both", labelsize=9)
    ax2.tick_params(axis="y", labelsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.set_title("ResNet50 CIFAR-10 – Saliency & Accuracy (Collapse IMP)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out("saliency_vs_accuracy_imp.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out('saliency_vs_accuracy_imp.pdf')}")

print("\nAll plots generated.")
