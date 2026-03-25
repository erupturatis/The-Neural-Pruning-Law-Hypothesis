"""
plot_resnet50_cifar10_multi_saliency.py
========================================
Plots for the multi-saliency random-pruning experiment (ResNet50 CIFAR-10).

One PDF per saliency metric, each showing avg saliency + accuracy on twin
axes vs remaining parameters (log-log, x inverted).

Outputs:
  per_method_random_prune_magnitude.pdf
  per_method_random_prune_taylor.pdf
  per_method_random_prune_gradient.pdf

Usage (from project root):
    python -m neural_pruning_law.plot_resnet50_cifar10_multi_saliency
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── locate run folder ─────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")

_runs = sorted(d for d in os.listdir(_FINAL_DATA) if "multi_saliency" in d)
if not _runs:
    raise FileNotFoundError(
        f"No 'multi_saliency' run folder found in {_FINAL_DATA}."
    )
RUN_DIR = os.path.join(_FINAL_DATA, _runs[-1])
print(f"Using run: {_runs[-1]}")

# ── helpers ───────────────────────────────────────────────────────────────────

def out(name: str) -> str:
    return os.path.join(RUN_DIR, name)


_PCT_FMT = ticker.FuncFormatter(lambda x, _: f"{x:g}")
_FIG_W, _FIG_H = 8, 5
_COLLAPSE_THRESHOLD = 15.0

METRICS = {
    "random_prune_magnitude": {
        "label":      "Avg |w|",
        "color":      "#1f77b4",   # blue
        "title":      "ResNet50 CIFAR-10 – Random Pruning + Training: Magnitude Saliency (avg |w|)",
        "pdf":        "per_method_random_prune_magnitude.pdf",
    },
    "random_prune_taylor": {
        "label":      "Avg |w · ∂L/∂w|",
        "color":      "#9467bd",   # purple
        "title":      "ResNet50 CIFAR-10 – Random Pruning + Training: Taylor Saliency (avg |w·∂L/∂w|)",
        "pdf":        "per_method_random_prune_taylor.pdf",
    },
    "random_prune_gradient": {
        "label":      "Avg |∂L/∂w|",
        "color":      "#d62728",   # red
        "title":      "ResNet50 CIFAR-10 – Random Pruning + Training: Gradient Saliency (avg |∂L/∂w|)",
        "pdf":        "per_method_random_prune_gradient.pdf",
    },
}


def _find_collapse_density(df: pd.DataFrame) -> float | None:
    if "Accuracy" not in df.columns:
        return None
    hit = df[df["Accuracy"] < _COLLAPSE_THRESHOLD]
    return float(hit.iloc[0]["RemainingParams"]) if not hit.empty else None


for method_tag, cfg in METRICS.items():
    fname = f"resnet50_cifar10_avg_{method_tag}.csv"
    fpath = os.path.join(RUN_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [skip] {fname} not found")
        continue

    df = pd.read_csv(fpath)

    fig, ax1 = plt.subplots(figsize=(_FIG_W, _FIG_H))

    ax1.plot(df["RemainingParams"], df["Saliency"],
             color=cfg["color"], linewidth=1.8, label=cfg["label"])

    # Collapse vline
    cd = _find_collapse_density(df)
    if cd is not None:
        ax1.axvline(x=cd, color="red", ls=":", linewidth=1.4,
                    label=f"Collapse at {cd:.4f}%")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.xaxis.set_major_formatter(_PCT_FMT)
    ax1.set_xlabel("Remaining Parameters (%)", fontsize=11)
    ax1.set_ylabel(cfg["label"], fontsize=11, color=cfg["color"])
    ax1.tick_params(axis="both", labelsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    if "Accuracy" in df.columns:
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

    ax1.set_title(cfg["title"], fontsize=11)
    fig.tight_layout()
    fig.savefig(out(cfg["pdf"]), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out(cfg['pdf'])}")

print("\nAll plots generated.")
