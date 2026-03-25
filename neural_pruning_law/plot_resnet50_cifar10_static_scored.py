"""
Generate NPLH plots for ResNet50 CIFAR-10 static-scored experiment.

Produces (saved next to the CSVs):
  - per_method_gradient_static.pdf  – min & avg saliency for gradient_static
  - per_method_taylor_static.pdf    – min & avg saliency for taylor_static

Usage (from project root):
    python -m neural_pruning_law.plot_resnet50_cifar10_static_scored
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")

_runs = sorted(d for d in os.listdir(_FINAL_DATA) if "static_scored" in d)
RUN_DIR = os.path.join(_FINAL_DATA, _runs[-1])
print(f"Using run: {_runs[-1]}")


def load(saliency_type: str, method: str) -> pd.DataFrame:
    fname = f"resnet50_cifar10_{saliency_type}_{method}.csv"
    return pd.read_csv(os.path.join(RUN_DIR, fname))


def out(name: str) -> str:
    return os.path.join(RUN_DIR, name)


METHODS = ["gradient_static", "taylor_static"]
METHOD_LABELS = {
    "gradient_static": "Gradient Static",
    "taylor_static":   "Taylor Static",
}
COLORS = {
    "gradient_static": "#9467bd",   # purple
    "taylor_static":   "#d62728",   # red
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
    save_path = out(f"per_method_{method}.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

print("\nAll plots generated.")
