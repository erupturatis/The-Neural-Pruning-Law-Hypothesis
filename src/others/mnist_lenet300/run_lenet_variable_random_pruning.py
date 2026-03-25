"""
run_lenet_variable_random_pruning.py
=====================================
Runner for the intrinsic-saliency random-pruning experiment on 10
variable-width LeNet architectures (MNIST).

For each architecture:
  A. Train 25 epochs → random prune 10% + 3-epoch fine-tuning → repeat
     until accuracy collapses below 15%.  Record avg |w| at each step.
  B. Train 25 epochs (fresh) → random prune 10% static (no retraining) →
     down to the collapse density from A.  Record avg |w| at each step.

Outputs (2 CSVs per architecture, 20 total):
  lenet_{h1}_{h2}_mnist_avg_random_prune_trained.csv
  lenet_{h1}_{h2}_mnist_avg_random_prune_static.csv

Usage (from project root):
    CUDA_VISIBLE_DEVICES=1 python -m src.mnist_lenet300.run_lenet_variable_random_pruning
"""

from src.infrastructure.nplh_run_context import NplhRunContext
from src.mnist_lenet300.train_lenet_variable_random_pruning import (
    run_arch_trained,
    run_arch_static,
    TRAIN_EPOCHS,
    PRUNING_RATE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    COLLAPSE_THRESHOLD,
)

# ------------------------------------------------------------------
# 10 architectures: very under-parameterized → very over-parameterized
# ------------------------------------------------------------------
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


def main() -> None:
    run_ctx = NplhRunContext.create(
        run_name="lenet_variable_random_pruning",
        description={
            "model":              "Variable LeNet (784→h1→h2→10)",
            "dataset":            "MNIST",
            "train_epochs":       TRAIN_EPOCHS,
            "pruning_rate":       PRUNING_RATE,
            "finetune_epochs":    FINETUNE_EPOCHS,
            "finetune_lr":        FINETUNE_LR,
            "collapse_threshold": f"{COLLAPSE_THRESHOLD}%",
            "pruning_criterion":  "random (uniform)",
            "architectures":      ", ".join(
                f"({a['h1']},{a['h2']})" for a in ARCHITECTURES
            ),
            "hypothesis": (
                "If avg |w| grows during trained random pruning but not "
                "in the static control, saliency growth is intrinsic to "
                "training.  Testing across architectures spanning under- to "
                "over-parameterized regimes reveals whether overparameterization "
                "drives the magnitude inflection point."
            ),
        },
    )

    for arch in ARCHITECTURES:
        h1, h2   = arch["h1"], arch["h2"]
        prunable = 784 * h1 + h1 * h2 + h2 * 10
        print(f"\n{'='*60}")
        print(f"Architecture: LeNet ({h1}, {h2})  |  {prunable:,} prunable weights")
        print(f"{'='*60}")

        # A: trained random pruning until collapse
        collapse_density = run_arch_trained(h1, h2, run_ctx)

        # B: static random pruning to the same density
        run_arch_static(h1, h2, run_ctx, target_density=collapse_density)

    print(f"\n{'='*60}")
    print("All architectures complete.")
    print(f"Results → {run_ctx.folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
