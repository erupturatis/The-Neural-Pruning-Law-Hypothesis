"""
run_resnet50_cifar10_new_policies.py
=====================================
Runs only the two new pruning methods on the already-saved ResNet50
CIFAR-10 baseline — no retraining from scratch.

Experiments
-----------
  1. Gradient       – prune by raw gradient magnitude |∂L/∂w|
  2. RandomRegrowth – random prune 2× then regrow those whose GD update
                      aligns with the weight's original sign

Usage (from project root):
    python -m src.resnet50_cifar10.run_resnet50_cifar10_new_policies
"""

from src.infrastructure.nplh_run_context import (
    NplhRunContext,
    METHOD_GRADIENT, METHOD_RANDOM_REGROWTH,
)
from src.resnet50_cifar10.train_resnet50_cifar10_nplh import (
    run_gradient_pruning,
    run_random_regrowth,
    IMP_EPOCHS,
    PRUNING_RATE,
    TARGET_SPARSITY,
    BASELINE_SAVE_NAME,
)


def main() -> None:
    run_ctx = NplhRunContext.create(
        run_name="resnet50_cifar10_new_policies",
        description={
            "model":           "ResNet50",
            "dataset":         "CIFAR-10",
            "baseline":        BASELINE_SAVE_NAME,
            "imp_epochs":      IMP_EPOCHS,
            "pruning_rate":    PRUNING_RATE,
            "target_sparsity": TARGET_SPARSITY,
            "methods":         f"{METHOD_GRADIENT}, {METHOD_RANDOM_REGROWTH}",
            "notes":           "Uses existing baseline; no scratch training.",
        },
    )

    run_gradient_pruning(BASELINE_SAVE_NAME, run_ctx)
    run_random_regrowth(BASELINE_SAVE_NAME, run_ctx)

    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"Results → {run_ctx.folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
