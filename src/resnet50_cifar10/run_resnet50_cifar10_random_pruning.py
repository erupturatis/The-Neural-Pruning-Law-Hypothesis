"""
run_resnet50_cifar10_random_pruning.py
======================================
Random-pruning experiments for ResNet50 CIFAR-10.

Experiments
-----------
  1. random_static  – random pruning, NO retraining.
                      Records magnitude, gradient, and Taylor saliency → 6 CSVs.
  2. random_imp     – random pruning, WITH retraining (IMP schedule).
                      Records magnitude, gradient, and Taylor saliency → 6 CSVs.

Both experiments require the pre-trained baseline in networks_baseline/.

Usage (from project root):
    python -m src.resnet50_cifar10.run_resnet50_cifar10_random_pruning
"""

from src.infrastructure.nplh_run_context import NplhRunContext
from src.resnet50_cifar10.train_resnet50_cifar10_nplh import (
    run_random_static,
    run_random_retraining,
    BASELINE_SAVE_NAME,
    PRUNING_RATE,
    TARGET_SPARSITY,
    IMP_EPOCHS,
)


def main() -> None:
    run_ctx = NplhRunContext.create(
        run_name="resnet50_cifar10_random_pruning",
        description={
            "model":           "ResNet50",
            "dataset":         "CIFAR-10",
            "baseline":        BASELINE_SAVE_NAME,
            "pruning_rate":    PRUNING_RATE,
            "target_sparsity": TARGET_SPARSITY,
            "imp_epochs":      IMP_EPOCHS,
            "methods":         "random_static, random_retraining",
            "saliency_types":  "magnitude, gradient, taylor (all recorded per step)",
            "notes":           "Random pruning experiments; saliency measured independently of criterion.",
        },
    )

    run_random_static(BASELINE_SAVE_NAME, run_ctx)
    run_random_retraining(BASELINE_SAVE_NAME, run_ctx)

    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"Results → {run_ctx.folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
