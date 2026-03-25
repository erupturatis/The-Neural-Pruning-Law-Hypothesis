"""
run_resnet50_cifar10_static_scored.py
======================================
Static (no-retraining) pruning using scored criteria on ResNet50 CIFAR-10.

Experiments
-----------
  1. gradient_static  – prune by |∂L/∂w|,     no retraining
  2. taylor_static    – prune by |w · ∂L/∂w|, no retraining

Both mirror the magnitude static experiment but use data-dependent scores.
Each produces min- and avg-saliency CSVs:
  resnet50_cifar10_min_gradient_static.csv
  resnet50_cifar10_avg_gradient_static.csv
  resnet50_cifar10_min_taylor_static.csv
  resnet50_cifar10_avg_taylor_static.csv

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 python -m src.resnet50_cifar10.run_resnet50_cifar10_static_scored
"""

from src.infrastructure.nplh_run_context import (
    NplhRunContext,
    METHOD_GRADIENT_STATIC, METHOD_TAYLOR_STATIC,
)
from src.resnet50_cifar10.train_resnet50_cifar10_nplh import (
    run_static_gradient,
    run_static_taylor,
    PRUNING_RATE,
    TARGET_SPARSITY,
    BASELINE_SAVE_NAME,
)


def main() -> None:
    run_ctx = NplhRunContext.create(
        run_name="resnet50_cifar10_static_scored",
        description={
            "model":           "ResNet50",
            "dataset":         "CIFAR-10",
            "baseline":        BASELINE_SAVE_NAME,
            "pruning_rate":    PRUNING_RATE,
            "target_sparsity": TARGET_SPARSITY,
            "retraining":      "none",
            "methods":         f"{METHOD_GRADIENT_STATIC}, {METHOD_TAYLOR_STATIC}",
            "notes":           "Static control variants for gradient and Taylor criteria.",
        },
    )

    run_static_gradient(BASELINE_SAVE_NAME, run_ctx)
    run_static_taylor(BASELINE_SAVE_NAME, run_ctx)

    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"Results → {run_ctx.folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
