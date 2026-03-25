"""
run_resnet50_cifar10_collapse_experiment.py
============================================
Runner for the two collapse experiments on ResNet50 CIFAR-10.

  A. Collapse IMP  – prune 10% per step, 3 epochs constant-LR fine-tuning,
                     stop when accuracy < 15% (near-random for CIFAR-10).
  B. Collapse Static – static magnitude pruning (no retraining) to the
                       same density at which collapse was observed in A.

Uses the already-saved baseline — no scratch training.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=1 python -m src.resnet50_cifar10.run_resnet50_cifar10_collapse_experiment
"""

from src.infrastructure.nplh_run_context import NplhRunContext
from src.resnet50_cifar10.train_resnet50_cifar10_collapse import (
    run_collapse_imp,
    run_collapse_static,
    BASELINE_SAVE_NAME,
    COLLAPSE_PRUNING_RATE,
    COLLAPSE_FINETUNE_EPOCHS,
    COLLAPSE_LR,
    COLLAPSE_ACCURACY_THRESHOLD,
    METHOD_COLLAPSE_IMP,
    METHOD_COLLAPSE_STATIC,
)


def main() -> None:
    run_ctx = NplhRunContext.create(
        run_name="resnet50_cifar10_collapse",
        description={
            "model":               "ResNet50",
            "dataset":             "CIFAR-10",
            "baseline":            BASELINE_SAVE_NAME,
            "pruning_rate":        COLLAPSE_PRUNING_RATE,
            "finetune_epochs":     COLLAPSE_FINETUNE_EPOCHS,
            "finetune_lr":         COLLAPSE_LR,
            "collapse_threshold":  f"{COLLAPSE_ACCURACY_THRESHOLD}%",
            "methods":             f"{METHOD_COLLAPSE_IMP}, {METHOD_COLLAPSE_STATIC}",
            "notes": (
                "A: IMP until accuracy collapses. "
                "B: Static pruning to the collapse density observed in A."
            ),
        },
    )

    # A: prune with fine-tuning until accuracy collapses; get collapse density
    collapse_density = run_collapse_imp(BASELINE_SAVE_NAME, run_ctx)

    # B: static pruning (no retraining) to the same density
    run_collapse_static(BASELINE_SAVE_NAME, run_ctx, target_density=collapse_density)

    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"Collapse density: {collapse_density:.4f}%")
    print(f"Results → {run_ctx.folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
