"""
run_resnet50_cifar10_intrinsic_saliency.py
===========================================
Runner for the intrinsic-saliency experiment on ResNet50 CIFAR-10.

  A. Random pruning + training  — randomly prune 10%, fine-tune 5 epochs,
                                   repeat until accuracy < 15%.
  B. Random pruning static       — same random pruning with NO retraining,
                                   down to the collapse density from A.

Uses the already-saved baseline — no scratch training.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=1 python -m src.resnet50_cifar10.run_resnet50_cifar10_intrinsic_saliency
"""

from src.infrastructure.nplh_run_context import NplhRunContext
from src.resnet50_cifar10.train_resnet50_cifar10_intrinsic_saliency import (
    run_random_pruning_trained,
    run_random_pruning_static,
    BASELINE_SAVE_NAME,
    INTRINSIC_PRUNING_RATE,
    INTRINSIC_FINETUNE_EPOCHS,
    INTRINSIC_LR,
    INTRINSIC_COLLAPSE_THRESHOLD,
    METHOD_INTRINSIC_TRAINED,
    METHOD_INTRINSIC_STATIC,
)


def main() -> None:
    run_ctx = NplhRunContext.create(
        run_name="resnet50_cifar10_intrinsic_saliency",
        description={
            "model":               "ResNet50",
            "dataset":             "CIFAR-10",
            "baseline":            BASELINE_SAVE_NAME,
            "pruning_rate":        INTRINSIC_PRUNING_RATE,
            "finetune_epochs":     INTRINSIC_FINETUNE_EPOCHS,
            "finetune_lr":         INTRINSIC_LR,
            "collapse_threshold":  f"{INTRINSIC_COLLAPSE_THRESHOLD}%",
            "pruning_criterion":   "random (uniform)",
            "saliency_metric":     "avg |w| of surviving weights",
            "methods":             f"{METHOD_INTRINSIC_TRAINED}, {METHOD_INTRINSIC_STATIC}",
            "hypothesis": (
                "If avg |w| grows in A but not B, saliency increase is driven "
                "by training (intrinsic), not by the pruning criterion."
            ),
        },
    )

    # A: random prune + training until collapse
    collapse_density = run_random_pruning_trained(BASELINE_SAVE_NAME, run_ctx)

    # B: random prune static to the same density
    run_random_pruning_static(BASELINE_SAVE_NAME, run_ctx, target_density=collapse_density)

    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"Collapse density: {collapse_density:.4f}%")
    print(f"Results → {run_ctx.folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
