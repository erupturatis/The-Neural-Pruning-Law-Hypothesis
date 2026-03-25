"""
run_resnet50_cifar10_multi_saliency.py
=======================================
Runner for the multi-saliency random-pruning experiment on ResNet50 CIFAR-10.

Randomly prunes 10% of remaining weights per step, fine-tunes 3 epochs,
and records magnitude / Taylor / gradient saliency simultaneously at each
step until accuracy collapses below 15%.

Uses the pre-saved baseline — no scratch training.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 python -m src.resnet50_cifar10.run_resnet50_cifar10_multi_saliency
"""

from src.infrastructure.nplh_run_context import NplhRunContext
from src.resnet50_cifar10.train_resnet50_cifar10_multi_saliency import (
    run_random_pruning_multi_saliency,
    BASELINE_SAVE_NAME,
    MULTI_PRUNING_RATE,
    MULTI_FINETUNE_EPOCHS,
    MULTI_LR,
    MULTI_COLLAPSE_THRESHOLD,
    METHOD_MULTI_MAGNITUDE,
    METHOD_MULTI_TAYLOR,
    METHOD_MULTI_GRADIENT,
)


def main() -> None:
    run_ctx = NplhRunContext.create(
        run_name="resnet50_cifar10_multi_saliency",
        description={
            "model":              "ResNet50",
            "dataset":            "CIFAR-10",
            "baseline":           BASELINE_SAVE_NAME,
            "pruning_rate":       MULTI_PRUNING_RATE,
            "finetune_epochs":    MULTI_FINETUNE_EPOCHS,
            "finetune_lr":        MULTI_LR,
            "collapse_threshold": f"{MULTI_COLLAPSE_THRESHOLD}%",
            "pruning_criterion":  "random (uniform)",
            "saliency_metrics":   (
                f"{METHOD_MULTI_MAGNITUDE} (mean |w|), "
                f"{METHOD_MULTI_TAYLOR} (mean |w·∂L/∂w|), "
                f"{METHOD_MULTI_GRADIENT} (mean |∂L/∂w|)"
            ),
            "hypothesis": (
                "If all three saliency metrics grow as density falls, "
                "effective saliency growth is independent of the pruning criterion."
            ),
        },
    )

    collapse_density = run_random_pruning_multi_saliency(BASELINE_SAVE_NAME, run_ctx)

    print(f"\n{'='*60}")
    print("Experiment complete.")
    print(f"Collapse density : {collapse_density:.4f}%")
    print(f"Results          → {run_ctx.folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
