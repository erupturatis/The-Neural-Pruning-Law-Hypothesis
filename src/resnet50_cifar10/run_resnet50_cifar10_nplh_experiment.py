"""
run_resnet50_cifar10_nplh_experiment.py
=======================================
Runner for the ResNet50 CIFAR-10 NPLH experiment.

Steps
-----
  1. Train ResNet50 from scratch on CIFAR-10 (device-agnostic, no wandb).
  2. Save the baseline to networks_baseline/ as "resnet50_cifar10_nplh_baseline".
  3. Run three pruning experiments from the same saved baseline:
       a. Static   – magnitude pruning, no retraining (control)
       b. IMP      – magnitude pruning + fine-tuning
       c. Taylor   – first-order Taylor pruning + fine-tuning
  4. All CSVs (min-saliency + avg-saliency per experiment) are saved to a
     timestamped folder under neural_pruning_law/final_data/.

Usage (from project root):
    python -m src.resnet50_cifar10.run_resnet50_cifar10_nplh_experiment
"""

from src.infrastructure.nplh_run_context import (
    NplhRunContext,
    METHOD_IMP_STATIC, METHOD_IMP_MAGNITUDE, METHOD_IMP_TAYLOR,
)
from src.resnet50_cifar10.train_resnet50_cifar10_nplh import (
    train_resnet50_cifar10_baseline,
    run_static_pruning,
    run_imp_magnitude,
    run_imp_taylor,
    SCRATCH_EPOCHS,
    IMP_EPOCHS,
    PRUNING_RATE,
    TARGET_SPARSITY,
    BASELINE_SAVE_NAME,
)


def main() -> None:
    # 1. Shared run context — one timestamped folder for all 3 experiments
    run_ctx = NplhRunContext.create(
        run_name="resnet50_cifar10_all_policies",
        description={
            "model":            "ResNet50",
            "dataset":          "CIFAR-10",
            "baseline_epochs":  SCRATCH_EPOCHS,
            "imp_epochs":       IMP_EPOCHS,
            "pruning_rate":     PRUNING_RATE,
            "target_sparsity":  TARGET_SPARSITY,
            "methods":          f"{METHOD_IMP_STATIC}, {METHOD_IMP_MAGNITUDE}, {METHOD_IMP_TAYLOR}",
            "notes":            "Train once from scratch, then run 3 experiments from the same baseline.",
        },
    )

    # 2. Train baseline from scratch (device-agnostic)
    baseline_name = train_resnet50_cifar10_baseline()

    # 3a. Static pruning (no retraining) — control baseline
    run_static_pruning(baseline_name, run_ctx)

    # 3b. IMP – magnitude
    run_imp_magnitude(baseline_name, run_ctx)

    # 3c. IMP – Taylor
    run_imp_taylor(baseline_name, run_ctx)

    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"Results → {run_ctx.folder_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
