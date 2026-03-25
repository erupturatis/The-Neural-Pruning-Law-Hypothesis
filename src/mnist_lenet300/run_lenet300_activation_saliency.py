"""
run_lenet300_activation_saliency.py
=====================================
Runner for the random-pruning + dual-saliency experiment on LeNet-300-100
(MNIST).

Records both saliency types at each pruning step (measured before weights
are removed):
  1. Magnitude saliency  — mean |w| over all active weights.
  2. APoZ saliency       — Average Percentage of Zeros over active hidden
                           neurons (those with ≥1 unpruned incoming weight);
                           fraction of batch samples where each neuron does
                           not fire after ReLU.

Pruning criterion: uniformly random (10% of remaining weights per step).
Fine-tuning:       3 epochs of Adam between each pruning step.
Stopping:          when test accuracy drops below 15%.

Outputs (written to neural_pruning_law/final_data/<timestamped_folder>/):
  lenet_300_100_mnist_avg_random_retrain_magnitude.csv
  lenet_300_100_mnist_avg_random_retrain_apoz.csv

Run from the project root:
    python -m src.mnist_lenet300.run_lenet300_activation_saliency
"""

from src.infrastructure.nplh_run_context import NplhRunContext
from src.mnist_lenet300.train_lenet300_activation_saliency import (
    run_lenet300_random_activation,
    HIDDEN1, HIDDEN2,
    TRAIN_EPOCHS, PRUNING_RATE, FINETUNE_EPOCHS, FINETUNE_LR, COLLAPSE_THRESHOLD,
)


def main() -> None:
    run_ctx = NplhRunContext.create(
        run_name="lenet300_activation_saliency",
        description={
            "model":              f"LeNet-{HIDDEN1}-{HIDDEN2} (784→{HIDDEN1}→{HIDDEN2}→10)",
            "dataset":            "MNIST",
            "pruning_criterion":  "random (uniform)",
            "train_epochs":       TRAIN_EPOCHS,
            "pruning_rate":       PRUNING_RATE,
            "finetune_epochs":    FINETUNE_EPOCHS,
            "finetune_lr":        FINETUNE_LR,
            "collapse_threshold": f"{COLLAPSE_THRESHOLD}%",
            "saliency_1":         "magnitude — mean |w| of active weights before pruning",
            "saliency_2":         "APoZ — Average Percentage of Zeros over active hidden neurons (>=1 unpruned input) on one batch before pruning",
            "hypothesis": (
                "Does APoZ follow a power-law relationship with density "
                "under random pruning with fine-tuning, analogous to the "
                "weight-magnitude saliency trend?"
            ),
        },
    )

    run_lenet300_random_activation(run_ctx)

    print(f"\nRun complete. Results → {run_ctx.folder_path}")


if __name__ == "__main__":
    main()
