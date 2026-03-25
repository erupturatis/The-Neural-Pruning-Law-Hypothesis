"""
run_lenet_variable_experiment.py
=================================
Runs IMP (Iterative Magnitude Pruning) on 10 variable-width LeNet
architectures of increasing size on MNIST, spanning clearly
under-parameterized to clearly over-parameterized regimes.

For each architecture this script:
  1. Trains from scratch for BASELINE_EPOCHS dense epochs and records
     the baseline test accuracy.
  2. Continues training for IMP_EPOCHS more epochs, pruning 10 % of
     remaining weights at evenly-spaced checkpoints and recording NPLH
     data (epoch, min-saliency, density, accuracy) at every pruning step.

Outputs (all inside neural_pruning_law/final_data/lenet_variable/):
  lenet_{h1}_{h2}_imp.csv          — NPLH data for each architecture
  lenet_variable_baselines.csv     — summary of baseline accuracies

Run from the project root:
    python -m src.mnist_lenet300.run_lenet_variable_experiment
"""

import os

from src.infrastructure.nplh_run_context import NplhRunContext
from src.infrastructure.pruning_policy import MagnitudePruningPolicy, PruningPolicy
from src.infrastructure.read_write import save_dict_to_csv
from src.mnist_lenet300.train_NPLH_IMP_lenet_variable import (
    train_lenet_variable_imp,
    BASELINE_EPOCHS, IMP_EPOCHS, PRUNING_RATE, TARGET_SPARSITY,
)

# ------------------------------------------------------------------
# 10 architectures: very under-parameterized → very over-parameterized
#
# Prunable params = 784*h1 + h1*h2 + h2*10
#
#  (4,   2)   →      3 164   very under-parameterized
#  (10,  5)   →      7 940   under-parameterized
#  (20, 10)   →     15 980   under-parameterized
#  (50, 25)   →     40 700   borderline
#  (100, 50)  →     83 900   approaching optimal
#  (300,100)  →    266 200   canonical / optimal
#  (600,200)  →    592 400   mildly over-parameterized
# (1200,400)  →  1 424 800   over-parameterized
# (2500,800)  →  3 968 000   clearly over-parameterized
# (5000,1500) → 11 435 000   very over-parameterized
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


def run_all(
    pruning_policy: PruningPolicy | None = None,
    save_weight_distributions: bool = False,
):
    if pruning_policy is None:
        pruning_policy = MagnitudePruningPolicy()

    run_ctx = NplhRunContext.create(
        run_name="lenet_variable_mnist",
        description={
            "model":                "Variable LeNet (784→h1→h2→10)",
            "dataset":              "MNIST",
            "method":               pruning_policy.method_tag,
            "architectures":        ", ".join(
                f"({a['h1']},{a['h2']})" for a in ARCHITECTURES
            ),
            "baseline_epochs":      BASELINE_EPOCHS,
            "imp_epochs":           IMP_EPOCHS,
            "pruning_rate":         PRUNING_RATE,
            "target_sparsity":      TARGET_SPARSITY,
            "save_weight_distributions": save_weight_distributions,
        },
    )

    arch_names    = []
    h1_list       = []
    h2_list       = []
    param_counts  = []
    baseline_accs = []

    for arch in ARCHITECTURES:
        h1, h2   = arch["h1"], arch["h2"]
        prunable = 784 * h1 + h1 * h2 + h2 * 10

        print(f"\n{'='*60}")
        print(f"Architecture: LeNet ({h1}, {h2})  |  {prunable:,} prunable weights")
        print(f"{'='*60}")

        baseline_acc = train_lenet_variable_imp(
            h1, h2, run_ctx,
            pruning_policy=pruning_policy,
            save_weight_distributions=save_weight_distributions,
        )

        arch_names.append(f"lenet_{h1}_{h2}")
        h1_list.append(h1)
        h2_list.append(h2)
        param_counts.append(prunable)
        baseline_accs.append(baseline_acc)

        print(f"  Finished. Baseline accuracy: {baseline_acc:.2f}%")

    # Save baseline summary inside the run folder
    summary_path = os.path.join(run_ctx.folder_path, "lenet_variable_baselines.csv")
    save_dict_to_csv(
        {
            "Architecture":     arch_names,
            "H1":               h1_list,
            "H2":               h2_list,
            "PrunableParams":   param_counts,
            "BaselineAccuracy": baseline_accs,
        },
        filename=summary_path,
    )

    print(f"\n{'='*60}")
    print(f"All experiments complete.")
    print(f"Run folder      → {run_ctx.folder_path}")
    print(f"Baseline summary → {summary_path}")
    print(f"{'='*60}")
    print("\nResults:")
    for name, acc in zip(arch_names, baseline_accs):
        print(f"  {name:<22}  baseline: {acc:.2f}%")

    return run_ctx


if __name__ == "__main__":
    run_all()
