"""
run_lenet300_experiment.py
==========================
Runs 3 pruning policies + a no-retrain control on the canonical LeNet300
architecture (784 → 300 → 100 → 10, ~266 k prunable weights) on MNIST.

Experiments
-----------
  1. IMP – Magnitude       prune by |w|, fine-tune between steps
  2. IMP – Taylor          prune by |w · ∂L/∂w|, fine-tune between steps
  3. IMP – GraSP           prune by −w · Hg, fine-tune between steps
  4. Control               prune by magnitude, NO retraining at all

Each experiment produces two CSVs (min saliency + avg saliency) inside a
single timestamped run folder.

Run from the project root:
    python -m src.mnist_lenet300.run_lenet300_experiment
"""

import torch.nn as nn

from src.infrastructure.nplh_run_context import NplhRunContext
from src.infrastructure.pruning_policy import (
    MagnitudePruningPolicy,
    TaylorPruningPolicy,
)
from src.mnist_lenet300.train_NPLH_IMP_lenet_variable import train_lenet_variable_imp
from src.mnist_lenet300.train_NPLH_control_lenet_variable import train_lenet_control

# ------------------------------------------------------------------
# Architecture
# ------------------------------------------------------------------
H1, H2 = 300, 100

# ------------------------------------------------------------------
# Training hypers
# ------------------------------------------------------------------
BASELINE_EPOCHS = 20
IMP_EPOCHS      = 300
PRUNING_RATE    = 0.05
TARGET_SPARSITY = 0.999   # prune until 0.1 % of weights remain


def run() -> NplhRunContext:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    policies = [
        MagnitudePruningPolicy(),
        TaylorPruningPolicy(criterion),
    ]

    run_ctx = NplhRunContext.create(
        run_name="lenet300_all_policies",
        description={
            "model":           f"LeNet ({H1}, {H2})  –  784→{H1}→{H2}→10",
            "dataset":         "MNIST",
            "baseline_epochs": BASELINE_EPOCHS,
            "imp_epochs":      IMP_EPOCHS,
            "pruning_rate":    PRUNING_RATE,
            "target_sparsity": TARGET_SPARSITY,
            "policies":        ", ".join(p.method_tag for p in policies),
            "control":         "magnitude pruning, no retraining",
        },
    )

    # ── 3 IMP experiments ──────────────────────────────────────────────────
    for policy in policies:
        print(f"\n{'='*60}")
        print(f"Policy: {policy.method_tag}")
        print(f"{'='*60}")
        train_lenet_variable_imp(
            H1, H2,
            run_ctx=run_ctx,
            pruning_policy=policy,
            baseline_epochs=BASELINE_EPOCHS,
            imp_epochs=IMP_EPOCHS,
            pruning_rate=PRUNING_RATE,
            target_sparsity=TARGET_SPARSITY,
        )

    # ── Control (no retraining) ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Control: magnitude pruning, no retraining")
    print(f"{'='*60}")
    train_lenet_control(
        H1, H2,
        run_ctx=run_ctx,
        baseline_epochs=BASELINE_EPOCHS,
        pruning_rate=PRUNING_RATE,
        target_sparsity=TARGET_SPARSITY,
    )

    print(f"\n{'='*60}")
    print(f"All experiments complete.")
    print(f"Run folder → {run_ctx.folder_path}")
    print(f"{'='*60}")
    return run_ctx


if __name__ == "__main__":
    run()
