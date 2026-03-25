"""
train_resnet50_cifar10_collapse.py
===================================
Two experiments that prune until accuracy collapses to near-random.

Experiment A – Collapse IMP
    Magnitude pruning with constant-LR fine-tuning between steps.
    Prunes COLLAPSE_PRUNING_RATE of remaining weights each step, then
    trains COLLAPSE_FINETUNE_EPOCHS epochs.  Stops as soon as test
    accuracy drops below COLLAPSE_ACCURACY_THRESHOLD.
    Returns the density (remaining%) at which collapse was observed.

Experiment B – Collapse Static
    Static magnitude pruning (no retraining) from the baseline down to
    the collapse density found in experiment A.  Records the saliency
    path to that same density for comparison.

Both save two CSVs (min saliency + avg saliency) incrementally.
"""

import torch
import torch.nn as nn

from src.infrastructure.constants import BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType,
)
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.nplh_run_context import (
    NplhRunContext,
    COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
    SAL_MIN, SAL_AVG,
)
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent
from src.infrastructure.pruning_policy import MagnitudePruningPolicy
from src.infrastructure.read_write import save_dict_to_csv
from src.infrastructure.training_common import get_model_weights_params
from src.resnet50_cifar10.resnet50_cifar10_class import Resnet50Cifar10
from src.resnet50_cifar10.train_resnet50_cifar10_nplh import (
    _make_cifar10_dataset_configs,
    _train_epoch,
    _test_epoch,
)

# -----------------------------------------------------------------------
# Collapse experiment constants
# -----------------------------------------------------------------------
MODEL_NAME   = "resnet50"
DATASET_NAME = "cifar10"
BASELINE_SAVE_NAME = "resnet50_cifar10_nplh_baseline"

COLLAPSE_PRUNING_RATE        = 0.10   # prune 10% of remaining weights per step
COLLAPSE_FINETUNE_EPOCHS     = 3      # fine-tuning epochs between pruning steps
COLLAPSE_LR                  = 0.001  # constant learning rate throughout
COLLAPSE_ACCURACY_THRESHOLD  = 15.0  # % — stop when accuracy drops below this

# Method tags used in CSV filenames
METHOD_COLLAPSE_IMP    = "collapse_imp"
METHOD_COLLAPSE_STATIC = "collapse_static"


# -----------------------------------------------------------------------
# Shared model loader
# -----------------------------------------------------------------------

def _load_model(baseline_name: str) -> Resnet50Cifar10:
    device = get_device()
    configs_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    model = Resnet50Cifar10(configs_masks).to(device)
    model.load(baseline_name, BASELINE_MODELS_PATH)
    return model


# -----------------------------------------------------------------------
# Experiment A: IMP until accuracy collapses
# -----------------------------------------------------------------------

def run_collapse_imp(baseline_name: str, run_ctx: NplhRunContext) -> float:
    """
    Magnitude-prune 10% per step, fine-tune 3 epochs at constant LR,
    repeat until accuracy < COLLAPSE_ACCURACY_THRESHOLD.

    Returns
    -------
    collapse_density : float
        The remaining-weight percentage at the step where collapse was
        first detected.  Pass this to run_collapse_static().
    """
    print(f"\n{'='*60}")
    print("Experiment A: Collapse IMP (prune until accuracy collapses)")
    print(f"  pruning_rate={COLLAPSE_PRUNING_RATE}  finetune_epochs={COLLAPSE_FINETUNE_EPOCHS}")
    print(f"  collapse threshold={COLLAPSE_ACCURACY_THRESHOLD}%  LR={COLLAPSE_LR}")
    print(f"{'='*60}")

    model = _load_model(baseline_name)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )
    policy = MagnitudePruningPolicy()

    optimizer = torch.optim.SGD(
        get_model_weights_params(model),
        lr=COLLAPSE_LR, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )

    csv_min = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_MIN, METHOD_COLLAPSE_IMP)
    csv_avg = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, METHOD_COLLAPSE_IMP)

    steps_min = []; sal_min = []; rem_min = []; acc_min = []
    steps_avg = []; sal_avg = []; rem_avg = []; acc_avg = []

    step           = 0
    collapse_density = None
    global_epoch   = 0

    while True:
        # ── 1. Prune ──────────────────────────────────────────────────
        try:
            result = policy.prune_step(model, COLLAPSE_PRUNING_RATE)
        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {step}: {exc}")
            break

        remaining = get_custom_model_sparsity_percent(model)
        step += 1

        # ── 2. Fine-tune for COLLAPSE_FINETUNE_EPOCHS epochs ──────────
        acc = 0.0
        for _ in range(COLLAPSE_FINETUNE_EPOCHS):
            global_epoch += 1
            dataset_context.init_data_split()
            _train_epoch(model, dataset_context, optimizer)
            acc = _test_epoch(model, dataset_context, global_epoch)

        print(
            f"  [step {step:3d}] remaining={remaining:.4f}%  "
            f"threshold={result.threshold:.4e}  avg={result.avg_saliency:.4e}  "
            f"acc={acc:.2f}%"
        )

        steps_min.append(step); sal_min.append(result.threshold)
        rem_min.append(remaining); acc_min.append(acc)

        steps_avg.append(step); sal_avg.append(result.avg_saliency)
        rem_avg.append(remaining); acc_avg.append(acc)

        save_dict_to_csv(
            {COL_STEP: steps_min, COL_REMAINING: rem_min,
             COL_SALIENCY: sal_min, COL_ACCURACY: acc_min},
            filename=csv_min,
        )
        save_dict_to_csv(
            {COL_STEP: steps_avg, COL_REMAINING: rem_avg,
             COL_SALIENCY: sal_avg, COL_ACCURACY: acc_avg},
            filename=csv_avg,
        )

        # ── 3. Check for collapse ──────────────────────────────────────
        if acc < COLLAPSE_ACCURACY_THRESHOLD:
            collapse_density = remaining
            print(f"\n  *** Collapse detected at step {step}: "
                  f"acc={acc:.2f}% < {COLLAPSE_ACCURACY_THRESHOLD}%  "
                  f"remaining={remaining:.4f}% ***")
            break

    if collapse_density is None:
        # Pruning exhausted weights before collapse — use final density
        collapse_density = get_custom_model_sparsity_percent(model)
        print(f"  Warning: collapse threshold never reached. "
              f"Using final density {collapse_density:.4f}%.")

    print(f"  Collapse IMP done. Steps: {step}  Collapse density: {collapse_density:.4f}%")
    return collapse_density


# -----------------------------------------------------------------------
# Experiment B: Static pruning to the observed collapse density
# -----------------------------------------------------------------------

def run_collapse_static(
    baseline_name: str,
    run_ctx: NplhRunContext,
    target_density: float,
) -> None:
    """
    Static magnitude pruning (no retraining) from the baseline down to
    target_density%.  Records the saliency path along the way.

    Parameters
    ----------
    target_density : float
        Remaining-weight % at which to stop (from run_collapse_imp).
    """
    print(f"\n{'='*60}")
    print(f"Experiment B: Collapse Static (prune to {target_density:.4f}% with no retraining)")
    print(f"{'='*60}")

    model = _load_model(baseline_name)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )
    policy = MagnitudePruningPolicy()

    csv_min = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_MIN, METHOD_COLLAPSE_STATIC)
    csv_avg = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, METHOD_COLLAPSE_STATIC)

    steps_min = []; sal_min = []; rem_min = []; acc_min = []
    steps_avg = []; sal_avg = []; rem_avg = []; acc_avg = []
    step = 0

    while get_custom_model_sparsity_percent(model) > target_density:
        try:
            result    = policy.prune_step(model, COLLAPSE_PRUNING_RATE)
            remaining = get_custom_model_sparsity_percent(model)
            step += 1
        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {step}: {exc}")
            break

        dataset_context.init_data_split()
        acc = _test_epoch(model, dataset_context, step)

        print(
            f"  [step {step:3d}] remaining={remaining:.4f}%  "
            f"threshold={result.threshold:.4e}  avg={result.avg_saliency:.4e}  "
            f"acc={acc:.2f}%"
        )

        steps_min.append(step); sal_min.append(result.threshold)
        rem_min.append(remaining); acc_min.append(acc)

        steps_avg.append(step); sal_avg.append(result.avg_saliency)
        rem_avg.append(remaining); acc_avg.append(acc)

        save_dict_to_csv(
            {COL_STEP: steps_min, COL_REMAINING: rem_min,
             COL_SALIENCY: sal_min, COL_ACCURACY: acc_min},
            filename=csv_min,
        )
        save_dict_to_csv(
            {COL_STEP: steps_avg, COL_REMAINING: rem_avg,
             COL_SALIENCY: sal_avg, COL_ACCURACY: acc_avg},
            filename=csv_avg,
        )

    print(f"  Collapse Static done.  Steps: {step}  Final density: {get_custom_model_sparsity_percent(model):.4f}%")
