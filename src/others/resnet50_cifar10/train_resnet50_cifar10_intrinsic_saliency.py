"""
train_resnet50_cifar10_intrinsic_saliency.py
=============================================
Tests whether saliency (avg weight magnitude) grows intrinsically during
training, independently of which weights are removed.

Experiment A – Random pruning with training
    At each step: randomly prune 10% of remaining active weights, then
    fine-tune for 5 epochs at constant LR.  Record avg |w| of surviving
    weights before each pruning event.  Stop when accuracy collapses.
    Returns the collapse density.

Experiment B – Random pruning static (control)
    Load a fresh baseline and randomly prune 10% per step with NO
    retraining, down to the collapse density found in experiment A.
    Record avg |w| at each step.

The hypothesis: if saliency grows intrinsically, avg |w| in experiment A
will trend upward as density falls, while experiment B will not show the
same growth (since no training occurs to reorganise the remaining weights).
"""

import torch
import torch.nn as nn

from src.infrastructure.constants import BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType,
)
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.nplh_run_context import (
    NplhRunContext,
    COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
    SAL_AVG,
)
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent
from src.infrastructure.pruning_policy import RandomPruningPolicy
from src.infrastructure.read_write import save_dict_to_csv
from src.infrastructure.training_utils import get_model_weights_params
from src.resnet50_cifar10.resnet50_cifar10_class import Resnet50Cifar10
from src.resnet50_cifar10.train_resnet50_cifar10_nplh import (
    _make_cifar10_dataset_configs,
    _train_epoch,
    _test_epoch,
)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
MODEL_NAME         = "resnet50"
DATASET_NAME       = "cifar10"
BASELINE_SAVE_NAME = "resnet50_cifar10_nplh_baseline"

INTRINSIC_PRUNING_RATE       = 0.10   # 10% of remaining weights per step
INTRINSIC_FINETUNE_EPOCHS    = 3      # fine-tuning epochs between pruning steps
INTRINSIC_LR                 = 0.001  # constant LR throughout
INTRINSIC_COLLAPSE_THRESHOLD = 15.0  # % — stop when accuracy drops below this

METHOD_INTRINSIC_TRAINED = "random_prune_trained"
METHOD_INTRINSIC_STATIC  = "random_prune_static"


# -----------------------------------------------------------------------
# Shared model loader
# -----------------------------------------------------------------------

def _load_model(baseline_name: str) -> Resnet50Cifar10:
    device = get_device()
    configs_masks = ConfigsNetworkMask(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    model = Resnet50Cifar10(configs_masks).to(device)
    model.load(baseline_name, BASELINE_MODELS_PATH)
    return model


# -----------------------------------------------------------------------
# Experiment A: random pruning + training until collapse
# -----------------------------------------------------------------------

def run_random_pruning_trained(baseline_name: str, run_ctx: NplhRunContext) -> float:
    """
    Randomly prune 10% → fine-tune 5 epochs → repeat until collapse.

    avg_saliency recorded = mean |w| of active weights immediately
    before each pruning step (i.e. after the previous round of training).

    Returns
    -------
    collapse_density : float  — remaining% at the collapse step.
    """
    print(f"\n{'='*60}")
    print("Experiment A: Random pruning with training (intrinsic saliency test)")
    print(f"  pruning_rate={INTRINSIC_PRUNING_RATE}  "
          f"finetune_epochs={INTRINSIC_FINETUNE_EPOCHS}  LR={INTRINSIC_LR}")
    print(f"  collapse threshold={INTRINSIC_COLLAPSE_THRESHOLD}%")
    print(f"{'='*60}")

    model = _load_model(baseline_name)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )
    policy = RandomPruningPolicy()

    optimizer = torch.optim.SGD(
        get_model_weights_params(model),
        lr=INTRINSIC_LR, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )

    # Only avg saliency is meaningful for random pruning
    csv_avg = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, METHOD_INTRINSIC_TRAINED)

    steps = []; sal_avg = []; rem = []; acc_list = []
    step             = 0
    collapse_density = None
    global_epoch     = 0

    while True:
        # ── 1. Prune randomly; record avg |w| before pruning ──────────
        try:
            result = policy.prune_step(model, INTRINSIC_PRUNING_RATE)
        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {step}: {exc}")
            break

        remaining = get_custom_model_sparsity_percent(model)
        step += 1

        # ── 2. Fine-tune ───────────────────────────────────────────────
        acc = 0.0
        for _ in range(INTRINSIC_FINETUNE_EPOCHS):
            global_epoch += 1
            dataset_context.init_data_split()
            _train_epoch(model, dataset_context, optimizer)
            acc = _test_epoch(model, dataset_context, global_epoch)

        print(
            f"  [step {step:3d}] remaining={remaining:.4f}%  "
            f"avg_mag={result.avg_saliency:.4e}  acc={acc:.2f}%"
        )

        steps.append(step);        sal_avg.append(result.avg_saliency)
        rem.append(remaining);     acc_list.append(acc)

        save_dict_to_csv(
            {COL_STEP: steps, COL_REMAINING: rem,
             COL_SALIENCY: sal_avg, COL_ACCURACY: acc_list},
            filename=csv_avg,
        )

        # ── 3. Check collapse ──────────────────────────────────────────
        if acc < INTRINSIC_COLLAPSE_THRESHOLD:
            collapse_density = remaining
            print(f"\n  *** Collapse at step {step}: acc={acc:.2f}%  "
                  f"remaining={remaining:.4f}% ***")
            break

    if collapse_density is None:
        collapse_density = get_custom_model_sparsity_percent(model)
        print(f"  Warning: collapse never reached. Using final density {collapse_density:.4f}%.")

    print(f"  Experiment A done. Steps: {step}  Collapse density: {collapse_density:.4f}%")
    return collapse_density


# -----------------------------------------------------------------------
# Experiment B: random pruning static (control) to collapse density
# -----------------------------------------------------------------------

def run_random_pruning_static(
    baseline_name: str,
    run_ctx: NplhRunContext,
    target_density: float,
) -> None:
    """
    Randomly prune 10% per step with NO retraining, down to target_density%.
    Records avg |w| of surviving weights at each step.

    Parameters
    ----------
    target_density : float — collapse density from experiment A.
    """
    print(f"\n{'='*60}")
    print(f"Experiment B: Random pruning static (control, target={target_density:.4f}%)")
    print(f"{'='*60}")

    model = _load_model(baseline_name)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )
    policy = RandomPruningPolicy()

    csv_avg = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, METHOD_INTRINSIC_STATIC)

    steps = []; sal_avg = []; rem = []; acc_list = []
    step = 0

    while get_custom_model_sparsity_percent(model) > target_density:
        try:
            result    = policy.prune_step(model, INTRINSIC_PRUNING_RATE)
            remaining = get_custom_model_sparsity_percent(model)
            step += 1
        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {step}: {exc}")
            break

        dataset_context.init_data_split()
        acc = _test_epoch(model, dataset_context, step)

        print(
            f"  [step {step:3d}] remaining={remaining:.4f}%  "
            f"avg_mag={result.avg_saliency:.4e}  acc={acc:.2f}%"
        )

        steps.append(step);    sal_avg.append(result.avg_saliency)
        rem.append(remaining); acc_list.append(acc)

        save_dict_to_csv(
            {COL_STEP: steps, COL_REMAINING: rem,
             COL_SALIENCY: sal_avg, COL_ACCURACY: acc_list},
            filename=csv_avg,
        )

    print(f"  Experiment B done. Steps: {step}  "
          f"Final density: {get_custom_model_sparsity_percent(model):.4f}%")
