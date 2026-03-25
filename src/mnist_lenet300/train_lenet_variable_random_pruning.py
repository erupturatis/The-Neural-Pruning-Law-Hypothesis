"""
train_lenet_variable_random_pruning.py
=======================================
Intrinsic-saliency random-pruning experiment on variable-width LeNet
architectures (MNIST).

For each architecture this module provides:

  run_arch_trained(h1, h2, run_ctx) -> float
      Train 25 epochs from scratch, then randomly prune 10% per step with
      3 epochs of fine-tuning in between.  Record avg |w| (magnitude
      saliency) at each step until accuracy collapses below
      COLLAPSE_THRESHOLD.  Returns the collapse density.

  run_arch_static(h1, h2, run_ctx, target_density)
      Load a fresh model, train 25 epochs, then randomly prune 10% per
      step with NO retraining, down to target_density%.  Record avg |w|
      at each step.

The hypothesis: if avg |w| grows in the trained experiment but not in the
static control, saliency growth is driven by training (intrinsic), not by
the removal of specific weights.  This is tested across architectures
spanning under- to over-parameterized regimes.

CSV filenames (one pair per architecture):
  lenet_{h1}_{h2}_mnist_avg_random_prune_trained.csv
  lenet_{h1}_{h2}_mnist_avg_random_prune_static.csv
"""

import torch
import torch.nn as nn

from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, dataset_context_configs_mnist,
)
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.nplh_run_context import (
    NplhRunContext,
    COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
    SAL_AVG,
)
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent
from src.infrastructure.pruning_policy import RandomPruningPolicy
from src.infrastructure.read_write import save_dict_to_csv
from src.infrastructure.training_common import get_model_weights_params
from src.mnist_lenet300.model_class_variable import ModelLenetVariable
from src.mnist_lenet300.train_NPLH_IMP_lenet_variable import _train_epoch, _test_epoch

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
TRAIN_EPOCHS       = 25    # epochs of dense training before pruning
PRUNING_RATE       = 0.10  # fraction of remaining weights pruned per step
FINETUNE_EPOCHS    = 3     # fine-tuning epochs between pruning steps (trained exp)
FINETUNE_LR        = 0.001 # constant LR throughout fine-tuning
COLLAPSE_THRESHOLD = 15.0  # % — stop when accuracy drops below this

METHOD_TRAINED = "random_prune_trained"
METHOD_STATIC  = "random_prune_static"
DATASET_NAME   = "mnist"


# -----------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------

def _build_model(hidden1: int, hidden2: int) -> ModelLenetVariable:
    configs_layers_initialization_all_kaiming_sqrt5()
    configs_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    return ModelLenetVariable(hidden1, hidden2, configs_masks).to(get_device())


def _train_dense(
    model: ModelLenetVariable,
    dataset_context: DatasetSmallContext,
    n_epochs: int,
) -> float:
    """Train for n_epochs with Adam and return the final test accuracy."""
    optimizer = torch.optim.Adam(
        get_model_weights_params(model), lr=FINETUNE_LR, weight_decay=1e-4,
    )
    acc = 0.0
    for epoch in range(1, n_epochs + 1):
        dataset_context.init_data_split()
        _train_epoch(model, dataset_context, optimizer)
        acc = _test_epoch(model, dataset_context, epoch)
    return acc


# -----------------------------------------------------------------------
# Experiment A: random pruning + training until collapse
# -----------------------------------------------------------------------

def run_arch_trained(
    hidden1: int,
    hidden2: int,
    run_ctx: NplhRunContext,
) -> float:
    """
    Train from scratch for TRAIN_EPOCHS, then randomly prune + fine-tune
    until accuracy < COLLAPSE_THRESHOLD.  Records avg |w| before each
    pruning step.

    Returns
    -------
    collapse_density : float — remaining% at the collapse step.
    """
    arch_name = f"lenet_{hidden1}_{hidden2}"
    print(f"\n  [{arch_name}] Experiment A: random prune + training")

    model          = _build_model(hidden1, hidden2)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.MNIST,
        configs=dataset_context_configs_mnist(),
    )

    # Phase 1: dense training
    print(f"  [{arch_name}] Dense training ({TRAIN_EPOCHS} epochs) ...")
    baseline_acc = _train_dense(model, dataset_context, TRAIN_EPOCHS)
    print(f"  [{arch_name}] Baseline accuracy: {baseline_acc:.2f}%")

    # Phase 2: random prune → fine-tune → repeat
    policy = RandomPruningPolicy()
    optimizer = torch.optim.Adam(
        get_model_weights_params(model), lr=FINETUNE_LR, weight_decay=1e-4,
    )

    csv_avg = run_ctx.csv_path(arch_name, DATASET_NAME, SAL_AVG, METHOD_TRAINED)

    steps = []; sal_avg = []; rem = []; acc_list = []
    step             = 0
    collapse_density = None
    global_epoch     = TRAIN_EPOCHS

    while True:
        # Prune randomly (avg saliency = mean|w| before pruning)
        try:
            result = policy.prune_step(model, PRUNING_RATE)
        except (ValueError, RuntimeError) as exc:
            print(f"  [{arch_name}] Pruning stopped at step {step}: {exc}")
            break

        remaining = get_custom_model_sparsity_percent(model)
        step += 1

        # Fine-tune
        acc = 0.0
        for _ in range(FINETUNE_EPOCHS):
            global_epoch += 1
            dataset_context.init_data_split()
            _train_epoch(model, dataset_context, optimizer)
            acc = _test_epoch(model, dataset_context, global_epoch)

        print(
            f"  [{arch_name}] step {step:3d}  remaining={remaining:.4f}%  "
            f"avg_mag={result.avg_saliency:.4e}  acc={acc:.2f}%"
        )

        steps.append(step);   sal_avg.append(result.avg_saliency)
        rem.append(remaining); acc_list.append(acc)

        save_dict_to_csv(
            {COL_STEP: steps, COL_REMAINING: rem,
             COL_SALIENCY: sal_avg, COL_ACCURACY: acc_list},
            filename=csv_avg,
        )

        if acc < COLLAPSE_THRESHOLD:
            collapse_density = remaining
            print(f"\n  [{arch_name}] *** Collapse: acc={acc:.2f}%  "
                  f"remaining={remaining:.4f}% ***")
            break

    if collapse_density is None:
        collapse_density = get_custom_model_sparsity_percent(model)
        print(f"  [{arch_name}] Warning: collapse never reached. "
              f"Using final density {collapse_density:.4f}%.")

    print(f"  [{arch_name}] Experiment A done. "
          f"Steps: {step}  Collapse density: {collapse_density:.4f}%")
    return collapse_density


# -----------------------------------------------------------------------
# Experiment B: random pruning static (control) to collapse density
# -----------------------------------------------------------------------

def run_arch_static(
    hidden1: int,
    hidden2: int,
    run_ctx: NplhRunContext,
    target_density: float,
) -> None:
    """
    Train from scratch for TRAIN_EPOCHS, then randomly prune with NO
    retraining, down to target_density%.  Records avg |w| at each step.

    Parameters
    ----------
    target_density : float — collapse density from experiment A.
    """
    arch_name = f"lenet_{hidden1}_{hidden2}"
    print(f"\n  [{arch_name}] Experiment B: random prune static "
          f"(target={target_density:.4f}%)")

    model           = _build_model(hidden1, hidden2)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.MNIST,
        configs=dataset_context_configs_mnist(),
    )

    # Dense training (same initialisation point as experiment A)
    print(f"  [{arch_name}] Dense training ({TRAIN_EPOCHS} epochs) ...")
    baseline_acc = _train_dense(model, dataset_context, TRAIN_EPOCHS)
    print(f"  [{arch_name}] Baseline accuracy: {baseline_acc:.2f}%")

    policy  = RandomPruningPolicy()
    csv_avg = run_ctx.csv_path(arch_name, DATASET_NAME, SAL_AVG, METHOD_STATIC)

    steps = []; sal_avg = []; rem = []; acc_list = []
    step  = 0

    while get_custom_model_sparsity_percent(model) > target_density:
        try:
            result    = policy.prune_step(model, PRUNING_RATE)
            remaining = get_custom_model_sparsity_percent(model)
            step += 1
        except (ValueError, RuntimeError) as exc:
            print(f"  [{arch_name}] Pruning stopped at step {step}: {exc}")
            break

        # Evaluate without training
        dataset_context.init_data_split()
        acc = _test_epoch(model, dataset_context, step)

        print(
            f"  [{arch_name}] step {step:3d}  remaining={remaining:.4f}%  "
            f"avg_mag={result.avg_saliency:.4e}  acc={acc:.2f}%"
        )

        steps.append(step);   sal_avg.append(result.avg_saliency)
        rem.append(remaining); acc_list.append(acc)

        save_dict_to_csv(
            {COL_STEP: steps, COL_REMAINING: rem,
             COL_SALIENCY: sal_avg, COL_ACCURACY: acc_list},
            filename=csv_avg,
        )

    print(f"  [{arch_name}] Experiment B done. Steps: {step}  "
          f"Final density: {get_custom_model_sparsity_percent(model):.4f}%")
