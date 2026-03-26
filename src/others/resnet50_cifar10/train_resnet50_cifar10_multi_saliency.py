"""
train_resnet50_cifar10_multi_saliency.py
=========================================
Random pruning + retraining on ResNet50 CIFAR-10, recording THREE
saliency metrics simultaneously at each pruning step:

  1. Magnitude avg  —  mean |w|         of active weights
  2. Taylor avg     —  mean |w · ∂L/∂w| of active weights
  3. Gradient avg   —  mean |∂L/∂w|     of active weights

All three are computed at the same instant (after the previous round of
fine-tuning, immediately before the next random pruning step) using a
single forward+backward pass.

Outputs (3 CSVs):
  resnet50_cifar10_avg_random_prune_magnitude.csv
  resnet50_cifar10_avg_random_prune_taylor.csv
  resnet50_cifar10_avg_random_prune_gradient.csv

Each CSV has columns: Step, RemainingParams, Saliency, Accuracy.

Constants
---------
  PRUNING_RATE   = 0.10   (10% of remaining weights per step)
  FINETUNE_EPOCHS = 3     (fine-tuning epochs between pruning steps)
  FINETUNE_LR    = 0.001  (constant LR throughout)
  COLLAPSE_THRESHOLD = 15.0%  (stop when accuracy drops below this)
"""

import torch
import torch.nn as nn

from src.infrastructure.constants import BASELINE_MODELS_PATH, WEIGHTS_ATTR, MASK_ATTR
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType,
)
from src.infrastructure.layers import ConfigsNetworkMask, get_layers_primitive
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

MULTI_PRUNING_RATE        = 0.10
MULTI_FINETUNE_EPOCHS     = 3
MULTI_LR                  = 0.001
MULTI_COLLAPSE_THRESHOLD  = 15.0

METHOD_MULTI_MAGNITUDE = "random_prune_magnitude"
METHOD_MULTI_TAYLOR    = "random_prune_taylor"
METHOD_MULTI_GRADIENT  = "random_prune_gradient"


# -----------------------------------------------------------------------
# Model loader
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
# Multi-saliency computation  (one backprop pass → 3 numbers)
# -----------------------------------------------------------------------

def _compute_all_saliencies(
    model: nn.Module,
    criterion: nn.Module,
    get_batch,
) -> tuple[float, float, float]:
    """
    Compute all three average saliency metrics over active weights.

    Returns
    -------
    (mag_avg, taylor_avg, grad_avg)
      mag_avg    = mean |w|
      taylor_avg = mean |w · ∂L/∂w|
      grad_avg   = mean |∂L/∂w|
    """
    layers = [
        layer for layer in get_layers_primitive(model)
        if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, MASK_ATTR)
    ]

    # ── Magnitude saliency (no backprop needed) ───────────────────────
    mags = []
    for layer in layers:
        w = getattr(layer, WEIGHTS_ATTR)
        m = getattr(layer, MASK_ATTR)
        active = m.data >= 0
        if active.any():
            mags.append(w.data[active].abs().flatten())
    mag_avg = float(torch.cat(mags).mean().item()) if mags else 0.0

    # ── One forward+backward pass for Taylor + Gradient ──────────────
    weight_params = [getattr(layer, WEIGHTS_ATTR) for layer in layers]
    orig_req_grad = [p.requires_grad for p in weight_params]
    for p in weight_params:
        p.requires_grad_(True)

    model.eval()
    model.zero_grad()
    inputs, targets = get_batch()
    loss = criterion(model(inputs), targets)
    loss.backward()

    taylor_vals = []
    grad_vals   = []
    for layer in layers:
        w = getattr(layer, WEIGHTS_ATTR)
        m = getattr(layer, MASK_ATTR)
        active = m.data >= 0
        g = w.grad
        if g is None or not active.any():
            continue
        w_active = w.data[active].flatten()
        g_active = g.detach()[active].flatten()
        taylor_vals.append((w_active * g_active).abs())
        grad_vals.append(g_active.abs())

    taylor_avg = float(torch.cat(taylor_vals).mean().item()) if taylor_vals else 0.0
    grad_avg   = float(torch.cat(grad_vals).mean().item()) if grad_vals else 0.0

    model.zero_grad()
    for p, req in zip(weight_params, orig_req_grad):
        p.requires_grad_(req)

    return mag_avg, taylor_avg, grad_avg


# -----------------------------------------------------------------------
# Main experiment: random pruning + retraining, 3 saliencies in parallel
# -----------------------------------------------------------------------

def run_random_pruning_multi_saliency(
    baseline_name: str,
    run_ctx: NplhRunContext,
) -> float:
    """
    Random pruning with fine-tuning, recording magnitude / Taylor / gradient
    saliency simultaneously at each step.

    Returns
    -------
    collapse_density : float  — remaining% at the collapse step.
    """
    print(f"\n{'='*60}")
    print("Random pruning + training — multi-saliency (magnitude / Taylor / gradient)")
    print(f"  pruning_rate={MULTI_PRUNING_RATE}  "
          f"finetune_epochs={MULTI_FINETUNE_EPOCHS}  LR={MULTI_LR}")
    print(f"  collapse threshold={MULTI_COLLAPSE_THRESHOLD}%")
    print(f"{'='*60}")

    model = _load_model(baseline_name)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )
    policy    = RandomPruningPolicy()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        get_model_weights_params(model),
        lr=MULTI_LR, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )

    csv_mag = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, METHOD_MULTI_MAGNITUDE)
    csv_tay = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, METHOD_MULTI_TAYLOR)
    csv_grd = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, METHOD_MULTI_GRADIENT)

    steps = []; rem = []; acc_list = []
    sal_mag = []; sal_tay = []; sal_grd = []
    step             = 0
    collapse_density = None
    global_epoch     = 0

    while True:
        # ── 1. Compute all 3 saliencies before pruning ────────────────
        def _get_batch():
            dataset_context.init_data_split()
            return dataset_context.get_training_data_and_labels()

        mag_avg, taylor_avg, grad_avg = _compute_all_saliencies(
            model, criterion, _get_batch,
        )

        # ── 2. Prune randomly ─────────────────────────────────────────
        try:
            policy.prune_step(model, MULTI_PRUNING_RATE)
        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {step}: {exc}")
            break

        remaining = get_custom_model_sparsity_percent(model)
        step += 1

        # ── 3. Fine-tune ──────────────────────────────────────────────
        acc = 0.0
        for _ in range(MULTI_FINETUNE_EPOCHS):
            global_epoch += 1
            dataset_context.init_data_split()
            _train_epoch(model, dataset_context, optimizer)
            acc = _test_epoch(model, dataset_context, global_epoch)

        print(
            f"  [step {step:3d}] remaining={remaining:.4f}%  "
            f"mag={mag_avg:.4e}  taylor={taylor_avg:.4e}  grad={grad_avg:.4e}  acc={acc:.2f}%"
        )

        steps.append(step);        rem.append(remaining);   acc_list.append(acc)
        sal_mag.append(mag_avg);   sal_tay.append(taylor_avg); sal_grd.append(grad_avg)

        base = {COL_STEP: steps, COL_REMAINING: rem, COL_ACCURACY: acc_list}
        save_dict_to_csv({**base, COL_SALIENCY: sal_mag}, filename=csv_mag)
        save_dict_to_csv({**base, COL_SALIENCY: sal_tay}, filename=csv_tay)
        save_dict_to_csv({**base, COL_SALIENCY: sal_grd}, filename=csv_grd)

        # ── 4. Check collapse ─────────────────────────────────────────
        if acc < MULTI_COLLAPSE_THRESHOLD:
            collapse_density = remaining
            print(f"\n  *** Collapse at step {step}: acc={acc:.2f}%  "
                  f"remaining={remaining:.4f}% ***")
            break

    if collapse_density is None:
        collapse_density = get_custom_model_sparsity_percent(model)
        print(f"  Warning: collapse never reached. Using final density {collapse_density:.4f}%.")

    print(f"\n  Done. Steps: {step}  Collapse density: {collapse_density:.4f}%")
    print(f"  CSVs:")
    print(f"    {csv_mag}")
    print(f"    {csv_tay}")
    print(f"    {csv_grd}")
    return collapse_density
