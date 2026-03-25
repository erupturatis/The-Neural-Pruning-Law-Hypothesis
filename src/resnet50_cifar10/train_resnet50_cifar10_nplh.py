"""
train_resnet50_cifar10_nplh.py
==============================
Device-agnostic training and NPLH experiment functions for ResNet50 on CIFAR-10.

Three experiment modes
----------------------
  1. Static  – magnitude pruning, NO retraining (control baseline)
  2. IMP     – magnitude pruning with fine-tuning between steps
  3. Taylor  – first-order Taylor pruning with fine-tuning between steps

All experiments share the same pre-trained baseline model.
Run via run_resnet50_cifar10_nplh_experiment.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K

from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_relu
from src.infrastructure.constants import BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, DatasetContextConfigs,
)
from src.infrastructure.layers import (
    ConfigsNetworkMasksImportance, calculate_pruning_epochs,
)
from src.infrastructure.nplh_run_context import (
    NplhRunContext,
    COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
    SAL_MIN, SAL_AVG,
    METHOD_IMP_STATIC, METHOD_IMP_MAGNITUDE, METHOD_IMP_TAYLOR,
    METHOD_GRADIENT_STATIC, METHOD_TAYLOR_STATIC,
    METHOD_RANDOM_STATIC_MAGNITUDE, METHOD_RANDOM_STATIC_GRADIENT, METHOD_RANDOM_STATIC_TAYLOR,
    METHOD_RANDOM_RETRAIN_MAGNITUDE, METHOD_RANDOM_RETRAIN_GRADIENT, METHOD_RANDOM_RETRAIN_TAYLOR,
)
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent
from src.infrastructure.pruning_policy import (
    MagnitudePruningPolicy, TaylorPruningPolicy, PruningPolicy,
    GradientPruningPolicy, RandomRegrowthPruningPolicy,
    RandomPruningPolicy, measure_multi_saliency,
)
from src.infrastructure.read_write import save_dict_to_csv
from src.infrastructure.training_common import get_model_weights_params
from src.resnet50_cifar10.resnet50_cifar10_class import Resnet50Cifar10

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
MODEL_NAME         = "resnet50"
DATASET_NAME       = "cifar10"

SCRATCH_EPOCHS     = 160     # dense training epochs for baseline
PRUNING_RATE       = 0.05    # fraction of remaining weights pruned each step
TARGET_SPARSITY    = 0.998   # remove 99.8% → leave 0.2% remaining
IMP_EPOCHS         = 200     # fine-tuning epochs for IMP / Taylor

BASELINE_SAVE_NAME = "resnet50_cifar10_nplh_baseline"

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD  = [0.247,  0.243,  0.261]


# -----------------------------------------------------------------------
# MPS-compatible CIFAR-10 augmentations
# (kornia.RandomCrop uses get_perspective_transform which fails on MPS)
# -----------------------------------------------------------------------

class _RandomCropPad(nn.Module):
    """Drop-in for kornia.RandomCrop(32, padding=4) that works on MPS."""
    def __init__(self, size: int, padding: int):
        super().__init__()
        self.size    = size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, [self.padding] * 4, mode="reflect")
        _, _, h, w = x.shape
        top  = torch.randint(0, h - self.size + 1, (1,)).item()
        left = torch.randint(0, w - self.size + 1, (1,)).item()
        return x[:, :, top:top + self.size, left:left + self.size]


def _make_cifar10_dataset_configs() -> DatasetContextConfigs:
    """Build CIFAR-10 dataset configs with MPS-compatible augmentations."""
    device = get_device()
    augmentations = nn.Sequential(
        _RandomCropPad(32, padding=4),
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ).to(device)
    augmentations_test = nn.Sequential(
        K.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ).to(device)
    return DatasetContextConfigs(
        batch_size=128,
        augmentations=augmentations,
        augmentations_test=augmentations_test,
    )


# -----------------------------------------------------------------------
# Device-agnostic training helpers (no GradScaler — works on MPS + CPU)
# -----------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    dataset_context: DatasetSmallContext,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()


def _test_epoch(
    model: nn.Module,
    dataset_context: DatasetSmallContext,
    epoch: int,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        while dataset_context.any_data_testing_available():
            data, target = dataset_context.get_testing_data_and_labels()
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total    = dataset_context.get_data_testing_length()
    accuracy = 100.0 * correct / total
    print(f"  [epoch {epoch:3d}] acc: {correct}/{total} = {accuracy:.2f}%")
    return accuracy


# -----------------------------------------------------------------------
# Baseline: train from scratch, device-agnostic, no wandb
# -----------------------------------------------------------------------

def train_resnet50_cifar10_baseline() -> str:
    """
    Train ResNet50 on CIFAR-10 from scratch and save to networks_baseline/.

    Uses device-agnostic training (no GradScaler/autocast) so it runs on
    CUDA, MPS (MacBook GPU), or CPU.  WandB is bypassed entirely.

    Returns
    -------
    save_name : str  (pass to _load_fresh_model or model.load())
    """
    print("\n=== Training ResNet50 baseline from scratch ===")
    configs_layers_initialization_all_kaiming_relu()

    device = get_device()
    print(f"  Device: {device}")

    configs_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    model = Resnet50Cifar10(configs_masks).to(device)

    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )

    optimizer = torch.optim.SGD(
        get_model_weights_params(model),
        lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[SCRATCH_EPOCHS // 2, SCRATCH_EPOCHS * 3 // 4],
    )

    acc = 0.0
    for epoch in range(1, SCRATCH_EPOCHS + 1):
        dataset_context.init_data_split()
        _train_epoch(model, dataset_context, optimizer)
        acc = _test_epoch(model, dataset_context, epoch)
        scheduler.step()

    model.save(name=BASELINE_SAVE_NAME, folder=BASELINE_MODELS_PATH)
    print(f"\n  Baseline saved → {BASELINE_SAVE_NAME}  (acc={acc:.2f}%)")
    return BASELINE_SAVE_NAME


# -----------------------------------------------------------------------
# Helper: reload baseline weights onto a fresh model instance
# -----------------------------------------------------------------------

def _load_fresh_model(baseline_name: str) -> Resnet50Cifar10:
    device = get_device()
    configs_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    model = Resnet50Cifar10(configs_masks).to(device)
    model.load(baseline_name, BASELINE_MODELS_PATH)
    return model


# -----------------------------------------------------------------------
# Experiment 1: Static pruning (magnitude, no retraining) — control
# -----------------------------------------------------------------------

def run_static_pruning(baseline_name: str, run_ctx: NplhRunContext) -> None:
    """
    Prune 5% of remaining weights at each step using magnitude, NO retraining.
    Stop when ≤ 0.2% of weights remain.  Records saliency and accuracy after
    each pruning event.
    """
    print(f"\n{'='*60}")
    print("Experiment 1: Static pruning (no retraining)")
    print(f"{'='*60}")

    model = _load_fresh_model(baseline_name)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )
    policy = MagnitudePruningPolicy()

    csv_min = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_MIN, METHOD_IMP_STATIC)
    csv_avg = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, METHOD_IMP_STATIC)

    steps_min = []; sal_min = []; rem_min = []; acc_min = []
    steps_avg = []; sal_avg = []; rem_avg = []; acc_avg = []
    step = 0

    target_remaining = (1.0 - TARGET_SPARSITY) * 100  # 0.2%

    while get_custom_model_sparsity_percent(model) > target_remaining:
        try:
            result    = policy.prune_step(model, PRUNING_RATE)
            remaining = get_custom_model_sparsity_percent(model)
            step += 1

            dataset_context.init_data_split()
            acc = _test_epoch(model, dataset_context, step)

            steps_min.append(step); sal_min.append(result.threshold)
            rem_min.append(remaining); acc_min.append(acc)

            steps_avg.append(step); sal_avg.append(result.avg_saliency)
            rem_avg.append(remaining); acc_avg.append(acc)

            print(
                f"  [step {step:3d}] remaining={remaining:.3f}%  "
                f"threshold={result.threshold:.4e}  avg={result.avg_saliency:.4e}"
            )

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

        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {step}: {exc}")
            break

    print(f"  Static pruning done.  Steps: {step}")


# -----------------------------------------------------------------------
# Shared IMP loop (magnitude or Taylor)
# -----------------------------------------------------------------------

def _run_imp_experiment(
    baseline_name: str,
    run_ctx: NplhRunContext,
    policy: PruningPolicy,
    imp_epochs: int = IMP_EPOCHS,
) -> None:
    """
    Load baseline, fine-tune with IMP for imp_epochs.

    Pruning events are evenly spaced across imp_epochs via
    calculate_pruning_epochs.  Records (step, remaining%, saliency, accuracy)
    at every pruning event and saves incrementally to two CSVs.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: IMP  method={policy.method_tag}")
    print(f"{'='*60}")

    model = _load_fresh_model(baseline_name)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )

    optimizer = torch.optim.SGD(
        get_model_weights_params(model),
        lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=imp_epochs, eta_min=1e-5,
    )

    pruning_epochs = set(calculate_pruning_epochs(
        target_sparsity=TARGET_SPARSITY,
        pruning_rate=PRUNING_RATE,
        total_epochs=imp_epochs,
        start_epoch=1,
    ))
    print(f"  {len(pruning_epochs)} pruning steps over {imp_epochs} epochs.")

    csv_min = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_MIN, policy.method_tag)
    csv_avg = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, policy.method_tag)

    steps_min = []; sal_min = []; rem_min = []; acc_min = []
    steps_avg = []; sal_avg = []; rem_avg = []; acc_avg = []
    pruning_step = 0

    for epoch in range(1, imp_epochs + 1):
        dataset_context.init_data_split()
        _train_epoch(model, dataset_context, optimizer)
        acc = _test_epoch(model, dataset_context, epoch)
        scheduler.step()

        if epoch in pruning_epochs:
            try:
                def _get_batch():
                    dataset_context.init_data_split()
                    return dataset_context.get_training_data_and_labels()

                result    = policy.prune_step(model, PRUNING_RATE, get_batch=_get_batch)
                remaining = get_custom_model_sparsity_percent(model)
                pruning_step += 1

                steps_min.append(pruning_step); sal_min.append(result.threshold)
                rem_min.append(remaining);       acc_min.append(acc)

                steps_avg.append(pruning_step); sal_avg.append(result.avg_saliency)
                rem_avg.append(remaining);       acc_avg.append(acc)

                print(
                    f"  [step {pruning_step:3d}] epoch {epoch}: "
                    f"remaining={remaining:.3f}%  "
                    f"threshold={result.threshold:.4e}  avg={result.avg_saliency:.4e}"
                )

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

            except (ValueError, RuntimeError) as exc:
                print(f"  Pruning stopped early at epoch {epoch}: {exc}")
                break

    print(f"  {policy.method_tag} done.  Pruning steps: {pruning_step}")


# -----------------------------------------------------------------------
# Experiment 2: IMP – magnitude
# -----------------------------------------------------------------------

def run_imp_magnitude(baseline_name: str, run_ctx: NplhRunContext) -> None:
    _run_imp_experiment(
        baseline_name=baseline_name,
        run_ctx=run_ctx,
        policy=MagnitudePruningPolicy(),
    )


# -----------------------------------------------------------------------
# Experiment 3: IMP – Taylor
# -----------------------------------------------------------------------

def run_imp_taylor(baseline_name: str, run_ctx: NplhRunContext) -> None:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    _run_imp_experiment(
        baseline_name=baseline_name,
        run_ctx=run_ctx,
        policy=TaylorPruningPolicy(criterion),
    )


# -----------------------------------------------------------------------
# Experiment 4: Gradient-magnitude pruning
# -----------------------------------------------------------------------

def run_gradient_pruning(baseline_name: str, run_ctx: NplhRunContext) -> None:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    _run_imp_experiment(
        baseline_name=baseline_name,
        run_ctx=run_ctx,
        policy=GradientPruningPolicy(criterion),
    )


# -----------------------------------------------------------------------
# Experiment 5: Random prune + gradient-guided regrowth
# -----------------------------------------------------------------------

def run_random_regrowth(baseline_name: str, run_ctx: NplhRunContext) -> None:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    _run_imp_experiment(
        baseline_name=baseline_name,
        run_ctx=run_ctx,
        policy=RandomRegrowthPruningPolicy(criterion, oversample_factor=2.0),
    )


# -----------------------------------------------------------------------
# Static-scored pruning (no retraining) — shared helper + two variants
# -----------------------------------------------------------------------

def _run_static_scored_experiment(
    baseline_name: str,
    run_ctx: NplhRunContext,
    policy: PruningPolicy,
    method_tag: str,
) -> None:
    """
    Prune 5% of remaining weights at each step using the given scored policy,
    with NO retraining between steps.  Evaluates accuracy after each step.
    Stops when ≤ 0.2% of weights remain.

    Parameters
    ----------
    method_tag : str
        Override the policy's own method_tag for CSV filenames so that
        static variants get distinct names (e.g. "gradient_static").
    """
    print(f"\n{'='*60}")
    print(f"Static scored pruning (no retraining)  method={method_tag}")
    print(f"{'='*60}")

    model = _load_fresh_model(baseline_name)
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )

    csv_min = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_MIN, method_tag)
    csv_avg = run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, method_tag)

    steps_min = []; sal_min = []; rem_min = []; acc_min = []
    steps_avg = []; sal_avg = []; rem_avg = []; acc_avg = []
    step = 0

    target_remaining = (1.0 - TARGET_SPARSITY) * 100  # 0.2%

    while get_custom_model_sparsity_percent(model) > target_remaining:
        try:
            def _get_batch():
                dataset_context.init_data_split()
                return dataset_context.get_training_data_and_labels()

            result    = policy.prune_step(model, PRUNING_RATE, get_batch=_get_batch)
            remaining = get_custom_model_sparsity_percent(model)
            step += 1

            dataset_context.init_data_split()
            acc = _test_epoch(model, dataset_context, step)

            steps_min.append(step); sal_min.append(result.threshold)
            rem_min.append(remaining); acc_min.append(acc)

            steps_avg.append(step); sal_avg.append(result.avg_saliency)
            rem_avg.append(remaining); acc_avg.append(acc)

            print(
                f"  [step {step:3d}] remaining={remaining:.3f}%  "
                f"threshold={result.threshold:.4e}  avg={result.avg_saliency:.4e}"
            )

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

        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {step}: {exc}")
            break

    print(f"  {method_tag} static done.  Steps: {step}")


def run_static_gradient(baseline_name: str, run_ctx: NplhRunContext) -> None:
    """Static gradient-magnitude pruning — score = |∂L/∂w|, no retraining."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    _run_static_scored_experiment(
        baseline_name=baseline_name,
        run_ctx=run_ctx,
        policy=GradientPruningPolicy(criterion),
        method_tag=METHOD_GRADIENT_STATIC,
    )


def run_static_taylor(baseline_name: str, run_ctx: NplhRunContext) -> None:
    """Static Taylor pruning — score = |w · ∂L/∂w|, no retraining."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    _run_static_scored_experiment(
        baseline_name=baseline_name,
        run_ctx=run_ctx,
        policy=TaylorPruningPolicy(criterion),
        method_tag=METHOD_TAYLOR_STATIC,
    )


# -----------------------------------------------------------------------
# Random pruning helpers — shared CSV path/list initialisation
# -----------------------------------------------------------------------

def _make_random_csv_paths(run_ctx: NplhRunContext, retrain: bool):
    """Return CSV path dict keyed by metric name."""
    if retrain:
        tags = (METHOD_RANDOM_RETRAIN_MAGNITUDE,
                METHOD_RANDOM_RETRAIN_GRADIENT,
                METHOD_RANDOM_RETRAIN_TAYLOR)
    else:
        tags = (METHOD_RANDOM_STATIC_MAGNITUDE,
                METHOD_RANDOM_STATIC_GRADIENT,
                METHOD_RANDOM_STATIC_TAYLOR)
    mag_tag, grad_tag, taylor_tag = tags
    return {
        "mag_min":    run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_MIN, mag_tag),
        "mag_avg":    run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, mag_tag),
        "grad_min":   run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_MIN, grad_tag),
        "grad_avg":   run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, grad_tag),
        "taylor_min": run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_MIN, taylor_tag),
        "taylor_avg": run_ctx.csv_path(MODEL_NAME, DATASET_NAME, SAL_AVG, taylor_tag),
    }


def _save_random_step(csv_paths, step, remaining, acc, ms):
    """Save one pruning step's multi-saliency data to all 6 CSVs."""
    base = {COL_STEP: [step], COL_REMAINING: [remaining], COL_ACCURACY: [acc]}
    save_dict_to_csv({**base, COL_SALIENCY: [ms.mag_min]},    csv_paths["mag_min"])
    save_dict_to_csv({**base, COL_SALIENCY: [ms.mag_avg]},    csv_paths["mag_avg"])
    save_dict_to_csv({**base, COL_SALIENCY: [ms.grad_min]},   csv_paths["grad_min"])
    save_dict_to_csv({**base, COL_SALIENCY: [ms.grad_avg]},   csv_paths["grad_avg"])
    save_dict_to_csv({**base, COL_SALIENCY: [ms.taylor_min]}, csv_paths["taylor_min"])
    save_dict_to_csv({**base, COL_SALIENCY: [ms.taylor_avg]}, csv_paths["taylor_avg"])


# -----------------------------------------------------------------------
# Experiment 6: Random pruning — NO retraining
# -----------------------------------------------------------------------

def run_random_static(baseline_name: str, run_ctx: NplhRunContext) -> None:
    """
    Prune 5% of remaining weights uniformly at random at each step.
    No retraining.  After each step, measure magnitude, gradient, and
    Taylor saliency on the remaining active weights and save 6 CSVs.
    """
    print(f"\n{'='*60}")
    print("Experiment: Random pruning (no retraining)")
    print(f"{'='*60}")

    model         = _load_fresh_model(baseline_name)
    criterion     = nn.CrossEntropyLoss(label_smoothing=0.05)
    policy        = RandomPruningPolicy()
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )

    csv_paths        = _make_random_csv_paths(run_ctx, retrain=False)
    target_remaining = (1.0 - TARGET_SPARSITY) * 100
    step             = 0

    while get_custom_model_sparsity_percent(model) > target_remaining:
        try:
            policy.prune_step(model, PRUNING_RATE)
            remaining = get_custom_model_sparsity_percent(model)
            step += 1

            def _get_batch():
                dataset_context.init_data_split()
                return dataset_context.get_training_data_and_labels()

            ms = measure_multi_saliency(model, criterion, _get_batch)

            dataset_context.init_data_split()
            acc = _test_epoch(model, dataset_context, step)

            print(
                f"  [step {step:3d}] remaining={remaining:.3f}%  "
                f"mag_avg={ms.mag_avg:.4e}  grad_avg={ms.grad_avg:.4e}  "
                f"taylor_avg={ms.taylor_avg:.4e}"
            )

            _save_random_step(csv_paths, step, remaining, acc, ms)

        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {step}: {exc}")
            break

    print(f"  Random static done.  Steps: {step}")


# -----------------------------------------------------------------------
# Experiment 7: Random pruning — WITH retraining
# -----------------------------------------------------------------------

def run_random_retraining(baseline_name: str, run_ctx: NplhRunContext) -> None:
    """
    Prune 5% of remaining weights uniformly at random on an IMP schedule
    (pruning events evenly spaced over IMP_EPOCHS, with SGD fine-tuning
    between steps).  After each pruning event, measure magnitude, gradient,
    and Taylor saliency on the remaining active weights and save 6 CSVs.
    """
    print(f"\n{'='*60}")
    print("Experiment: Random pruning (with retraining)")
    print(f"{'='*60}")

    model         = _load_fresh_model(baseline_name)
    criterion     = nn.CrossEntropyLoss(label_smoothing=0.05)
    policy        = RandomPruningPolicy()
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR10,
        configs=_make_cifar10_dataset_configs(),
    )

    optimizer = torch.optim.SGD(
        get_model_weights_params(model),
        lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=IMP_EPOCHS, eta_min=1e-5,
    )

    pruning_epochs = set(calculate_pruning_epochs(
        target_sparsity=TARGET_SPARSITY,
        pruning_rate=PRUNING_RATE,
        total_epochs=IMP_EPOCHS,
        start_epoch=1,
    ))
    print(f"  {len(pruning_epochs)} pruning steps over {IMP_EPOCHS} epochs.")

    csv_paths    = _make_random_csv_paths(run_ctx, retrain=True)
    pruning_step = 0

    for epoch in range(1, IMP_EPOCHS + 1):
        dataset_context.init_data_split()
        _train_epoch(model, dataset_context, optimizer)
        acc = _test_epoch(model, dataset_context, epoch)
        scheduler.step()

        if epoch in pruning_epochs:
            try:
                policy.prune_step(model, PRUNING_RATE)
                remaining = get_custom_model_sparsity_percent(model)
                pruning_step += 1

                def _get_batch():
                    dataset_context.init_data_split()
                    return dataset_context.get_training_data_and_labels()

                ms = measure_multi_saliency(model, criterion, _get_batch)

                print(
                    f"  [step {pruning_step:3d}] epoch {epoch}: "
                    f"remaining={remaining:.3f}%  "
                    f"mag_avg={ms.mag_avg:.4e}  grad_avg={ms.grad_avg:.4e}  "
                    f"taylor_avg={ms.taylor_avg:.4e}"
                )

                _save_random_step(csv_paths, pruning_step, remaining, acc, ms)

            except (ValueError, RuntimeError) as exc:
                print(f"  Pruning stopped early at epoch {epoch}: {exc}")
                break

    print(f"  Random retraining done.  Pruning steps: {pruning_step}")

