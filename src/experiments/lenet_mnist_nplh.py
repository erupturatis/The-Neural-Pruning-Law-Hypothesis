"""
Self-contained NPLH iterative pruning experiment for LeNet on MNIST.

Loop structure per round:
  1. Prune        – opaque PruningPolicy, removes a fraction of remaining weights
  2. Converge     – freeze mask params, train weights only until accuracy plateaus
  3. Saliency     – measure magnitude / gradient / Taylor scores on active weights
  Repeat until target sparsity or max rounds reached.
"""

import torch
import torch.nn as nn
from collections import deque
from statistics import stdev

from numpy.ma.core import sometrue

from src.common_files_experiments.load_save import load_model_entire_dict
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, dataset_context_configs_mnist,
)
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device
from src.infrastructure.pruning_policy import (
    MagnitudePruningPolicy, PruningPolicy, measure_multi_saliency,
)
from src.infrastructure.saliency_measurement_policy import SaliencyMeasurementPolicy
from src.infrastructure.training_convergence_policy import TrainingConvergencePolicy

from src.infrastructure.policies.nplh_stopping_policy import NPLHStoppingPolicy
from src.infrastructure.training_utils import get_model_flow_params_and_weights_params
from src.infrastructure.constants import BASELINE_MODELS_PATH, PRUNED_MODELS_PATH
from src.model_lenet.model_lenet300_class import ModelLenet300
from src.experiments.utils import get_model_sparsity
from src.model_lenet.model_lenetVariable_class import ModelLenetVariable

# ── Config ───────────────────────────────────────────────────────────────────

# THIS ONE NEEDS FIXING
experiment_context = {
LR_DENSE          = 1e-3
LR_FINETUNE       = 1e-3
DENSE_EPOCHS      = 60

MAX_ROUNDS        = 20
TARGET_SPARSITY   = 5.0     # stop when remaining% drops below this

# Convergence criterion: train until the last CONV_WINDOW accuracies have
# stdev < CONV_TOL, or until CONV_MAX_EPOCHS is reached.
CONV_MAX_EPOCHS   = 40
CONV_WINDOW       = 6
CONV_TOL          = 0.3     # stop if stdev of last window < this (%)
EPOCHS = 0

}
# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_model() -> ModelLenet300:
    cfg = ConfigsNetworkMask(mask_apply_enabled=True, mask_training_enabled=False, weights_training_enabled=True)
    return ModelLenet300(cfg).to(get_device())

def _make_dataset() -> DatasetSmallContext:
    return DatasetSmallContext(dataset=DatasetSmallType.MNIST, configs=dataset_context_configs_mnist())

def _train_one_epoch(model, dataset, optimizer, criterion):
    model.train()
    dataset.init_data_split()
    while dataset.any_data_training_available():
        data, target = dataset.get_training_data_and_labels()
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

def _test(model, dataset) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        while dataset.any_data_testing_available():
            data, target = dataset.get_testing_data_and_labels()
            out = model(data)
            total_loss += criterion(out, target).item()
            correct += out.argmax(dim=1).eq(target).sum().item()
    n = dataset.get_data_testing_length()
    acc = 100.0 * correct / n
    print(f"    acc={acc:.2f}%  remaining={get_model_sparsity(model):.1f}%")
    return acc

def _train_till_convergence(model, dataset, optimizer, criterion) -> float:
    """
    Train weights-only until accuracy stdev over the last CONV_WINDOW epochs
    drops below CONV_TOL, or CONV_MAX_EPOCHS is reached.
    """
    recent = deque(maxlen=CONV_WINDOW)
    acc = 0.0
    for epoch in range(1, CONV_MAX_EPOCHS + 1):
        _train_one_epoch(model, dataset, optimizer, criterion)
        acc = _test(model, dataset)
        recent.append(acc)
        if len(recent) == CONV_WINDOW and stdev(recent) < CONV_TOL:
            print(f"    Converged at epoch {epoch}  (stdev={stdev(recent):.3f})")
            break
    return acc

# ── Dense training ────────────────────────────────────────────────────────────
def train_dense_lenet_mnist():
    """Train a fresh LeNet300 on MNIST from scratch and save it."""
    model   = _make_model()
    dataset = _make_dataset()
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR_DENSE)

    acc = 0.0
    for epoch in range(1, DENSE_EPOCHS + 1):
        _train_one_epoch(model, dataset, optimizer, criterion)
        acc = _test(model, dataset)
        print(f"  Epoch {epoch}/{DENSE_EPOCHS}")

    model.save(f"lenet300_mnist_dense_acc{acc:.1f}", BASELINE_MODELS_PATH)
    print(f"Saved dense model  acc={acc:.2f}%")
    return model

# ── NPLH experiment ───────────────────────────────────────────────────────────

def load_dense_model(load_dense_name: str) -> ModelLenetVariable:
    model   = _make_model()
    if load_dense_name is None:
        raise Exception("No model name provided")

    load_model_entire_dict(model, load_dense_name, BASELINE_MODELS_PATH)
    print(f"Loaded: {load_dense_name}")

    return model

def experiment_lenet_mnist_nplh():
    dataset = _make_dataset()
    pass

def train_dense_from_scratch():
    print("Training dense model from scratch …")
    optimizer_dense = torch.optim.Adam(model.parameters(), lr=LR_DENSE)
    for epoch in range(1, DENSE_EPOCHS + 1):
        _train_one_epoch(model, dataset, optimizer_dense, criterion)
        if epoch % 10 == 0:
            _test(model, dataset)
    acc = _test(model, dataset)

def lenet_mnist_nplh(
    model: ModelLenetVariable,
    dataset: DatasetSmallContext,
    pruning_policy: PruningPolicy,
    training_convergence_policy: TrainingConvergencePolicy,
    saliency_measurement_policy: SaliencyMeasurementPolicy,
    nplh_stopping_policy: NPLHStoppingPolicy,
):
    if pruning_policy is None:
        raise Exception("No pruning policy given")

    if training_convergence_policy is None:
        raise Exception("No training convergence policy")

    if saliency_measurement_policy is None:
        raise Exception("No saliency measurement policy")

    # criterion = nn.CrossEntropyLoss()

    # ── Weights-only optimizer for convergence stages ────────────────────────
    # weight_params, _ = get_model_flow_params_and_weights_params(model)
    # optimizer_weights = torch.optim.Adam(weight_params, lr=LR_FINETUNE)

    # def get_batch():
    #     # Returns a fresh single batch for data-dependent pruning policies.
    #     dataset.init_data_split()
    #     return dataset.get_training_data_and_labels()

    # ── Main loop ────────────────────────────────────────────────────────────
    for round_idx in range(1, MAX_ROUNDS + 1):
        remaining = get_model_sparsity(model)
        print(f"\n=== Round {round_idx}/{MAX_ROUNDS}  |  remaining={remaining:.1f}% ===")

        if nplh_stopping_policy.finish_training(model, dataset, experiment_context):
            print("NPLH stopping policy reached.")
            break

        # 1. Prune (opaque – policy decides the criterion)
        prune_result = pruning_policy.prune_step(model)
        print(f"  [prune]  threshold={prune_result.threshold:.5f}"
              f"  avg_saliency={prune_result.avg_saliency:.5f}"
              f"  remaining={get_model_sparsity(model):.1f}%")

        # 2. Train weights only until convergence
        print("  [converge]")
        acc = _train_till_convergence(model, dataset, optimizer_weights, criterion)

        # 3. Measure saliency (opaque – runs one fwd+bwd pass, no pruning)
        saliency = measure_multi_saliency(model, criterion, get_batch)
        print(
            f"  [saliency]  mag_avg={saliency.mag_avg:.5f}"
            f"  grad_avg={saliency.grad_avg:.7f}"
            f"  taylor_avg={saliency.taylor_avg:.7f}"
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    remaining = get_model_sparsity(model)
    model.save(
        f"lenet300_mnist_nplh_remaining{remaining:.1f}_acc{acc:.1f}",
        PRUNED_MODELS_PATH,
    )
    print(f"\nFinished.  remaining={remaining:.1f}%  acc={acc:.2f}%")
    return model
