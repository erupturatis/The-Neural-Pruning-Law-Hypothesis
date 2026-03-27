from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from src.infrastructure.training_context import TrainingContext
from src.infrastructure.dataset_context.dataset_context import DatasetContextAbstract

# ── Private primitives ────────────────────────────────────────────────────────
# These are generic implementations that work for any nn.Module and any dataset
# that implements DatasetContextAbstract. Multi-GPU (DataParallel) is transparent
# since model(x) and param.data/.grad work identically on wrapped models.

def _train_one_epoch(model: nn.Module, dataset: DatasetContextAbstract,
                     optimizer: optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    dataset.init_data_split()
    total_loss, n_batches = 0.0, 0
    while dataset.any_data_training_available():
        data, target = dataset.get_training_data_and_labels()
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _evaluate(model: nn.Module, dataset: DatasetContextAbstract) -> float:
    # Calls init_data_split() itself so it is fully self-contained and can be
    # called independently from train_one_epoch without relying on a prior call
    # having initialised the test iterator.
    model.eval()
    dataset.init_data_split()
    correct = 0
    with torch.no_grad():
        while dataset.any_data_testing_available():
            data, target = dataset.get_testing_data_and_labels()
            out = model(data)
            correct += out.argmax(dim=1).eq(target).sum().item()
    return 100.0 * correct / dataset.get_data_testing_length()


def _accumulate_gradients(model: nn.Module, dataset: DatasetContextAbstract,
                           optimizer: optim.Optimizer, criterion: nn.Module,
                           n_batches: int | None) -> None:
    # Fills param.grad with the mean ABSOLUTE gradient over `n_batches` batches
    # (or the full epoch if n_batches is None). Does NOT call optimizer.step().
    # Absolute values are accumulated per-batch before summing so that gradients
    # of opposite sign across batches do not cancel out.
    model.train()
    dataset.init_data_split()
    abs_accum: dict[int, torch.Tensor] = {}
    count = 0
    while dataset.any_data_training_available():
        if n_batches is not None and count >= n_batches:
            break
        data, target = dataset.get_training_data_and_labels()
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                pid = id(param)
                if pid not in abs_accum:
                    abs_accum[pid] = param.grad.detach().abs().clone()
                else:
                    abs_accum[pid] += param.grad.detach().abs()
        count += 1
    # Write mean |grad| back into param.grad
    if count > 0:
        for param in model.parameters():
            pid = id(param)
            if pid in abs_accum:
                param.grad = abs_accum[pid] / count


def _compute_hessian_diagonal(model: nn.Module, dataset: DatasetContextAbstract,
                               criterion: nn.Module,
                               n_batches: int | None) -> dict[str, Tensor]:
    # Hutchinson's estimator: one Rademacher sample per batch, averaged across
    # batches. Holds the forward-pass compute graph in memory during each batch,
    # so keep n_batches small for large models to avoid OOM.
    model.train()
    dataset.init_data_split()

    param_dict = dict(model.named_parameters())
    hessian_diag = {name: torch.zeros_like(p) for name, p in param_dict.items()}
    param_list = list(param_dict.values())

    count = 0
    while dataset.any_data_training_available():
        if n_batches is not None and count >= n_batches:
            break
        data, target = dataset.get_training_data_and_labels()

        model.zero_grad()
        loss = criterion(model(data), target)

        # First-order gradients — keep graph for second-order pass
        grads = torch.autograd.grad(loss, param_list, create_graph=True)

        # Rademacher vector z ∈ {-1, +1}^n
        zs = [torch.randint_like(g, 0, 2).to(g.dtype) * 2 - 1 for g in grads]

        # Hessian-vector product Hv = ∂(g·z)/∂w
        gz = sum((g * z).sum() for g, z in zip(grads, zs))
        hvs = torch.autograd.grad(gz, param_list)

        # Accumulate diagonal estimate: H_ii ≈ z_i * (Hv)_i
        for name, z, hv in zip(param_dict.keys(), zs, hvs):
            hessian_diag[name] += z * hv.detach()

        count += 1

    if count > 0:
        for name in hessian_diag:
            hessian_diag[name] /= count

    return hessian_diag


def _run_forward(model: nn.Module, dataset: DatasetContextAbstract) -> None:
    # Runs the full training set through the model under no_grad.
    # Policies register their hooks on model before calling this; the forward
    # pass fires those hooks automatically.
    model.eval()
    dataset.init_data_split()
    with torch.no_grad():
        while dataset.any_data_training_available():
            data, _ = dataset.get_training_data_and_labels()
            model(data)


def _reset_optimizer_state(model: nn.Module, optimizer: optim.Optimizer) -> None:
    # After pruning, zero all optimizer state tensors (momentum buffers, Adam
    # moments, etc.) for weights that are currently masked out (value == 0).
    # Works generically for SGD+momentum, Adam, AdamW, and any optimizer that
    # stores per-parameter state tensors with the same shape as the parameter.
    for group in optimizer.param_groups:
        for param in group['params']:
            if param not in optimizer.state:
                continue
            mask = param.data == 0
            if not mask.any():
                continue
            for val in optimizer.state[param].values():
                if isinstance(val, Tensor) and val.shape == param.shape:
                    val[mask] = 0.0


# ── Factory ───────────────────────────────────────────────────────────────────

def make_training_context(
    model: nn.Module,
    dataset: DatasetContextAbstract,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    gradient_batches: int | None = None,
    hessian_batches: int | None = None,
) -> TrainingContext:
    """
    Builds a fully wired TrainingContext from the given objects.

    All primitives are network- and dataset-agnostic: they work for any
    nn.Module and any DatasetContextAbstract implementation (MNIST, CIFAR,
    ImageNet, ...). DataParallel is transparent. For DistributedDataParallel,
    add a DistributedSampler inside the dataset class — the primitives here
    are unaffected.

    Args:
        gradient_batches: batches to use for accumulate_gradients.
                          None = full epoch. A small value (e.g. 16) is
                          typically sufficient for Taylor/gradient saliency.
        hessian_batches:  batches to use for compute_hessian_diagonal.
                          Hessian estimation holds the compute graph per batch;
                          keep this small for large models.
    """
    ctx = TrainingContext(
        model=model,
        optimizer=optimizer,
        train_one_epoch=lambda: None,  # replaced below after ctx is created
        evaluate=lambda: _evaluate(model, dataset),
        accumulate_gradients=lambda: _accumulate_gradients(
            model, dataset, optimizer, criterion, gradient_batches
        ),
        compute_hessian_diagonal=lambda: _compute_hessian_diagonal(
            model, dataset, criterion, hessian_batches
        ),
        run_forward=lambda: _run_forward(model, dataset),
        reset_optimizer_state=lambda: _reset_optimizer_state(model, optimizer),
    )

    def _counting_train() -> float:
        loss = _train_one_epoch(model, dataset, optimizer, criterion)
        ctx.epoch_count += 1
        return loss

    ctx.train_one_epoch = _counting_train
    return ctx
