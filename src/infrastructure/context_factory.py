from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from src.infrastructure.layers import ModelCustom, get_flow_params_loss, set_mask_apply_all, set_mask_training_all, \
    set_weights_training_all
from src.infrastructure.schedulers import AbstractScheduler
from src.infrastructure.training_context import TrainingContext
from src.infrastructure.dataset_context.dataset_context import DatasetContextAbstract

def _train_one_epoch_hyperflux(model: ModelCustom, dataset: DatasetContextAbstract, optimizer_weights: optim.Optimizer, optimizer_masks: optim.Optimizer, criterion:nn.Module, scheduler: AbstractScheduler) -> None:
    """
    Trains the model normally, but with mask parameters included in the optimization. Basically hyperflux. Also takes scheduler into account for loss adjustment
    """
    set_mask_apply_all(model, True)
    set_mask_training_all(model, True)
    set_weights_training_all(model, True)

    model.train()
    dataset.init_data_split()

    while dataset.any_data_training_available():
        data, target = dataset.get_training_data_and_labels()

        optimizer_weights.zero_grad()
        optimizer_masks.zero_grad()

        output = model(data)
        loss_masks = model.get_hyperflux_loss() * scheduler.get_multiplier()
        loss_data = criterion(output, target)
        loss = loss_masks + loss_data

        loss.backward()

        optimizer_weights.step()
        optimizer_masks.step()


def _train_one_epoch(model: ModelCustom, dataset: DatasetContextAbstract,
                     optimizer: optim.Optimizer, criterion: nn.Module) -> None:
    """
    Normal training, nothing fancy, masks are just applied
    """
    set_mask_apply_all(model, True)
    set_mask_training_all(model, False)
    set_weights_training_all(model, True)

    model.train()
    dataset.init_data_split()
    while dataset.any_data_training_available():
        data, target = dataset.get_training_data_and_labels()

        optimizer.zero_grad()

        loss = criterion(model(data), target)

        loss.backward()
        optimizer.step()


def _evaluate(model: ModelCustom, dataset: DatasetContextAbstract, criterion: nn.Module) -> tuple[float, float]:
    """
    Evaluates the model and computes both accuracy and loss.
    Returns:
        tuple[float, float]: (accuracy, loss)
    """
    set_mask_apply_all(model, True)
    set_mask_training_all(model, False)
    set_weights_training_all(model, True)

    model.eval()
    dataset.init_data_split()
    correct = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        while dataset.any_data_testing_available():
            data, target = dataset.get_testing_data_and_labels()
            out = model(data)

            # Compute accuracy
            correct += out.argmax(dim=1).eq(target).sum().item()

            # Compute loss
            total_loss += criterion(out, target).item()
            n_batches += 1

    accuracy = 100.0 * correct / dataset.get_data_testing_length()
    avg_loss = total_loss / max(n_batches, 1)

    return accuracy, avg_loss

def _accumulate_gradients(model: ModelCustom, dataset: DatasetContextAbstract,
                          optimizer: optim.Optimizer, criterion: nn.Module,
                          n_batches: int | None) -> None:
    """
    averages of absolute values of gradients over n batches, masks are just applied not trained, only does this for weights
    """
    set_mask_apply_all(model, True)
    set_mask_training_all(model, False)
    set_weights_training_all(model, True)

    model.train()
    dataset.init_data_split()
    absolute_accumulation: dict[int, torch.Tensor] = {}
    squared_accumulation:  dict[int, torch.Tensor] = {}
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
                g = param.grad.detach()
                if pid not in absolute_accumulation:
                    absolute_accumulation[pid] = g.abs().clone()
                    squared_accumulation[pid]  = g.pow(2).clone()
                else:
                    absolute_accumulation[pid] += g.abs()
                    squared_accumulation[pid]  += g.pow(2)
        count += 1

    # Write mean |grad| into param.grad and mean g² into param._hessian_diag
    if count > 0:
        for param in model.parameters():
            pid = id(param)
            if pid in absolute_accumulation:
                param.grad = absolute_accumulation[pid] / count
                param._hessian_diag = squared_accumulation[pid] / count


def _accumulate_mask_gradients(model: ModelCustom, dataset: DatasetContextAbstract,
                               optimizer_weights: optim.Optimizer, criterion: nn.Module,
                               n_batches: int | None) -> None:
    """
    Averages of mask gradients over n batches. Weights are fixed, masks are trainable.
    Gradients are NOT absolute-valued — we want the signed flux.
    """
    set_mask_apply_all(model, True)
    set_mask_training_all(model, True)
    set_weights_training_all(model, False)

    model.train()
    dataset.init_data_split()

    accumulation: dict[int, torch.Tensor] = {}
    count = 0

    while dataset.any_data_training_available():
        if n_batches is not None and count >= n_batches:
            break

        data, target = dataset.get_training_data_and_labels()
        optimizer_weights.zero_grad() # This clears weight grads, but we need to clear mask grads too.
        # However, masks might not be in optimizer_weights.
        # Let's manually zero all parameters just in case.
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        loss = criterion(model(data), target)
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                pid = id(param)
                g = param.grad.detach()
                if pid not in accumulation:
                    accumulation[pid] = g.clone()
                else:
                    accumulation[pid] += g
        count += 1

    if count > 0:
        for param in model.parameters():
            pid = id(param)
            if pid in accumulation:
                param.grad = accumulation[pid] / count


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
    model: ModelCustom,
    dataset: DatasetContextAbstract,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    gradient_batches: int | None = None,
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
                          typically sufficient for Taylor/gradient/Hessian saliency.
    """
    ctx = TrainingContext(
        model=model,
        optimizer=optimizer,
        train_one_epoch=lambda: None,
        train_one_epoch_hyperflux=lambda sched, optim: None,
        evaluate=lambda: _evaluate(model, dataset, criterion),
        accumulate_gradients=lambda: _accumulate_gradients(
            model, dataset, optimizer, criterion, gradient_batches
        ),
        accumulate_mask_gradients=lambda: _accumulate_mask_gradients(
            model, dataset, optimizer, criterion, gradient_batches
        ),
        reset_optimizer_state=lambda: _reset_optimizer_state(model, optimizer),
    )

    def _counting_train() -> None:
        _train_one_epoch(model, dataset, optimizer, criterion)
        ctx.epoch_count += 1

    def _counting_train_hyperflux(scheduler: AbstractScheduler, optimizer_masks: optim.Optimizer) -> None:
        _train_one_epoch_hyperflux(
            model=model,
            dataset=dataset,
            optimizer_weights=optimizer,
            optimizer_masks=optimizer_masks,
            criterion=criterion,
            scheduler=scheduler
        )
        ctx.epoch_count += 1

    ctx.train_one_epoch = _counting_train
    ctx.train_one_epoch_hyperflux = _counting_train_hyperflux

    return ctx
