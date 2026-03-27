from __future__ import annotations
from abc import ABC, abstractmethod

import torch

from src.infrastructure.training_context import TrainingContext
from src.infrastructure.constants import WEIGHTS_ATTR, MASK_ATTR
from src.infrastructure.layers import get_layers_primitive
from warnings import warn


def _save_grads(model) -> dict:
    return {
        name: (param.grad.clone() if param.grad is not None else None)
        for name, param in model.named_parameters()
    }


def _restore_grads(model, saved: dict) -> None:
    for name, param in model.named_parameters():
        param.grad = saved.get(name)


class PruningPolicy(ABC):
    @abstractmethod
    def apply_pruning(self, ctx: TrainingContext) -> None:
        pass


class MagnitudePruningPolicy(PruningPolicy):
    # uses: ctx.model
    def __init__(self, pruning_rate: float = 0.1):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        if self.pruning_rate == 0.0:
            warn("Pruning rate is 0.0, skipping pruning step.")
            return

        active_magnitudes = []
        for layer in get_layers_primitive(ctx.model):
            weights = getattr(layer, WEIGHTS_ATTR).data
            mask = getattr(layer, MASK_ATTR).data
            active = weights[mask >= 0]
            if active.numel() > 0:
                # flatten redundant
                active_magnitudes.append(torch.abs(active).flatten())

        if len(active_magnitudes) == 0:
            raise Exception("No layers found, active magnitudes is empty")
        if any(layer_tensor.numel() == 0 for layer_tensor in active_magnitudes):
            raise Exception("One or more layers have 0 active weights.")

        all_magnitudes = torch.cat(active_magnitudes)
        prune_count = int(all_magnitudes.numel() * self.pruning_rate)

        if prune_count == 0:
            raise Exception("Pruning rate too low or no weights to prune")
        if prune_count >= all_magnitudes.numel():
            raise Exception("Pruning rate too high, would prune all weights, something went wrong")

        threshold = torch.kthvalue(all_magnitudes, prune_count).values.item()

        with torch.no_grad():
            for layer in get_layers_primitive(ctx.model):
                weights = getattr(layer, WEIGHTS_ATTR).data
                mask = getattr(layer, MASK_ATTR).data
                mask[(mask >= 0) & (torch.abs(weights) <= threshold)] = -1.0


class RandomPruningPolicy(PruningPolicy):
    # uses: ctx.model
    def __init__(self, pruning_rate: float = 0.1):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        with torch.no_grad():
            for layer in get_layers_primitive(ctx.model):
                mask = getattr(layer, MASK_ATTR).data
                flat = mask.flatten()
                active_idx = (flat >= 0).nonzero(as_tuple=False).squeeze(1)
                if active_idx.numel() == 0:
                    continue
                n_prune = max(1, int(active_idx.numel() * self.pruning_rate))
                perm = torch.randperm(active_idx.numel(), device=mask.device)[:n_prune]
                flat[active_idx[perm]] = -1.0


class GradientPruningPolicy(PruningPolicy):
    # uses: ctx.model, ctx.accumulate_gradients
    # Saves and restores param.grad so training momentum is undisturbed.
    def __init__(self, pruning_rate: float = 0.1):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        saved_grads = _save_grads(ctx.model)
        ctx.accumulate_gradients()

        layers = get_layers_primitive(ctx.model)
        active_scores = []
        for layer in layers:
            w = getattr(layer, WEIGHTS_ATTR)
            mask = getattr(layer, MASK_ATTR).data
            if w.grad is None:
                continue
            active = mask >= 0
            if active.any():
                active_scores.append(w.grad.detach()[active].abs().flatten())

        if active_scores:
            all_scores = torch.cat(active_scores)
            k = min(max(1, int(all_scores.numel() * self.pruning_rate)), all_scores.numel())
            threshold = torch.kthvalue(all_scores, k).values.item()

            with torch.no_grad():
                for layer in layers:
                    w = getattr(layer, WEIGHTS_ATTR)
                    mask = getattr(layer, MASK_ATTR).data
                    if w.grad is None:
                        continue
                    active = mask >= 0
                    mask[active & (w.grad.detach().abs() <= threshold)] = -1.0

        _restore_grads(ctx.model, saved_grads)


class TaylorPruningPolicy(PruningPolicy):
    # uses: ctx.model, ctx.accumulate_gradients
    # Saves and restores param.grad so training momentum is undisturbed.
    def __init__(self, pruning_rate: float = 0.1):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        saved_grads = _save_grads(ctx.model)
        ctx.accumulate_gradients()

        layers = get_layers_primitive(ctx.model)
        active_scores = []
        for layer in layers:
            w = getattr(layer, WEIGHTS_ATTR)
            mask = getattr(layer, MASK_ATTR).data
            if w.grad is None:
                continue
            active = mask >= 0
            if active.any():
                taylor = (w.data * w.grad.detach())[active].abs().flatten()
                active_scores.append(taylor)

        if active_scores:
            all_scores = torch.cat(active_scores)
            k = min(max(1, int(all_scores.numel() * self.pruning_rate)), all_scores.numel())
            threshold = torch.kthvalue(all_scores, k).values.item()

            with torch.no_grad():
                for layer in layers:
                    w = getattr(layer, WEIGHTS_ATTR)
                    mask = getattr(layer, MASK_ATTR).data
                    if w.grad is None:
                        continue
                    active = mask >= 0
                    taylor = (w.data * w.grad.detach()).abs()
                    mask[active & (taylor <= threshold)] = -1.0

        _restore_grads(ctx.model, saved_grads)


class HessianPruningPolicy(PruningPolicy):
    # uses: ctx.model, ctx.optimizer, ctx.compute_hessian_diagonal, ctx.reset_optimizer_state
    def __init__(self, pruning_rate: float = 0.1):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        pass


class HyperfluxPruningPolicy(PruningPolicy):
    # uses: ctx.model, ctx.optimizer, ctx.train_one_epoch, ctx.reset_optimizer_state
    def __init__(self, pruning_rate: float = 0.1):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        pass
