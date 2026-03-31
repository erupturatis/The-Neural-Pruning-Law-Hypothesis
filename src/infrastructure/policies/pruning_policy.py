from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.optim as optim

from src.experiments.utils import get_model_density
from src.infrastructure.training_context import TrainingContext
from src.infrastructure.constants import WEIGHTS_ATTR, MASK_ATTR
from src.infrastructure.layers import get_layers_primitive, set_mask_apply_all, set_mask_training_all, set_weights_training_all
from src.infrastructure.schedulers import AbstractScheduler
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
    def __init__(self, pruning_rate: float):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        print(f"Applying MagnitudePruningPolicy with pruning_rate={self.pruning_rate:.2f}%")
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
        prune_count = int(all_magnitudes.numel() * self.pruning_rate / 100)

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
    def __init__(self, pruning_rate: float):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        if self.pruning_rate == 0.0:
            warn("Pruning rate is 0.0, skipping pruning step.")
            return

        with torch.no_grad():
            for layer in get_layers_primitive(ctx.model):
                mask = getattr(layer, MASK_ATTR).data
                flat = mask.flatten()
                active_idx = (flat >= 0).nonzero(as_tuple=False).squeeze(1)
                if active_idx.numel() == 0:
                    continue

                to_prune_count = int(active_idx.numel() * self.pruning_rate / 100)
                if to_prune_count == 0:
                    warn(f"Pruning rate too low for layer {layer}, skipping pruning for this layer.")
                    continue
                if to_prune_count >= active_idx.numel():
                    raise Exception(f"Pruning rate too high for layer {layer}, would prune all weights, something went wrong")

                perm = torch.randperm(active_idx.numel(), device=mask.device)[:to_prune_count]
                flat[active_idx[perm]] = -1.0


def _gradient_based_prune(ctx: TrainingContext, pruning_rate: float, score_fn) -> None:

    """
    Shared implementation for gradient- and taylor-based pruning.
    score_fn(w) -> Tensor: computes per-weight score from a weight parameter (w.grad is available).
    Saves and restores param.grad so training momentum is undisturbed.
    """
    saved_grads = _save_grads(ctx.model)
    ctx.accumulate_gradients()

    layers = get_layers_primitive(ctx.model)
    layer_data = []  # (layer, active_mask, scores_tensor)
    for layer in layers:
        w    = getattr(layer, WEIGHTS_ATTR)
        mask = getattr(layer, MASK_ATTR).data
        if w.grad is None:
            raise Exception(f"Layer {layer} has no gradients after accumulation.")
        active = mask >= 0
        if not active.any():
            raise Exception(f"Layer {layer} has no active weights.")
        layer_data.append((layer, active, score_fn(w)))

    if not layer_data:
        raise Exception("No layers found.")

    all_scores = torch.cat([scores[active].abs().flatten() for _, active, scores in layer_data])
    to_prune_count = int(all_scores.numel() * pruning_rate / 100)

    if to_prune_count == 0:
        warn("Pruning rate too low, no weights will be pruned.")
        _restore_grads(ctx.model, saved_grads)
        return
    if to_prune_count >= all_scores.numel():
        raise Exception("Pruning rate too high, would prune all weights.")

    threshold = torch.kthvalue(all_scores, to_prune_count).values.item()

    with torch.no_grad():
        for layer, active, scores in layer_data:
            getattr(layer, MASK_ATTR).data[active & (scores.abs() <= threshold)] = -1.0

    _restore_grads(ctx.model, saved_grads)


class GradientPruningPolicy(PruningPolicy):
    # uses: ctx.model, ctx.accumulate_gradients
    def __init__(self, pruning_rate: float = 10):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        _gradient_based_prune(ctx, self.pruning_rate, lambda w: w.grad.detach())


class TaylorPruningPolicy(PruningPolicy):
    # uses: ctx.model, ctx.accumulate_gradients
    def __init__(self, pruning_rate: float = 10):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        _gradient_based_prune(ctx, self.pruning_rate, lambda w: w.data * w.grad.detach())


class HessianPruningPolicy(PruningPolicy):
    # Diagonal-Fisher approximation: saliency(w_i) = ½ · mean(g_i²) · w_i²
    # mean(g²) is written to param._hessian_diag by accumulate_gradients (same pass, no extra cost).
    # uses: ctx.model, ctx.accumulate_gradients
    def __init__(self, pruning_rate: float = 10):
        self.pruning_rate = pruning_rate

    def apply_pruning(self, ctx: TrainingContext) -> None:
        def hessian_score(w):
            if not hasattr(w, '_hessian_diag'):
                raise Exception("HessianPruning: no _hessian_diag on weight — accumulate_gradients must run first")
            return 0.5 * w._hessian_diag * w.data.pow(2)

        _gradient_based_prune(ctx, self.pruning_rate, hessian_score)


class HyperfluxPruningPolicy(PruningPolicy):
    # uses: ctx.train_one_epoch_hyperflux
    # optimizer_masks owns the mask Adam state — it is never touched by normal
    # training or saliency measurement, so hyperflux state resumes seamlessly
    # across convergence training and saliency cycles.
    def __init__(
        self,
        scheduler: AbstractScheduler,
        optimizer_masks: optim.Optimizer,
        pruning_rate: float = 10,
        max_epochs_per_cycle: int = 1000,
    ):
        self.scheduler = scheduler
        self.optimizer_masks = optimizer_masks
        self.pruning_rate = pruning_rate
        self.max_epochs_per_cycle = max_epochs_per_cycle
        self._epoch_count: int = 0        # monotonically increasing across all cycles
        self._target_density: float = 100.0

    def apply_pruning(self, ctx: TrainingContext) -> None:
        self._target_density *= (1.0 - self.pruning_rate / 100)
        print(f"[HyperfluxPruning] target density this cycle: {self._target_density:.4f}%")

        for _ in range(self.max_epochs_per_cycle):
            ctx.train_one_epoch_hyperflux(self.scheduler, self.optimizer_masks)
            self._epoch_count += 1
            density = get_model_density(ctx.model)
            self.scheduler.step(self._epoch_count, density)
            print(f"  epoch={self._epoch_count}  density={density:.4f}%  target={self._target_density:.4f}%  pressure={self.scheduler.get_multiplier():.4f}")
            if density <= self._target_density:
                break
        else:
            warn(f"HyperfluxPruning: reached max_epochs_per_cycle={self.max_epochs_per_cycle} "
                 f"without hitting target density {self._target_density:.4f}% "
                 f"(current: {get_model_density(ctx.model):.4f}%). "
                 f"Consider increasing max_epochs_per_cycle or scheduler pressure.")



