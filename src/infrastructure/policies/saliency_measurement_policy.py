from __future__ import annotations
from abc import ABC, abstractmethod

import torch

from src.infrastructure.training_context import TrainingContext
from src.infrastructure.constants import WEIGHTS_ATTR, MASK_ATTR
from src.infrastructure.policies.pruning_policy import _save_grads, _restore_grads


class SaliencyMeasurementPolicy(ABC):
    @abstractmethod
    def measure_saliency(self, ctx: TrainingContext) -> tuple[float, float]:
        """Returns (min_saliency, avg_saliency) over all active weights/units."""
        pass


class MagnitudeSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    # uses: ctx.model
    def measure_saliency(self, ctx: TrainingContext) -> tuple[float, float]:
        active_mags = []
        for layer in ctx.model.get_layers_primitive():
            w = getattr(layer, WEIGHTS_ATTR).data
            mask = getattr(layer, MASK_ATTR).data
            active = mask >= 0
            if active.any():
                active_mags.append(w[active].abs().flatten())

        if not active_mags:
            return 0.0, 0.0

        all_mags = torch.cat(active_mags)
        min_sal = float(all_mags.min().item())
        avg_sal = float(all_mags.mean().item())
        print(f"[MagnitudeSaliency] min |w| = {min_sal:.6e}  avg |w| = {avg_sal:.6e}")
        return min_sal, avg_sal


class TaylorSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    # uses: ctx.model, ctx.accumulate_gradients
    # Saves and restores param.grad so training momentum is undisturbed.
    def measure_saliency(self, ctx: TrainingContext) -> tuple[float, float]:
        saved_grads = _save_grads(ctx.model)
        ctx.accumulate_gradients()

        active_scores = []
        for layer in ctx.model.get_layers_primitive():
            w = getattr(layer, WEIGHTS_ATTR)
            mask = getattr(layer, MASK_ATTR).data
            if w.grad is None:
                continue
            active = mask >= 0
            if active.any():
                taylor = (w.data * w.grad.detach())[active].abs().flatten()
                active_scores.append(taylor)

        _restore_grads(ctx.model, saved_grads)
        ctx.model.train()

        if not active_scores:
            return 0.0, 0.0

        all_scores = torch.cat(active_scores)
        min_sal = float(all_scores.min().item())
        avg_sal = float(all_scores.mean().item())
        print(f"[TaylorSaliency] min |w·g| = {min_sal:.6e}  avg |w·g| = {avg_sal:.6e}")
        return min_sal, avg_sal


class HessianSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    # uses: ctx.model, ctx.compute_hessian_diagonal
    def measure_saliency(self, ctx: TrainingContext) -> tuple[float, float]:
        return 0.0, 0.0


class APoZSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    # uses: ctx.model, ctx.run_forward
    # Registers forward hooks on all primitive layers to collect output
    # activations, then computes APoZ (fraction of outputs <= 0, i.e. dead
    # after ReLU) per active neuron, averaged across all active neurons.
    def measure_saliency(self, ctx: TrainingContext) -> tuple[float, float]:
        layers = ctx.model.get_layers_primitive()

        # Per-layer accumulators: zeros count and sample count per output unit
        zeros_per_layer = [None] * len(layers)
        counts_per_layer = [0] * len(layers)

        def _make_hook(idx):
            def hook(module, input, output):
                # output: [batch, out_features] for FC or [batch, C, H, W] for Conv
                # Flatten spatial dims so the channel/feature dim is always dim 1
                out = output.detach()
                if out.dim() > 2:
                    # [batch, C, H, W] → count zeros per channel across batch and spatial
                    dead = (out <= 0).float().sum(dim=(0, 2, 3))   # [C]
                    n_samples = out.shape[0] * out.shape[2] * out.shape[3]
                else:
                    dead = (out <= 0).float().sum(dim=0)            # [out_features]
                    n_samples = out.shape[0]

                if zeros_per_layer[idx] is None:
                    zeros_per_layer[idx] = dead.cpu()
                else:
                    zeros_per_layer[idx] += dead.cpu()
                counts_per_layer[idx] += n_samples
            return hook

        handles = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]
        ctx.run_forward()
        for h in handles:
            h.remove()
        ctx.model.train()

        apoz_vals = []
        for i, layer in enumerate(layers):
            if zeros_per_layer[i] is None or counts_per_layer[i] == 0:
                continue
            mask = getattr(layer, MASK_ATTR).data
            # Active output units: at least one unpruned incoming weight
            dims = tuple(range(1, mask.dim()))
            active_units = (mask >= 0).any(dim=dims).cpu()  # [out_features] or [C]
            if not active_units.any():
                continue
            apoz_per_unit = zeros_per_layer[i] / counts_per_layer[i]  # [units]
            apoz_vals.append(apoz_per_unit[active_units])

        if not apoz_vals:
            return 0.0, 0.0

        all_vals = torch.cat(apoz_vals)
        min_sal = float(all_vals.min().item())
        avg_sal = float(all_vals.mean().item())
        print(f"[APoZSaliency] min APoZ = {min_sal:.4f}  avg APoZ = {avg_sal:.4f}")
        return min_sal, avg_sal


class HyperfluxSampleEstimationSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    # uses: ctx.model, ctx.run_forward
    def measure_saliency(self, ctx: TrainingContext) -> tuple[float, float]:
        return 0.0, 0.0
