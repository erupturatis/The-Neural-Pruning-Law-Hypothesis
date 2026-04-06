from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from src.infrastructure.training_context import TrainingContext
from src.infrastructure.constants import WEIGHTS_ATTR, MASK_ATTR, GRADIENT_IDENTITY_SCALER
from src.infrastructure.layers import (
    get_layers_primitive,
    get_prunable_layers,
    set_mask_apply_all,
    set_mask_training_all,
    set_weights_training_all,
)
from src.infrastructure.policies.pruning_policy import _save_grads, _restore_grads


@dataclass
class SaliencyResult:
    avg_saliency:               float
    avg_saliency_contributing:  float
    min_saliency:               float
    min_saliency_contributing:  float

from dataclasses import dataclass
import torch

@dataclass
class NetworkState:
    """
    Snapshot computed once per pruning round, before the saliency loop.

    Weight-level:
      present_masks / active_masks — bool tensors per layer, same shape as weights.
        present = mask >= 0
        active  = present AND |grad| > 0
      total_count / present_count / active_count — scalars for percentage printing.

    Neuron-level:
      active_neurons  — bool [n_out] per layer; True if the neuron ever fired (output > 0)
                        across all samples in the gradient-accumulation pass.
      activation_freq — float [n_out] per layer; fraction of samples where output > 0.

    Cached gradients (set by compute_network_state, reused by saliency policies):
      weight_grads   — per-layer clone of param.grad from the accumulation pass.
      hessian_diags  — per-layer clone of param._hessian_diag from the same pass.
      Both are None if the layer had no gradient / no hessian diagonal.
      Gradient-based saliency policies read these instead of re-running
      accumulate_gradients(), saving 3 full forward+backward passes per round.
    """
    total_count:     int
    present_count:   int
    active_count:    int
    present_masks:   list[torch.Tensor]
    active_masks:    list[torch.Tensor]
    active_neurons:  list[torch.Tensor]
    activation_freq: list[torch.Tensor]
    weight_grads:    list[torch.Tensor | None]
    hessian_diags:   list[torch.Tensor | None]


def compute_network_state(ctx: TrainingContext) -> NetworkState:
    """
    Single pass (gradient accumulation) to compute all weight- and neuron-level state.
    Forward hooks are registered before the pass to collect per-neuron activation
    frequencies without an extra forward-only pass.
    Saves and restores param.grad so downstream training is undisturbed.
    """
    layers = get_prunable_layers(ctx.model)

    fire_count    = [None] * len(layers)
    sample_counts = [0]    * len(layers)

    def _make_hook(idx):
        def hook(module, input, output):
            out = output.detach()
            # Reduce all spatial/extra dims; keep only the output-unit dim (dim 1 for conv, dim 1 for fc).
            spatial = tuple(range(2, out.dim()))
            if spatial:
                fired = (out > 0).any(dim=spatial).float().sum(dim=0).cpu()  # [C]
            else:
                fired = (out > 0).float().sum(dim=0).cpu()                   # [out]
            fire_count[idx] = fired if fire_count[idx] is None else fire_count[idx] + fired
            sample_counts[idx] += out.shape[0]
        return hook

    handles = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]
    saved_grads = _save_grads(ctx.model)
    ctx.accumulate_gradients()
    for h in handles:
        h.remove()

    present_masks:   list[torch.Tensor] = []
    active_masks:    list[torch.Tensor] = []
    active_neurons:  list[torch.Tensor] = []
    activation_freq: list[torch.Tensor] = []
    weight_grads:    list[torch.Tensor | None] = []
    hessian_diags:   list[torch.Tensor | None] = []
    total_count = present_count = active_count = 0

    for i, layer in enumerate(layers):
        w    = getattr(layer, WEIGHTS_ATTR)
        mask = getattr(layer, MASK_ATTR).data
        if w.grad is None:
            raise Exception(f"compute_network_state: layer {layer} has no gradient after accumulate_gradients")

        present = mask >= 0
        active  = present & (w.grad.detach().abs() > 0)
        present_masks.append(present)
        active_masks.append(active)
        total_count   += mask.numel()
        present_count += int(present.sum().item())
        active_count  += int(active.sum().item())

        n_out = present.shape[0]

        freq = (fire_count[i] / sample_counts[i]) if (fire_count[i] is not None and sample_counts[i] > 0) else torch.zeros(n_out)
        activation_freq.append(freq)
        active_neurons.append(freq > 0)

        # Clone gradients before _restore_grads wipes them so saliency policies
        # can reuse them without a second accumulation pass.
        weight_grads.append(w.grad.detach().clone())
        hessian_diags.append(
            w._hessian_diag.detach().clone() if hasattr(w, '_hessian_diag') else None
        )

    _restore_grads(ctx.model, saved_grads)
    ctx.model.train()

    return NetworkState(
        total_count=total_count,
        present_count=present_count,
        active_count=active_count,
        present_masks=present_masks,
        active_masks=active_masks,
        active_neurons=active_neurons,
        activation_freq=activation_freq,
        weight_grads=weight_grads,
        hessian_diags=hessian_diags,
    )


class SaliencyMeasurementPolicy(ABC):
    @abstractmethod
    def measure_saliency(self, ctx: TrainingContext, state: NetworkState) -> SaliencyResult:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats(t: torch.Tensor) -> tuple[float, float]:
    """Returns (avg, min) of a non-empty 1-D tensor."""
    return float(t.mean().item()), float(t.min().item())


def _collect(layers, masks, transform=None) -> torch.Tensor:
    """
    Gather per-layer values selected by bool masks. If transform is provided it
    is called as transform(layer, i) -> Tensor; otherwise layer weights are used.
    Returns a flat abs-valued concatenated tensor, or empty tensor if nothing selected.
    """
    parts = []
    for i, layer in enumerate(layers):
        if not masks[i].any():
            continue
        t = getattr(layer, WEIGHTS_ATTR).data if transform is None else transform(layer, i)
        parts.append(t[masks[i]].abs().flatten())
    return torch.cat(parts) if parts else torch.tensor([])


def _print_saliency(name: str, avg_present: float, avg_active: float, state: NetworkState) -> None:
    pct_p = 100.0 * state.present_count / state.total_count
    pct_a = 100.0 * state.active_count  / state.total_count
    print(f"[{name}] avg = {avg_present:.6e} ({pct_p:.2f}% present)  "
          f"contributing avg = {avg_active:.6e} ({pct_a:.2f}% active)")


# ── Weight-based policies ─────────────────────────────────────────────────────

class MagnitudeSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    def measure_saliency(self, ctx: TrainingContext, state: NetworkState) -> SaliencyResult:
        layers = get_prunable_layers(ctx.model)
        present_mag = _collect(layers, state.present_masks)
        active_mag  = _collect(layers, state.active_masks)

        if present_mag.numel() == 0:
            raise Exception("MagnitudeSaliency: no present weights found")
        if active_mag.numel() == 0:
            warnings.warn("MagnitudeSaliency: no active weights found; all present weights have zero gradient")

        avg_sal, min_sal   = _stats(present_mag)
        avg_live, min_live = _stats(active_mag) if active_mag.numel() > 0 else (0.0, 0.0)
        _print_saliency("MagnitudeSaliency", avg_sal, avg_live, state)
        return SaliencyResult(avg_sal, avg_live, min_sal, min_live)


def _gradient_based_saliency(
    ctx: TrainingContext, state: NetworkState, score_fn, name: str
) -> SaliencyResult:
    """
    Shared skeleton for gradient-pass saliency policies.
    score_fn(layer, i) -> Tensor: per-weight scores; param.grad is available.

    If state.weight_grads is populated (always the case when compute_network_state
    was used), gradients are written directly onto param.grad from the cache and no
    new accumulation pass is run.  Otherwise falls back to ctx.accumulate_gradients().
    """
    saved_grads = _save_grads(ctx.model)
    layers = get_prunable_layers(ctx.model)

    if state.weight_grads is not None:
        # Reuse gradients cached by compute_network_state — no extra GPU pass needed.
        for i, layer in enumerate(layers):
            w = getattr(layer, WEIGHTS_ATTR)
            w.grad = state.weight_grads[i].clone()
            if state.hessian_diags[i] is not None:
                w._hessian_diag = state.hessian_diags[i].clone()
    else:
        ctx.accumulate_gradients()

    present_scores = _collect(layers, state.present_masks, transform=score_fn)
    active_scores  = _collect(layers, state.active_masks,  transform=score_fn)
    _restore_grads(ctx.model, saved_grads)
    ctx.model.train()
    if present_scores.numel() == 0:
        raise Exception(f"{name}: no present weights found")
    if active_scores.numel() == 0:
        warnings.warn(f"{name}: no active weights found")
    avg_sal, min_sal   = _stats(present_scores)
    avg_live, min_live = _stats(active_scores) if active_scores.numel() > 0 else (0.0, 0.0)
    _print_saliency(name, avg_sal, avg_live, state)
    return SaliencyResult(avg_sal, avg_live, min_sal, min_live)


class GradientSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    def measure_saliency(self, ctx: TrainingContext, state: NetworkState) -> SaliencyResult:
        def score_fn(layer, i):
            w = getattr(layer, WEIGHTS_ATTR)
            if w.grad is None:
                raise Exception(f"GradientSaliency: layer {i} has no grad after accumulate_gradients")
            return w.grad.detach()
        return _gradient_based_saliency(ctx, state, score_fn, "GradientSaliency")


class TaylorSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    def measure_saliency(self, ctx: TrainingContext, state: NetworkState) -> SaliencyResult:
        def score_fn(layer, i):
            w = getattr(layer, WEIGHTS_ATTR)
            if w.grad is None:
                raise Exception(f"TaylorSaliency: layer {i} has no grad after accumulate_gradients")
            return w.data * w.grad.detach()
        return _gradient_based_saliency(ctx, state, score_fn, "TaylorSaliency")


class HessianSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    # Diagonal-Fisher approximation: saliency(w_i) = ½ · mean(g_i²) · w_i²
    # mean(g²) is written to param._hessian_diag by accumulate_gradients (same pass, no extra cost).
    def measure_saliency(self, ctx: TrainingContext, state: NetworkState) -> SaliencyResult:
        def score_fn(layer, i):
            w = getattr(layer, WEIGHTS_ATTR)
            if not hasattr(w, '_hessian_diag'):
                raise Exception(f"HessianSaliency: layer {i} has no _hessian_diag after accumulate_gradients")
            return 0.5 * w._hessian_diag * w.data.pow(2)
        return _gradient_based_saliency(ctx, state, score_fn, "HessianSaliency")


# ── Neuron-based policy ───────────────────────────────────────────────────────

class NeuronActivationFrequencyPolicy(SaliencyMeasurementPolicy):
    """
    Saliency = per-neuron activation frequency: fraction of training samples
    where the neuron's output was > 0. Data is read directly from NetworkState
    (collected during compute_network_state's gradient-accumulation pass) —
    this policy needs no forward or backward pass of its own.

    active neurons  — fired at least once across all samples (state.active_neurons)

    avg_saliency              — mean activation freq over all neurons
    avg_saliency_contributing — mean activation freq over active neurons
    """
    def measure_saliency(self, ctx: TrainingContext, state: NetworkState) -> SaliencyResult:
        all_parts = []
        active_parts  = []

        for freq, active in zip(state.activation_freq, state.active_neurons):
            all_parts.append(freq)
            if active.any():
                active_parts.append(freq[active])

        if not all_parts:
            raise Exception("NeuronActivationFrequency: no neurons found")

        all_freq = torch.cat(all_parts)
        active_freq  = torch.cat(active_parts) if active_parts else torch.tensor([])

        if active_freq.numel() == 0:
            warnings.warn("NeuronActivationFrequency: no active neurons found (all neurons dead)")

        avg_sal, min_sal   = _stats(all_freq)
        avg_live, min_live = _stats(active_freq) if active_freq.numel() > 0 else (0.0, 0.0)

        total_n   = all_freq.numel()
        active_n  = sum(int(t.sum()) for t in state.active_neurons)

        print(f"[NeuronActivationFreq] avg freq (all) = {avg_sal:.4f} ({total_n} neurons total)  "
              f"contributing avg = {avg_live:.4f} ({active_n}/{total_n} neurons active, {100*active_n/total_n:.2f}%)")

        return SaliencyResult(avg_sal, avg_live, min_sal, min_live)


# ── Hyperflux (sampling-based) ────────────────────────────────────────────────

class HyperfluxSampleEstimationSaliencyMeasurementPolicy(SaliencyMeasurementPolicy):
    def __init__(self, n_samples: int = 3, sample_fraction: float = 0.05):
        self.n_samples = n_samples
        self.sample_fraction = sample_fraction

    def measure_saliency(self, ctx: TrainingContext, state: NetworkState) -> SaliencyResult:
        saved_grads = _save_grads(ctx.model)
        layers = get_prunable_layers(ctx.model)

        # We'll accumulate scores and counts per layer
        accumulated_scores = [torch.zeros_like(getattr(l, MASK_ATTR).data) for l in layers]
        sample_counts = [torch.zeros_like(getattr(l, MASK_ATTR).data) for l in layers]

        for s in range(self.n_samples):
            # 1. Sample and temporarily prune
            original_values = []
            sampled_indices_list = []
            
            for i, layer in enumerate(layers):
                mask_param = getattr(layer, MASK_ATTR)
                present_mask = state.present_masks[i]
                
                # Get indices of present weights
                present_indices = torch.where(present_mask)
                n_present = present_indices[0].numel()
                
                if n_present == 0:
                    sampled_indices_list.append(None)
                    original_values.append(None)
                    continue
                
                n_to_sample = max(1, int(n_present * self.sample_fraction))
                perm = torch.randperm(n_present, device=mask_param.device)[:n_to_sample]
                
                sampled_indices = tuple(idx[perm] for idx in present_indices)
                sampled_indices_list.append(sampled_indices)
                
                # Save and prune
                vals = mask_param.data[sampled_indices].clone()
                original_values.append(vals)
                mask_param.data[sampled_indices] = -1.0
            
            # 2. Accumulate gradients
            ctx.accumulate_mask_gradients()
            
            # 3. Collect and restore
            for i, layer in enumerate(layers):
                sampled_indices = sampled_indices_list[i]
                if sampled_indices is None:
                    continue
                
                mask_param = getattr(layer, MASK_ATTR)
                if mask_param.grad is not None:
                    # We want the gradient as it is (signed)
                    accumulated_scores[i][sampled_indices] += mask_param.grad.detach()[sampled_indices]
                    sample_counts[i][sampled_indices] += 1
                
                # Restore
                mask_param.data[sampled_indices] = original_values[i]

        # 4. Compute final stats
        present_scores_list = []
        active_scores_list = []
        
        for i, layer in enumerate(layers):
            # Compute mean for sampled weights in this layer
            mask_counts = sample_counts[i]
            sampled_mask = mask_counts > 0
            
            if not sampled_mask.any():
                continue
                
            layer_scores = accumulated_scores[i][sampled_mask] / mask_counts[sampled_mask]
            
            # Note: We only have scores for SAMPLED weights.
            # To estimate network-wide stats, we use these sampled scores.
            present_scores_list.append(layer_scores.flatten())
            
            # For active (contributing) weights, we sub-sample our already sampled scores
            # that intersect with state.active_masks[i]
            active_intersect = sampled_mask & state.active_masks[i]
            if active_intersect.any():
                active_layer_scores = accumulated_scores[i][active_intersect] / mask_counts[active_intersect]
                active_scores_list.append(active_layer_scores.flatten())

        if not present_scores_list:
             # If no weights were sampled at all (extremely sparse or small model), return zeros
             return SaliencyResult(0.0, 0.0, 0.0, 0.0)

        all_sampled_present = torch.cat(present_scores_list)
        all_sampled_active = torch.cat(active_scores_list) if active_scores_list else torch.tensor([])

        avg_sal, min_sal   = _stats(all_sampled_present)
        avg_live, min_live = _stats(all_sampled_active) if all_sampled_active.numel() > 0 else (0.0, 0.0)

        _restore_grads(ctx.model, saved_grads)
        ctx.model.train()
        
        _print_saliency("HyperfluxSaliency", avg_sal, avg_live, state)
        return SaliencyResult(avg_sal, avg_live, min_sal, min_live)

