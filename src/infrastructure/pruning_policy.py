"""
pruning_policy.py
=================
Decoupled, network-agnostic pruning policies.

Design
------
Training and pruning are completely separated.  The training loop calls
``policy.prune_step(model, pruning_rate, get_batch)`` once per pruning event
and receives a ``PruningStepResult`` carrying the saliency metrics for NPLH
recording.  The policy knows nothing about the training loop; the training
loop knows nothing about how weights are scored.

All policies operate at the individual-weight level (unstructured pruning) on
any model that exposes ``get_layers_primitive()`` and has layers with
``weights`` / ``mask_pruning`` attributes.  Pruning is applied by setting
``mask_pruning = -1.0`` for selected weights.

Available policies
------------------
MagnitudePruningPolicy
    score = |w|                               (no data required)
    Prune lowest-scoring active weights.

TaylorPruningPolicy(criterion)
    score = |w · ∂L/∂w|                       (requires one data batch)
    First-order Taylor estimate of the loss change if weight w is zeroed.
    Prune lowest-scoring active weights.

GradientPruningPolicy(criterion)
    score = |∂L/∂w|                           (requires one data batch)
    Raw gradient magnitude — sensitivity independent of weight scale.
    Prune lowest-scoring active weights.

RandomRegrowthPruningPolicy(criterion, oversample_factor)
    Each step randomly prunes oversample_factor × n_net active weights,
    computes ∂L/∂w at w=0 for those candidates, then regorws the top
    (oversample_factor − 1) × n_net whose regrowth score
      −sign(w_orig) · ∂L/∂w  > 0
    (gradient-descent update points toward original weight sign).
    Net pruned per step = n_net, same as all other policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from src.infrastructure.constants import WEIGHTS_ATTR, WEIGHTS_PRUNING_ATTR

# Callable that returns one (inputs, targets) batch already on the right device

GetBatchFn = Callable[[], tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PruningStepResult:
    """
    Outcome of one pruning step, for NPLH recording.

    threshold    – boundary saliency score.
                   For magnitude/Taylor: score of the last pruned weight
                   (smallest kept is just above this).
                   For GraSP: score of the last pruned weight
                   (highest kept is just below this).
    avg_saliency – mean *absolute* saliency score of all active weights
                   before this pruning step.
    """
    threshold:    float
    avg_saliency: float


@dataclass
class FisherPruningStepResult(PruningStepResult):
    """
    Extended result for EmpiricalFisherPruningPolicy.

    threshold    – Fisher score (w²·g²) of the last pruned weight.
    avg_saliency – mean Fisher score over all active weights  (= avg_fisher).
    avg_magnitude  – mean |w| over active weights.
    avg_curvature  – mean g² over active weights  (diagonal Fisher estimate).
    """
    avg_magnitude: float = 0.0
    avg_curvature: float = 0.0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PruningPolicy(ABC):
    """
    Abstract pruning policy.

    Subclasses implement ``prune_step``, which scores active weights,
    applies the mask, and returns a ``PruningStepResult``.

    The class attribute ``method_tag`` is used to build CSV filenames so
    that different policies produce clearly named outputs.
    """

    method_tag: str = "IMP_unknown"

    @abstractmethod
    def prune_step(
        self,
        model,
        pruning_rate: float,
        get_batch: GetBatchFn | None = None,
    ) -> PruningStepResult:
        """
        Parameters
        ----------
        model        : any model with ``get_layers_primitive()``
        pruning_rate : fraction of *remaining* active weights to prune
        get_batch    : callable → (inputs, targets) on the correct device.
                       Required for data-dependent policies; ignored by
                       MagnitudePruningPolicy.
        """

    # ------------------------------------------------------------------
    # Shared helpers (private)
    # ------------------------------------------------------------------

    def _collect_scored_layers(self, model, score_fn):
        """
        Apply score_fn to every primitive layer and return a list of
        (layer, scores_tensor, active_bool_mask).
        scores_tensor has the same shape as layer.weights.
        """
        results = []
        for layer in model.get_layers_primitive():
            if not (
                hasattr(layer, WEIGHTS_ATTR)
                and hasattr(layer, WEIGHTS_PRUNING_ATTR)
            ):
                continue
            w      = getattr(layer, WEIGHTS_ATTR)
            m      = getattr(layer, WEIGHTS_PRUNING_ATTR)
            scores = score_fn(layer, w, m)
            active = m.data >= 0
            results.append((layer, scores, active))
        return results

    def _apply_pruning(
        self,
        layer_score_tuples,
        pruning_rate: float,
        prune_largest: bool = False,
    ) -> PruningStepResult:
        """
        Given per-layer score tensors, find the global threshold and set
        mask_pruning = -1.0 for the target fraction of active weights.

        prune_largest=False  → prune lowest scores  (magnitude, Taylor)
        prune_largest=True   → prune highest scores  (GraSP)
        """
        all_active_scores = []
        for _layer, scores, active in layer_score_tuples:
            if active.any():
                all_active_scores.append(scores[active].flatten())

        if not all_active_scores:
            raise ValueError("No active weights remaining to prune.")

        all_scores = torch.cat(all_active_scores)
        n_active   = all_scores.numel()
        if n_active == 0:
            raise ValueError("No active weights remaining to prune.")

        n_to_prune = max(1, min(int(n_active * pruning_rate), n_active))

        avg_saliency = float(all_scores.abs().mean().item())

        # Use topk to get the exact n_to_prune scores (ties handled by rank,
        # not by value, so we never accidentally prune more than requested).
        topk_vals, _ = torch.topk(all_scores, n_to_prune, largest=prune_largest)
        threshold = float(topk_vals[-1].item())  # boundary score

        # Apply masks in two passes to respect the exact budget:
        #   pass 1 – weights strictly beyond the threshold (no ties)
        #   pass 2 – weights exactly at the threshold, up to remaining budget
        with torch.no_grad():
            budget = n_to_prune
            for layer, scores, active in layer_score_tuples:
                m = getattr(layer, WEIGHTS_PRUNING_ATTR).data
                if prune_largest:
                    strict = active & (scores > threshold)
                else:
                    strict = active & (scores < threshold)
                m[strict] = -1.0
                budget -= int(strict.sum().item())

            # Fill remaining budget from tied weights at exactly the threshold
            if budget > 0:
                for layer, scores, active in layer_score_tuples:
                    m = getattr(layer, WEIGHTS_PRUNING_ATTR).data
                    # Re-read live mask so we don't re-prune already pruned
                    still_active = m >= 0
                    tied = still_active & (scores == threshold)
                    positions = tied.nonzero(as_tuple=False)
                    for pos in positions:
                        if budget <= 0:
                            break
                        m[tuple(pos.tolist())] = -1.0
                        budget -= 1
                    if budget <= 0:
                        break

        return PruningStepResult(threshold=threshold, avg_saliency=avg_saliency)


# ---------------------------------------------------------------------------
# Policy 1: Magnitude (standard IMP)
# ---------------------------------------------------------------------------

class MagnitudePruningPolicy(PruningPolicy):
    """
    Prune the fraction of active weights with the smallest absolute magnitude.
    No data required.  Equivalent to ``prune_model_globally`` in layers.py.
    """

    method_tag = "IMP_magnitude"

    def prune_step(
        self,
        model,
        pruning_rate: float,
        get_batch: GetBatchFn | None = None,
    ) -> PruningStepResult:
        def _score(layer, w, m):
            return torch.abs(w.data)

        tuples = self._collect_scored_layers(model, _score)
        return self._apply_pruning(tuples, pruning_rate, prune_largest=False)


# ---------------------------------------------------------------------------
# Policy 2: Taylor expansion  (weight-level, unstructured)
# ---------------------------------------------------------------------------

class TaylorPruningPolicy(PruningPolicy):
    """
    First-order Taylor criterion.

        score(w_ij) = |w_ij · ∂L/∂w_ij|

    Estimates the absolute change in loss if weight w_ij were zeroed.
    Prune the lowest-scoring active weights.

    Requires one data batch and a loss criterion.
    """

    method_tag = "taylor"

    def __init__(self, criterion: nn.Module):
        self.criterion = criterion

    def prune_step(
        self,
        model,
        pruning_rate: float,
        get_batch: GetBatchFn | None = None,
    ) -> PruningStepResult:
        if get_batch is None:
            raise ValueError("TaylorPruningPolicy requires get_batch.")

        weight_params = [
            getattr(layer, WEIGHTS_ATTR)
            for layer in model.get_layers_primitive()
            if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR)
        ]

        # Temporarily enable grad for weights
        orig_req_grad = [p.requires_grad for p in weight_params]
        for p in weight_params:
            p.requires_grad_(True)

        model.eval()
        model.zero_grad()
        inputs, targets = get_batch()
        loss = self.criterion(model(inputs), targets)
        loss.backward()

        def _score(layer, w, m):
            g = w.grad
            if g is None:
                return torch.zeros_like(w.data)
            return torch.abs(w.data * g.detach())

        tuples = self._collect_scored_layers(model, _score)

        # Restore
        model.zero_grad()
        for p, req in zip(weight_params, orig_req_grad):
            p.requires_grad_(req)

        return self._apply_pruning(tuples, pruning_rate, prune_largest=False)


# ---------------------------------------------------------------------------
# Policy 3: Gradient magnitude  (weight-level, unstructured)
# ---------------------------------------------------------------------------

class GradientPruningPolicy(PruningPolicy):
    """
    Raw gradient-magnitude criterion.

        score(w_ij) = |∂L/∂w_ij|

    Measures loss sensitivity to a perturbation at the current weight value,
    independent of weight scale.  Unlike Taylor, a tiny weight with a large
    gradient is kept; a large weight with a near-zero gradient is pruned.

    Requires one data batch and a loss criterion.
    """

    method_tag = "gradient"

    def __init__(self, criterion: nn.Module):
        self.criterion = criterion

    def prune_step(
        self,
        model,
        pruning_rate: float,
        get_batch: GetBatchFn | None = None,
    ) -> PruningStepResult:
        if get_batch is None:
            raise ValueError("GradientPruningPolicy requires get_batch.")

        weight_params = [
            getattr(layer, WEIGHTS_ATTR)
            for layer in model.get_layers_primitive()
            if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR)
        ]

        orig_req_grad = [p.requires_grad for p in weight_params]
        for p in weight_params:
            p.requires_grad_(True)

        model.eval()
        model.zero_grad()
        inputs, targets = get_batch()
        loss = self.criterion(model(inputs), targets)
        loss.backward()

        def _score(layer, w, m):
            g = w.grad
            if g is None:
                return torch.zeros_like(w.data)
            return torch.abs(g.detach())

        tuples = self._collect_scored_layers(model, _score)

        model.zero_grad()
        for p, req in zip(weight_params, orig_req_grad):
            p.requires_grad_(req)

        return self._apply_pruning(tuples, pruning_rate, prune_largest=False)


# ---------------------------------------------------------------------------
# Policy 4: Random prune + gradient-guided regrowth  (weight-level)
# ---------------------------------------------------------------------------

class RandomRegrowthPruningPolicy(PruningPolicy):
    """
    Random prune + gradient-guided regrowth.

    Each step:
      1. Uniformly at random select oversample_factor × n_net active weights
         as candidates (n_net = pruning_rate × n_active).
      2. Zero their weight values while keeping their mask active (≥ 0), so
         that ∂L/∂w at w = 0 is available via normal backprop.
      3. Run one forward+backward pass.
      4. Compute the regrowth score for each candidate:
             score = −sign(w_orig) · ∂L/∂w|_{w=0}
         score > 0  ⟺  the gradient-descent update (−grad) points in the
         same direction as the weight's original sign — the network "wants
         this weight back".
      5. Regrow the top (oversample_factor − 1) × n_net candidates with
         score > 0 (restore their original value; mask stays active).
      6. Permanently prune the remainder (mask → −1).
         Net pruned per step ≈ n_net, same budget as all other policies.

    Saliency threshold reported = min regrowth score among regrown weights
    (0.0 when nothing is regrown).
    Avg saliency = mean |w| of all active weights before the step.

    Parameters
    ----------
    criterion         : loss function (nn.Module)
    oversample_factor : ratio of randomly pruned to net pruned per step.
                        Default 2.0 → prune 2×, regrow 1×.  Must be > 1.
    """

    method_tag = "random_regrowth"

    def __init__(self, criterion: nn.Module, oversample_factor: float = 2.0):
        self.criterion = criterion
        self.oversample_factor = max(1.01, float(oversample_factor))

    def prune_step(
        self,
        model,
        pruning_rate: float,
        get_batch: GetBatchFn | None = None,
    ) -> PruningStepResult:
        if get_batch is None:
            raise ValueError("RandomRegrowthPruningPolicy requires get_batch.")

        layers = [
            layer for layer in model.get_layers_primitive()
            if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR)
        ]
        if not layers:
            raise ValueError("No prunable layers found.")

        device = getattr(layers[0], WEIGHTS_ATTR).device

        # ── 1. Active weight inventory & avg saliency ────────────────────
        active_mags = []
        n_active = 0
        for layer in layers:
            w = getattr(layer, WEIGHTS_ATTR)
            m = getattr(layer, WEIGHTS_PRUNING_ATTR)
            active = m.data >= 0
            n_active += int(active.sum().item())
            if active.any():
                active_mags.append(w.data[active].abs())

        if n_active == 0:
            raise ValueError("No active weights remaining.")

        avg_saliency = float(torch.cat(active_mags).mean().item())

        n_net    = max(1, int(n_active * pruning_rate))
        n_random = min(n_active, max(n_net + 1, round(n_net * self.oversample_factor)))
        n_regrow = n_random - n_net  # target regrowth budget

        # ── 2. Uniform random sample of n_random active weights ──────────
        # Assign uniform random scores to active positions, -inf elsewhere;
        # global topk = uniform random sample without replacement.
        rand_scores   = []
        layer_w_numel = []
        for layer in layers:
            m = getattr(layer, WEIGHTS_PRUNING_ATTR)
            active_flat = (m.data >= 0).flatten()
            r = torch.where(
                active_flat,
                torch.rand(active_flat.numel(), device=device),
                torch.full((active_flat.numel(),), float('-inf'), device=device),
            )
            rand_scores.append(r)
            layer_w_numel.append(active_flat.numel())

        all_rand = torch.cat(rand_scores)
        _, global_selected = torch.topk(all_rand, n_random)  # global flat indices

        # Map global flat indices → (layer index, local flat index within layer)
        offsets = torch.zeros(len(layers), dtype=torch.long, device=device)
        acc = 0
        for li, n in enumerate(layer_w_numel):
            offsets[li] = acc
            acc += n

        # searchsorted(right=True): first j where offsets[j] > g → layer = j-1
        li_of_selected    = torch.searchsorted(offsets.contiguous(),
                                               global_selected.contiguous(),
                                               right=True) - 1
        local_of_selected = global_selected - offsets[li_of_selected]

        # Per-layer boolean candidate masks over the flattened weight tensor
        cand_masks_flat = []
        for li, layer in enumerate(layers):
            mask = torch.zeros(layer_w_numel[li], dtype=torch.bool, device=device)
            in_layer = li_of_selected == li
            if in_layer.any():
                mask[local_of_selected[in_layer]] = True
            cand_masks_flat.append(mask)

        # ── 3. Save original values, zero candidates ─────────────────────
        # Mask stays ≥ 0 so MaskPruningFunctionConstant returns 1 in forward
        # → weight contributes 0 to output, ∂(w·1)/∂w = 1 → full grad flows.
        saved_vals = []
        for layer, cand_flat in zip(layers, cand_masks_flat):
            w  = getattr(layer, WEIGHTS_ATTR)
            wv = w.data.view(-1)
            saved_vals.append(wv[cand_flat].clone())
            wv[cand_flat] = 0.0

        # ── 4. Forward + backward ─────────────────────────────────────────
        weight_params = [getattr(layer, WEIGHTS_ATTR) for layer in layers]
        orig_req_grad = [p.requires_grad for p in weight_params]
        for p in weight_params:
            p.requires_grad_(True)

        model.eval()
        model.zero_grad()
        inputs, targets = get_batch()
        loss = self.criterion(model(inputs), targets)
        loss.backward()

        # ── 5. Regrowth score = −sign(w_orig) · ∂L/∂w|_{w=0} ────────────
        rg_scores_per_layer = []
        for layer, cand_flat, w_orig in zip(layers, cand_masks_flat, saved_vals):
            n_cands = w_orig.numel()
            if n_cands == 0:
                rg_scores_per_layer.append(torch.zeros(0, device=device))
                continue
            w = getattr(layer, WEIGHTS_ATTR)
            g = w.grad
            if g is None:
                rg_scores_per_layer.append(
                    torch.full((n_cands,), float('-inf'), device=device))
                continue
            grad_cands = g.data.view(-1)[cand_flat]
            signs  = torch.sign(w_orig)
            scores = -signs * grad_cands           # >0 ⟺ GD update aligns with original sign
            scores[w_orig == 0.0] = float('-inf')  # original direction undefined
            rg_scores_per_layer.append(scores)

        model.zero_grad()
        for p, req in zip(weight_params, orig_req_grad):
            p.requires_grad_(req)

        # ── 6. Global regrowth ranking ────────────────────────────────────
        all_rg_scores = torch.cat(rg_scores_per_layer)  # length = n_random
        k = min(n_regrow, int((all_rg_scores > float('-inf')).sum().item()))

        regrow_set = torch.zeros(n_random, dtype=torch.bool, device=device)
        threshold  = 0.0
        n_actually_regrown = 0

        if k > 0:
            top_vals, top_indices = torch.topk(all_rg_scores, k)
            valid = top_vals > 0
            regrow_set[top_indices[valid]] = True
            n_actually_regrown = int(valid.sum().item())
            if n_actually_regrown > 0:
                threshold = float(top_vals[valid].min().item())

        # ── 7. Apply regrowth / permanent pruning ────────────────────────
        cand_offset = 0
        with torch.no_grad():
            for layer, cand_flat, w_orig in zip(layers, cand_masks_flat, saved_vals):
                n_cands = w_orig.numel()
                if n_cands == 0:
                    continue

                layer_regrow = regrow_set[cand_offset: cand_offset + n_cands]
                w = getattr(layer, WEIGHTS_ATTR)
                m = getattr(layer, WEIGHTS_PRUNING_ATTR)
                cand_indices = cand_flat.nonzero(as_tuple=False).squeeze(1)

                # Restore all original weight values first
                w.data.view(-1)[cand_indices] = w_orig

                # Permanently prune those not regrown
                prune_indices = cand_indices[~layer_regrow]
                m.data.view(-1)[prune_indices] = -1.0

                cand_offset += n_cands

        return PruningStepResult(threshold=threshold, avg_saliency=avg_saliency)


# ---------------------------------------------------------------------------
# Policy 5: Pure random pruning  (weight-level, unstructured)
# ---------------------------------------------------------------------------

class RandomPruningPolicy(PruningPolicy):
    """
    Uniformly random pruning — no scoring, no gradient needed.

    Selects pruning_rate fraction of active weights uniformly at random
    and permanently removes them (mask → -1).

    threshold    = 0.0 always (selection is random, not score-based).
    avg_saliency = mean |w| of active weights *before* this pruning step.

    Use this to test whether saliency growth is intrinsic to training
    (independent of which weights are removed).
    """

    method_tag = "random_pruning"

    def prune_step(
        self,
        model,
        pruning_rate: float,
        get_batch: GetBatchFn | None = None,
    ) -> PruningStepResult:
        layers = [
            layer for layer in model.get_layers_primitive()
            if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR)
        ]
        if not layers:
            raise ValueError("No prunable layers found.")

        device = getattr(layers[0], WEIGHTS_ATTR).device

        # ── avg saliency + random scores for active weights ───────────────
        active_mags   = []
        rand_scores   = []
        layer_w_numel = []
        n_active      = 0

        for layer in layers:
            w = getattr(layer, WEIGHTS_ATTR)
            m = getattr(layer, WEIGHTS_PRUNING_ATTR)
            active_flat = (m.data >= 0).flatten()
            n_active += int(active_flat.sum().item())
            if active_flat.any():
                active_mags.append(w.data.flatten()[active_flat].abs())
            r = torch.where(
                active_flat,
                torch.rand(active_flat.numel(), device=device),
                torch.full((active_flat.numel(),), float('-inf'), device=device),
            )
            rand_scores.append(r)
            layer_w_numel.append(active_flat.numel())

        if n_active == 0:
            raise ValueError("No active weights remaining.")

        avg_saliency = float(torch.cat(active_mags).mean().item())
        n_prune      = max(1, int(n_active * pruning_rate))

        # ── global random sample via topk on uniform scores ───────────────
        all_rand = torch.cat(rand_scores)
        _, global_selected = torch.topk(all_rand, n_prune)

        offsets = torch.zeros(len(layers), dtype=torch.long, device=device)
        acc = 0
        for li, n in enumerate(layer_w_numel):
            offsets[li] = acc
            acc += n

        li_of_selected    = torch.searchsorted(offsets.contiguous(),
                                               global_selected.contiguous(),
                                               right=True) - 1
        local_of_selected = global_selected - offsets[li_of_selected]

        with torch.no_grad():
            for li, layer in enumerate(layers):
                m = getattr(layer, WEIGHTS_PRUNING_ATTR)
                in_layer = li_of_selected == li
                if in_layer.any():
                    m.data.view(-1)[local_of_selected[in_layer]] = -1.0

        return PruningStepResult(threshold=0.0, avg_saliency=avg_saliency)


# ---------------------------------------------------------------------------
# Policy 6: Empirical Fisher diagonal  (weight-level, unstructured)
# ---------------------------------------------------------------------------

class EmpiricalFisherPruningPolicy(PruningPolicy):
    """
    Empirical Fisher diagonal pruning.

        score(w_i) = w_i² · E[g_i²]

    where E[g_i²] is estimated by averaging g_i² over `n_batches` independent
    forward+backward passes.  Weights with the lowest score are pruned.

    Interpretation
    --------------
    - w_i² alone → magnitude²  (standard IMP, squared).
    - E[g_i²] alone → diagonal Fisher / gradient variance: how much the loss
      is sensitive to perturbations at w_i, regardless of weight size.
    - Combined: a small weight in a high-curvature region can outrank a large
      weight in a flat region.  This is the second-order approximation of loss
      change from zeroing w_i (Optimal Brain Damage derivation).

    Returns FisherPruningStepResult which extends PruningStepResult with:
        avg_magnitude  – mean |w| of active weights (same as magnitude IMP)
        avg_curvature  – mean E[g²] of active weights (new, curvature only)
        avg_saliency   – mean Fisher score w²·E[g²]  (combined)
        threshold      – Fisher score of the last pruned weight

    Parameters
    ----------
    criterion : loss function
    n_batches : number of batches to average g² over (default 1).
                More batches → stabler curvature estimate but slower.
    """

    method_tag = "fisher"

    def __init__(self, criterion: nn.Module, n_batches: int = 1):
        self.criterion = criterion
        self.n_batches = max(1, n_batches)

    def prune_step(
        self,
        model,
        pruning_rate: float,
        get_batch: GetBatchFn | None = None,
    ) -> FisherPruningStepResult:
        if get_batch is None:
            raise ValueError("EmpiricalFisherPruningPolicy requires get_batch.")

        layers = [
            layer for layer in model.get_layers_primitive()
            if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR)
        ]
        if not layers:
            raise ValueError("No prunable layers found.")

        weight_params = [getattr(layer, WEIGHTS_ATTR) for layer in layers]
        orig_req_grad = [p.requires_grad for p in weight_params]
        for p in weight_params:
            p.requires_grad_(True)

        # Accumulate g² over n_batches passes
        g2_accum = [torch.zeros_like(getattr(l, WEIGHTS_ATTR).data) for l in layers]

        model.eval()
        for _ in range(self.n_batches):
            model.zero_grad()
            inputs, targets = get_batch()
            loss = self.criterion(model(inputs), targets)
            loss.backward()
            for i, layer in enumerate(layers):
                g = getattr(layer, WEIGHTS_ATTR).grad
                if g is not None:
                    g2_accum[i] += g.detach() ** 2

        model.zero_grad()
        for p, req in zip(weight_params, orig_req_grad):
            p.requires_grad_(req)

        # Average g² over batches
        for i in range(len(g2_accum)):
            g2_accum[i] /= self.n_batches

        # Collect per-layer scores and stats for active weights
        all_fisher, all_magnitude, all_curvature = [], [], []

        def _score(layer, w, m):
            idx = layers.index(layer)
            return w.data ** 2 * g2_accum[idx]

        layer_score_tuples = self._collect_scored_layers(model, _score)

        for layer, scores, active in layer_score_tuples:
            if not active.any():
                continue
            idx = layers.index(layer)
            w   = getattr(layer, WEIGHTS_ATTR)
            all_fisher.append(scores[active].flatten())
            all_magnitude.append(w.data[active].abs().flatten())
            all_curvature.append(g2_accum[idx][active].flatten())

        if not all_fisher:
            raise ValueError("No active weights remaining.")

        avg_fisher    = float(torch.cat(all_fisher).mean().item())
        avg_magnitude = float(torch.cat(all_magnitude).mean().item())
        avg_curvature = float(torch.cat(all_curvature).mean().item())

        base = self._apply_pruning(layer_score_tuples, pruning_rate, prune_largest=False)

        return FisherPruningStepResult(
            threshold     = base.threshold,
            avg_saliency  = avg_fisher,
            avg_magnitude = avg_magnitude,
            avg_curvature = avg_curvature,
        )

    def score_only(
        self,
        model,
        get_batch: GetBatchFn,
    ) -> FisherPruningStepResult:
        """
        Compute Fisher statistics on active weights without pruning anything.
        Used to score after warm-up epochs when gradients are still informative.
        threshold is set to the minimum Fisher score among active weights
        (the weight that would be pruned first).
        """
        layers = [
            layer for layer in model.get_layers_primitive()
            if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR)
        ]

        weight_params = [getattr(layer, WEIGHTS_ATTR) for layer in layers]
        orig_req_grad = [p.requires_grad for p in weight_params]
        for p in weight_params:
            p.requires_grad_(True)

        g2_accum = [torch.zeros_like(getattr(l, WEIGHTS_ATTR).data) for l in layers]

        model.eval()
        for _ in range(self.n_batches):
            model.zero_grad()
            inputs, targets = get_batch()
            loss = self.criterion(model(inputs), targets)
            loss.backward()
            for i, layer in enumerate(layers):
                g = getattr(layer, WEIGHTS_ATTR).grad
                if g is not None:
                    g2_accum[i] += g.detach() ** 2

        model.zero_grad()
        for p, req in zip(weight_params, orig_req_grad):
            p.requires_grad_(req)

        for i in range(len(g2_accum)):
            g2_accum[i] /= self.n_batches

        all_fisher, all_magnitude, all_curvature = [], [], []
        for i, layer in enumerate(layers):
            w      = getattr(layer, WEIGHTS_ATTR)
            m      = getattr(layer, WEIGHTS_PRUNING_ATTR)
            active = m.data >= 0
            if not active.any():
                continue
            fisher = (w.data ** 2 * g2_accum[i])[active].flatten()
            all_fisher.append(fisher)
            all_magnitude.append(w.data[active].abs().flatten())
            all_curvature.append(g2_accum[i][active].flatten())

        if not all_fisher:
            return FisherPruningStepResult(0.0, 0.0, 0.0, 0.0)

        cat_fisher = torch.cat(all_fisher)
        return FisherPruningStepResult(
            threshold     = float(cat_fisher.min().item()),
            avg_saliency  = float(cat_fisher.mean().item()),
            avg_magnitude = float(torch.cat(all_magnitude).mean().item()),
            avg_curvature = float(torch.cat(all_curvature).mean().item()),
        )


# ---------------------------------------------------------------------------
# Multi-saliency measurement  (magnitude + gradient + Taylor in one pass)
# ---------------------------------------------------------------------------

@dataclass
class MultiSaliencyResult:
    """
    All three saliency scores measured on active weights after a pruning step.

    Each score type has a min (smallest value among active weights) and an avg
    (mean absolute value over all active weights).

    mag_*    – |w|                  weight magnitude
    grad_*   – |∂L/∂w|             gradient magnitude
    taylor_* – |w · ∂L/∂w|        Taylor first-order importance score
    """
    mag_min:    float
    mag_avg:    float
    grad_min:   float
    grad_avg:   float
    taylor_min: float
    taylor_avg: float


def measure_multi_saliency(
    model,
    criterion: nn.Module,
    get_batch: GetBatchFn,
) -> MultiSaliencyResult:
    """
    Measure magnitude, gradient, and Taylor saliency on all active weights
    using a single forward+backward pass.

    Intended for random-pruning experiments where the pruning criterion is
    independent of the saliency metrics being recorded.

    Parameters
    ----------
    model     : any model with get_layers_primitive()
    criterion : loss function
    get_batch : callable → (inputs, targets) already on the correct device

    Returns
    -------
    MultiSaliencyResult  with min and avg for each of the three score types.
    """
    layers = [
        layer for layer in model.get_layers_primitive()
        if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR)
    ]

    weight_params = [getattr(layer, WEIGHTS_ATTR) for layer in layers]
    orig_req_grad = [p.requires_grad for p in weight_params]
    for p in weight_params:
        p.requires_grad_(True)

    model.eval()
    model.zero_grad()
    inputs, targets = get_batch()
    loss = criterion(model(inputs), targets)
    loss.backward()

    mag_scores    = []
    grad_scores   = []
    taylor_scores = []

    for layer in layers:
        w = getattr(layer, WEIGHTS_ATTR)
        m = getattr(layer, WEIGHTS_PRUNING_ATTR)
        active = m.data >= 0
        if not active.any():
            continue
        w_active = w.data[active].abs()
        mag_scores.append(w_active)
        g = w.grad
        if g is not None:
            g_active = g.data[active].abs()
            grad_scores.append(g_active)
            taylor_scores.append(w_active * g_active)

    model.zero_grad()
    for p, req in zip(weight_params, orig_req_grad):
        p.requires_grad_(req)

    def _stats(tensors):
        if not tensors:
            return 0.0, 0.0
        t = torch.cat(tensors)
        return float(t.min().item()), float(t.mean().item())

    mag_min,    mag_avg    = _stats(mag_scores)
    grad_min,   grad_avg   = _stats(grad_scores)
    taylor_min, taylor_avg = _stats(taylor_scores)

    return MultiSaliencyResult(
        mag_min=mag_min,       mag_avg=mag_avg,
        grad_min=grad_min,     grad_avg=grad_avg,
        taylor_min=taylor_min, taylor_avg=taylor_avg,
    )
