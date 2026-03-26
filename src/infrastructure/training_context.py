from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from src.infrastructure.layers import LayerComposite


@dataclass
class TrainingContext:
    """
    Everything a policy needs, nothing it doesn't.

    model / optimizer are direct handles — policies use them for weight
    inspection, mask application, and optimizer state resets.

    All data-touching operations are zero-arg callables. They close over the
    dataset, DataLoader, criterion, and any multi-GPU setup so that policies
    are fully decoupled from those details and swap freely across experiments.
    """

    # ── Direct handles ────────────────────────────────────────────────────────
    model:     LayerComposite
    optimizer: optim.Optimizer

    # ── Core training primitives (all policies that train/evaluate need these) ─
    train_one_epoch: Callable[[], float]   # forward + backward + step →  avg training loss
    evaluate:        Callable[[], float]   # full test-set forward pass →  accuracy

    # ── Gradient / curvature  (Gradient, Taylor, Hessian policies) ────────────
    # forward + backward over training data, fills param.grad, no optimizer step
    accumulate_gradients:     Optional[Callable[[], None]]              = None
    # per-parameter Hessian diagonal estimates (Hutchinson or double-backprop)
    compute_hessian_diagonal: Optional[Callable[[], dict[str, Tensor]]] = None

    # ── Activation statistics  (APoZ and similar) ─────────────────────────────
    # full training-set forward under torch.no_grad(); triggers registered hooks
    run_forward: Optional[Callable[[], None]] = None

    # ── Post-pruning cleanup ──────────────────────────────────────────────────
    # zeros optimizer moments / state entries for currently masked-out weights
    reset_optimizer_state: Optional[Callable[[], None]] = None

    # ── Epoch counter ─────────────────────────────────────────────────────────
    # incremented by the factory each time train_one_epoch is called;
    # read by NPLHEpochLimitingStoppingPolicy
    epoch_count: int = 0
