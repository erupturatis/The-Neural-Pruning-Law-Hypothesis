from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn
import torch.optim as optim

from src.infrastructure.layers import LayerComposite, ModelCustom
from src.infrastructure.schedulers import AbstractScheduler


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

    # Direct handles
    model:     ModelCustom
    optimizer: optim.Optimizer

    # Training primitives
    # forward + backward + step
    train_one_epoch: Callable[[], None]
    # forward + backward + step with mask params
    train_one_epoch_hyperflux: Optional[Callable[[AbstractScheduler, optim.Optimizer], None]]
    # full test-set forward pass →  accuracy
    evaluate:        Callable[[], tuple[float,float]]

    # Gradient / curvature (e.g. Gradient, Hessian policies, etc)
    # Fisher diagonal (mean g²) is written to param._hessian_diag inside accumulate_gradients.
    accumulate_gradients: Optional[Callable[[], None]]
    # Custom accumulation specifically for mask parameters.
    accumulate_mask_gradients: Optional[Callable[[], None]]

    # Post-pruning cleanup
    # zeros optimizer moments / state entries for currently masked-out weights
    reset_optimizer_state: Optional[Callable[[], None]]

    # Epoch counter
    epoch_count: int = 0
