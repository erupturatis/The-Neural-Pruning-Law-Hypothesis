from typing import TYPE_CHECKING
import torch
from src.infrastructure.constants import WEIGHTS_PRUNING_ATTR, WEIGHTS_ATTR, get_flow_params_init

if TYPE_CHECKING:
    from src.infrastructure.layers import  LayerPrimitive
from src.infrastructure.others import get_device

def get_weights_params_decay_all_(layer_primitive: 'LayerPrimitive') -> torch.Tensor:
    total = 0
    remaining = torch.tensor(0, device=get_device(), dtype=torch.float)

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    remaining += (weights * weights).float().sum()

    return remaining

def get_weights_params_decay_only_present_(layer_primitive: 'LayerPrimitive') -> torch.Tensor:
    total = 0
    remaining = torch.tensor(0, device=get_device(), dtype=torch.float)

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_PRUNING_ATTR)
    remaining += (weights * weights * (mask_pruning > 0).float()).sum()

    return remaining

def get_weights_params_total_(layer_primitive: 'LayerPrimitive') -> int:
    total = 0
    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    total += weights.numel()
    return total

def get_flow_params_loss_raw_(layer_primitive: 'LayerPrimitive') -> tuple[float, torch.Tensor]:
    total = 0
    remaining = torch.tensor(0, device=get_device(), dtype=torch.float)

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_PRUNING_ATTR)

    total += weights.numel()
    remaining += mask_pruning.sum()

    return total, remaining

def get_flow_params_statistics_raw_(layer_primitive: 'LayerPrimitive') -> tuple[float, float]:
    total = 0
    remaining = 0

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, WEIGHTS_PRUNING_ATTR)

    total += weights.numel()
    remaining += (mask_pruning >= 0).float().sum()
    return total, remaining

