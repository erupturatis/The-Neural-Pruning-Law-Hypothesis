from src.infrastructure.layer_initializations import kaiming_sqrt5, kaiming_relu, kaiming_sqrt0, bad_initialization
from typing import Dict

from src.infrastructure.others import get_device

def configs_get_layers_all_initialization(layer_name: str) -> Dict:
    return _configs_layers_init

def configs_get_layers_initialization(layer_name) -> callable:
    return _configs_layers_init[layer_name]

def configs_layers_initialization_all_kaiming_relu():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": kaiming_relu,
        "conv2d": kaiming_relu
    }

def configs_layers_initialization_all_kaiming_sqrt5():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": kaiming_sqrt5,
        "conv2d": kaiming_sqrt5
    }

def configs_layers_initialization_all_kaiming_sqrt0():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": kaiming_sqrt0,
        "conv2d": kaiming_sqrt0
    }

def configs_layers_initialization_all_bad():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": bad_initialization,
        "conv2d": bad_initialization
    }

import numpy as np
import torch
from src.infrastructure.constants import GRADIENT_IDENTITY_SCALER, WEIGHTS_ATTR, MASK_ATTR


class _MaskPruningFunctionConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_param):
        mask = mask_param
        mask_thresholded = (mask >= 0).float()

        ctx.save_for_backward(mask, mask_thresholded)
        return mask_thresholded

    @staticmethod
    def backward(ctx, grad_output):
        mask, _ = ctx.saved_tensors
        grad_mask_param = grad_output * GRADIENT_IDENTITY_SCALER

        return grad_mask_param

def _get_flow_params_loss_raw(layer_primitive: 'LayerPrimitive') -> tuple[float, torch.Tensor]:
    total = 0
    remaining = torch.tensor(0, device=get_device(), dtype=torch.float)

    weights = getattr(layer_primitive, WEIGHTS_ATTR)
    mask_pruning = getattr(layer_primitive, MASK_ATTR)

    total += weights.numel()
    remaining += mask_pruning.sum()

    return total, remaining

"""
ABSTRACTION CONSTANTS TO BE USED
"""

COUNT_FLOPS = False
MaskPruningFunction = _MaskPruningFunctionConstant
get_flow_params_loss_abstract = _get_flow_params_loss_raw

_configs_layers_init = {
    "fcn": kaiming_sqrt5,
    "conv2d": kaiming_sqrt5
}

