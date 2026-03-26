from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.infrastructure.configs_layers import MaskPruningFunction, \
    configs_get_layers_initialization, COUNT_FLOPS, get_flow_params_loss_abstract
from src.infrastructure.others import get_device
import math
from src.infrastructure.constants import WEIGHTS_ATTR, BIAS_ATTR, MASK_ATTR,  \
    WEIGHTS_BASE_ATTR, get_flow_params_init
import numpy as np

# =============================================================================
# SECTION 1: Layer Definitions
# =============================================================================

class ConfigsNetworkMask(TypedDict):
    mask_apply_enabled: bool
    mask_training_enabled: bool
    weights_training_enabled: bool

class LayerPrimitive(nn.Module, ABC):
    pass

class LayerComposite(nn.Module, ABC):
    pass

@dataclass
class ConfigsLayerLinear:
    in_features: int
    out_features: int
    bias_enabled: bool = True

class LayerLinearMaskImportance(LayerPrimitive):
    def __init__(self, configs_linear: ConfigsLayerLinear, configs_network: ConfigsNetworkMask):
        super().__init__()

        self.in_features = configs_linear.in_features
        self.out_features = configs_linear.out_features
        self.bias_enabled = configs_linear.bias_enabled

        self.mask_apply_enabled = configs_network['mask_apply_enabled']
        self.mask_training_enabled = configs_network['mask_training_enabled']
        self.weights_training_enabled = configs_network['weights_training_enabled']

        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, MASK_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_features)))

        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, MASK_ATTR).requires_grad = self.mask_training_enabled
        if self.bias_enabled:
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        self.init_parameters()

    def set_weights_training(self, weights_training_enabled: bool):
        self.weights_training_enabled = weights_training_enabled
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        if self.bias_enabled:
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

    def set_mask_training(self, mask_training_enabled:bool):
        self.mask_training_enabled = mask_training_enabled
        getattr(self, MASK_ATTR).requires_grad = self.mask_training_enabled

    def set_mask_apply(self, mask_apply_enabled:bool):
        self.mask_apply_enabled = mask_apply_enabled

    def get_masked_weights(self) -> any:
        masked_weight = getattr(self, WEIGHTS_ATTR)
        mask_changes = MaskPruningFunction.apply(getattr(self, MASK_ATTR))
        masked_weight = masked_weight * mask_changes
        return masked_weight

    def get_underlying_weights(self) -> any:
        masked_weight = getattr(self, WEIGHTS_ATTR)
        return masked_weight

    def init_parameters(self):
        init = configs_get_layers_initialization("fcn")
        init(getattr(self, WEIGHTS_ATTR))
        nn.init.uniform_(getattr(self, MASK_ATTR), a=get_flow_params_init(), b=get_flow_params_init() * 2.5)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        if COUNT_FLOPS:
            dense_flops, sparse_flops = get_forward_flops_fcn(self, input.shape)
            accumulate_flops(dense_flops, sparse_flops)

        masked_weight = getattr(self, WEIGHTS_ATTR)
        bias = torch.zeros(self.out_features, device=get_device())
        if hasattr(self, BIAS_ATTR):
            bias = getattr(self, BIAS_ATTR)

        if self.mask_apply_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_ATTR))
            masked_weight = masked_weight * mask_changes

        return F.linear(input, masked_weight, bias)

@dataclass
class ConfigsLayerConv2:
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int = 0
    stride: int = 1
    bias_enabled: bool = True

class LayerConv2MaskImportance(LayerPrimitive):
    def __init__(self, configs_conv2d: ConfigsLayerConv2, configs_network_masks: ConfigsNetworkMask):
        super(LayerConv2MaskImportance, self).__init__()

        self.in_channels = configs_conv2d.in_channels
        self.out_channels = configs_conv2d.out_channels
        self.kernel_size = configs_conv2d.kernel_size
        self.padding = configs_conv2d.padding
        self.stride = configs_conv2d.stride
        self.bias_enabled = configs_conv2d.bias_enabled

        self.mask_apply_enabled = configs_network_masks['mask_apply_enabled']
        self.mask_training_enabled = configs_network_masks['mask_training_enabled']
        self.weights_training_enabled = configs_network_masks['weights_training_enabled']

        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        setattr(self, MASK_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_channels)))

        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, MASK_ATTR).requires_grad = self.mask_training_enabled
        if self.bias_enabled:
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        self.init_parameters()

    def set_weights_training(self, weights_training_enabled: bool):
        self.weights_training_enabled = weights_training_enabled
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        if self.bias_enabled:
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

    def set_mask_training(self, mask_training_enabled: bool):
        self.mask_training_enabled = mask_training_enabled
        getattr(self, MASK_ATTR).requires_grad = self.mask_training_enabled

    def set_mask_apply(self, mask_apply_enabled: bool):
        self.mask_apply_enabled = mask_apply_enabled

    def get_masked_weights(self) -> any:
        masked_weights = getattr(self, WEIGHTS_ATTR)
        mask_changes = MaskPruningFunction.apply(getattr(self, MASK_ATTR))
        masked_weights = masked_weights * mask_changes
        return masked_weights

    def get_underlying_weights(self) -> any:
        return getattr(self, WEIGHTS_ATTR)

    def init_parameters(self):
        init = configs_get_layers_initialization("conv2d")
        init(getattr(self, WEIGHTS_ATTR))
        nn.init.uniform_(getattr(self, MASK_ATTR), a=get_flow_params_init(), b=get_flow_params_init() * 2.5)

        if hasattr(self, BIAS_ATTR):
            weights = getattr(self, WEIGHTS_ATTR)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        if COUNT_FLOPS:
            dense_flops, sparse_flops = get_forward_flops_cnn(self, input.shape)
            accumulate_flops(dense_flops, sparse_flops)

        masked_weights = getattr(self, WEIGHTS_ATTR)
        bias = torch.zeros(self.out_channels, device=get_device())
        if hasattr(self, BIAS_ATTR):
            bias = getattr(self, BIAS_ATTR)

        if self.mask_apply_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, MASK_ATTR))
            masked_weights = masked_weights * mask_changes

        return F.conv2d(input, masked_weights, bias, self.stride, self.padding)


# =============================================================================
# SECTION 2: FLOPs Tracking
# =============================================================================

accumulator = {
    "counter_dense": 0,
    "counter_sparse": 0
}

def accumulate_flops(flops_dense, flops_sparse):
    accumulator["counter_dense"] += flops_dense
    accumulator["counter_sparse"] += flops_sparse

def get_accumulated_flops():
    return accumulator

def get_forward_flops_fcn(self, input_shape: torch.Size) -> tuple[int, int]:
    """
    Returns (dense_flops, sparse_flops) for a fully-connected layer.
    dense_flops: batch_size * out_features * (2 * in_features - 1)
    sparse_flops: dense_flops scaled by active-weight density when mask_apply_enabled, else equal to dense_flops.
    """
    batch_size = input_shape[0]
    dense_flops = batch_size * self.out_features * (2 * self.in_features - 1)
    if self.mask_apply_enabled:
        density = (getattr(self, MASK_ATTR).data >= 0).float().mean().item()
        return dense_flops, int(dense_flops * density)
    return dense_flops, dense_flops

def get_forward_flops_cnn(self, input_shape: torch.Size) -> tuple[int, int]:
    """
    Returns (dense_flops, sparse_flops) for a convolutional layer.
    dense_flops: batch_size * out_channels * out_height * out_width * (2 * in_channels * kernel_size^2 - 1)
    sparse_flops: dense_flops scaled by active-weight density when mask_apply_enabled, else equal to dense_flops.
    """
    batch_size, _, in_height, in_width = input_shape
    out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width  = (in_width  + 2 * self.padding - self.kernel_size) // self.stride + 1
    kernel_ops = self.in_channels * self.kernel_size * self.kernel_size
    dense_flops = batch_size * self.out_channels * out_height * out_width * (2 * kernel_ops - 1)
    if self.mask_apply_enabled:
        density = (getattr(self, MASK_ATTR).data >= 0).float().mean().item()
        return dense_flops, int(dense_flops * density)
    return dense_flops, dense_flops

# =============================================================================
# SECTION 3: Statistics and Pure Functions
# =============================================================================

def get_layers_primitive(self: LayerComposite) -> List[LayerPrimitive]:
    layers: List[LayerPrimitive] = []
    for layer in self.registered_layers:
        if isinstance(layer, LayerPrimitive):
            layers.append(layer)
        elif isinstance(layer, LayerComposite):
            layers.extend(get_layers_primitive(layer))
    return layers

def get_total_and_remaining_params(self: LayerComposite) -> tuple[int, int]:
    total, remaining = 0, 0
    for layer in get_layers_primitive(self):
        mask = getattr(layer, MASK_ATTR).data
        total += mask.numel()
        remaining += int((mask >= 0).sum().item())
    return total, remaining

def get_total_params(self: LayerComposite) -> int:
    return sum(getattr(l, MASK_ATTR).data.numel() for l in get_layers_primitive(self))


def get_flow_params_loss(self: LayerComposite) -> tuple[float, torch.Tensor]:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    activations = torch.tensor(0, device=get_device(), dtype=torch.float)

    for layer in layers:
        layer_total, layer_remaining = get_flow_params_loss_abstract(layer)
        total += layer_total
        activations += layer_remaining

    return total, activations


