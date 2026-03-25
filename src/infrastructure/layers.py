from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict
from src.infrastructure.configs_layers import MaskPruningFunction,  get_flow_params_loss_abstract, \
    get_flow_params_statistics_abstract, configs_get_layers_initialization
from src.infrastructure.parameters_mask_processors import get_weights_params_total_, \
    get_weights_params_decay_only_present_, get_weights_params_decay_all_
from src.infrastructure.mask_functions import MaskPruningFunctionConstant
from src.infrastructure.others import get_device
import math
from src.infrastructure.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR,  \
    WEIGHTS_BASE_ATTR, get_flow_params_init
import numpy as np 

class ConfigsNetworkMasksImportance(TypedDict):
    mask_pruning_enabled: bool
    weights_training_enabled: bool

class LayerPrimitive(nn.Module, ABC):
    pass

class LayerComposite(nn.Module, ABC):
    @abstractmethod
    def get_layers_primitive(self) -> List[LayerPrimitive]:
        pass

def get_layer_composite_flow_params_statistics(self: LayerComposite) -> tuple[float, float]:
    layers = get_layers_primitive(self)
    total = 0
    remaining = 0
    for layer in layers:
        layer_total, layer_remaining = get_flow_params_statistics_abstract(layer)
        total += layer_total
        remaining += layer_remaining

    return total, remaining

def get_parameters_total_count(self: LayerComposite) -> int:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    for layer in layers:
        layer_total = get_weights_params_total_(layer)
        total += layer_total

    return total

def get_weight_decay_only_for_all(self: LayerComposite) -> torch.Tensor:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    weights = torch.tensor(0, device=get_device(), dtype=torch.float)
    for layer in layers:
        decay_params = get_weights_params_decay_all_(layer)
        weights += decay_params

    return weights

def get_weight_decay_only_for_present(self: LayerComposite) -> torch.Tensor:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    weights = torch.tensor(0, device=get_device(), dtype=torch.float)
    for layer in layers:
        decay_params = get_weights_params_decay_only_present_(layer)
        weights += decay_params

    return weights

def get_flow_params_loss(self: LayerComposite) -> tuple[float, torch.Tensor]:
    layers: List[LayerPrimitive] = get_layers_primitive(self)
    total = 0
    activations = torch.tensor(0, device=get_device(), dtype=torch.float)

    for layer in layers:
        layer_total, layer_remaining = get_flow_params_loss_abstract(layer)
        total += layer_total
        activations += layer_remaining

    return total, activations

def get_layers_primitive(self: LayerComposite) -> List[LayerPrimitive]:
    layers: List[LayerPrimitive] = []
    for layer in self.registered_layers:
        if isinstance(layer, LayerPrimitive):
            layers.append(layer)
        elif isinstance(layer, LayerComposite):
            layers.extend(get_layers_primitive(layer))

    return layers

accumulator = {
    "counter_dense": 0,
    "counter_sparse": 0
}

def accumulate_flops(flops_dense, flops_sparse):
    accumulator["counter_dense"] += flops_dense
    accumulator["counter_sparse"] += flops_sparse

def get_accumulated_flops():
    return accumulator

@dataclass
class ConfigsLayerLinear:
    in_features: int
    out_features: int
    bias_enabled: bool = True

class LayerLinearMaskImportance(LayerPrimitive):
    def __init__(self, configs_linear: ConfigsLayerLinear, configs_network: ConfigsNetworkMasksImportance):
        super().__init__()

        self.in_features = configs_linear.in_features
        self.out_features = configs_linear.out_features
        self.bias_enabled = configs_linear.bias_enabled

        self.mask_pruning_enabled = configs_network['mask_pruning_enabled']
        self.weights_training_enabled = configs_network['weights_training_enabled']

        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        setattr(self, WEIGHTS_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_features, self.in_features)))

        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_features)))
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled
        getattr(self, WEIGHTS_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled

        self.init_parameters()

    def get_bias_enabled(self):
        return self.bias_enabled

    def enable_weights_training(self):
        self.weights_training_enabled = True
        getattr(self, WEIGHTS_ATTR).requires_grad = True
        getattr(self, BIAS_ATTR).requires_grad = True

    def get_applied_weights(self) -> any:
        masked_weight = getattr(self, WEIGHTS_ATTR)
        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weight = masked_weight * mask_changes

        return masked_weight

    def init_parameters(self):
        init = configs_get_layers_initialization("fcn")
        init(getattr(self, WEIGHTS_ATTR))
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=get_flow_params_init(), b=get_flow_params_init() * 2.5)

        weights = getattr(self, WEIGHTS_ATTR)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input, inference = False):
        dense_flops = get_forward_flops_fcn_dense(self, input.shape)
        sparse_flops = get_forward_flops_fcn_sparse(self, input.shape)
        accumulate_flops(dense_flops, sparse_flops)

        masked_weight = getattr(self, WEIGHTS_ATTR)
        bias = torch.zeros(self.out_features, device=get_device())
        if hasattr(self, BIAS_ATTR):
            bias = getattr(self, BIAS_ATTR)

        # if self.mask_pruning_enabled:
        #     mask_changes = MaskPruningFunctionConstant.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
        #     masked_weight = masked_weight * mask_changes

        mask_changes = MaskPruningFunctionConstant.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
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
    def __init__(self, configs_conv2d: ConfigsLayerConv2, configs_network_masks: ConfigsNetworkMasksImportance):
        super(LayerConv2MaskImportance, self).__init__()
        # getting configs
        self.in_channels = configs_conv2d.in_channels
        self.out_channels = configs_conv2d.out_channels
        self.kernel_size = configs_conv2d.kernel_size
        self.padding = configs_conv2d.padding
        self.stride = configs_conv2d.stride
        self.bias_enabled = configs_conv2d.bias_enabled

        self.mask_pruning_enabled = configs_network_masks['mask_pruning_enabled']
        self.weights_training_enabled = configs_network_masks['weights_training_enabled']

        # defining parameters
        setattr(self, WEIGHTS_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        getattr(self, WEIGHTS_ATTR).requires_grad = self.weights_training_enabled

        setattr(self, WEIGHTS_PRUNING_ATTR, nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        getattr(self, WEIGHTS_PRUNING_ATTR).requires_grad = self.mask_pruning_enabled

        if self.bias_enabled:
            setattr(self, BIAS_ATTR, nn.Parameter(torch.Tensor(self.out_channels)))
            getattr(self, BIAS_ATTR).requires_grad = self.weights_training_enabled

        self.init_parameters()

    def get_bias_enabled(self):
        return self.bias_enabled

    def get_applied_weights(self) -> any:
        masked_weights = getattr(self, WEIGHTS_ATTR)
        if self.mask_pruning_enabled:
            mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
            masked_weights = masked_weights * mask_changes

        return masked_weights

    def init_parameters(self):
        init = configs_get_layers_initialization("conv2d")
        init(getattr(self, WEIGHTS_ATTR))
        nn.init.uniform_(getattr(self, WEIGHTS_PRUNING_ATTR), a=get_flow_params_init(), b=get_flow_params_init() * 2.5)

        if hasattr(self, BIAS_ATTR):
            weights = getattr(self, WEIGHTS_ATTR)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(getattr(self, BIAS_ATTR), -bound, bound)

    def forward(self, input):
        dense_flops = get_forward_flops_cnn_dense(self, input.shape)
        sparse_flops = get_forward_flops_cnn_sparse(self, input.shape)
        accumulate_flops(dense_flops, sparse_flops)

        masked_weights = getattr(self, WEIGHTS_ATTR)
        bias = torch.zeros(self.out_channels, device=get_device())
        if hasattr(self, BIAS_ATTR):
            bias = getattr(self, BIAS_ATTR)


        mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
        masked_weights = masked_weights * mask_changes

        # if self.mask_pruning_enabled:
        #     mask_changes = MaskPruningFunction.apply(getattr(self, WEIGHTS_PRUNING_ATTR))
        #     masked_weights = masked_weights * mask_changes

        return F.conv2d(input, masked_weights, bias, self.stride, self.padding)


def get_forward_flops_fcn_dense(self, input_shape: torch.Size) -> int:
    """
    Estimate the number of multiplications and additions for the forward pass
    in a fully-connected (dense) layer.

    Assumes input_shape is (batch_size, in_features).
    For F.linear (without bias), each output element requires:
      - in_features multiplications and
      - (in_features - 1) additions.

    Total FLOPs per output = 2 * in_features - 1
    Total FLOPs = batch_size * out_features * (2 * in_features - 1)
    """
    batch_size = input_shape[0]
    return batch_size * self.out_features * (2 * self.in_features - 1)


def get_forward_flops_cnn_dense(self, input_shape: torch.Size) -> int:
    """
    Estimate the number of multiplications and additions for the forward pass
    in a convolutional layer.

    Assumes input_shape is (batch_size, in_channels, height, width).
    For F.conv2d (without bias), each output element requires:
      - (in_channels * kernel_size^2) multiplications and
      - (in_channels * kernel_size^2 - 1) additions.

    Total FLOPs per output = 2 * (in_channels * kernel_size^2) - 1.
    Total FLOPs = batch_size * out_channels * out_height * out_width * (2 * (in_channels * kernel_size^2) - 1)
    """
    batch_size, _, in_height, in_width = input_shape
    out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width  = (in_width  + 2 * self.padding - self.kernel_size) // self.stride + 1
    kernel_ops = self.in_channels * self.kernel_size * self.kernel_size
    flops_per_instance = self.out_channels * out_height * out_width * (2 * kernel_ops - 1)
    return batch_size * flops_per_instance


def get_forward_flops_fcn_sparse(self, input_shape: torch.Size) -> int:
    """
    Estimate the number of multiplications and additions for the forward pass
    in a sparse fully-connected (dense) layer.

    Assumes input_shape is (batch_size, in_features).
    Dense FLOPs = batch_size * out_features * (2 * in_features - 1).
    Adjusted FLOPs = dense FLOPs * effective_density, where effective_density is
    the fraction of weights that are active (nonzero) as determined by the pruning mask.
    """
    batch_size = input_shape[0]
    dense_flops = batch_size * self.out_features * (2 * self.in_features - 1)
    if self.mask_pruning_enabled:
        # Get the pruning mask tensor (assumed to be stored in WEIGHTS_PRUNING_ATTR)
        mask_tensor = getattr(self, WEIGHTS_PRUNING_ATTR)
        mask_values = torch.sigmoid(mask_tensor)
        effective_mask = (mask_values > 0.5).float()
        density = effective_mask.mean().item()
        return int(dense_flops * density)
    else:
        return dense_flops


def get_forward_flops_cnn_sparse(self, input_shape: torch.Size) -> int:
    """
    Estimate the number of multiplications and additions for the forward pass
    in a sparse convolutional layer.

    Assumes input_shape is (batch_size, in_channels, height, width).
    Dense FLOPs = batch_size * out_channels * out_height * out_width * (2 * (in_channels * kernel_size^2) - 1).
    Adjusted FLOPs = dense FLOPs * effective_density, where effective_density is
    the fraction of active weights as determined by the pruning mask.
    """
    batch_size, _, in_height, in_width = input_shape
    out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width  = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
    kernel_ops = self.in_channels * self.kernel_size * self.kernel_size
    dense_flops_per_instance = self.out_channels * out_height * out_width * (2 * kernel_ops - 1)
    dense_flops = batch_size * dense_flops_per_instance
    if self.mask_pruning_enabled:
        mask_tensor = getattr(self, WEIGHTS_PRUNING_ATTR)
        mask_values = torch.sigmoid(mask_tensor)
        effective_mask = (mask_values > 0.5).float()
        density = effective_mask.mean().item()
        return int(dense_flops * density)
    else:
        return dense_flops


def count_pruned(model: LayerComposite):
    counter = 0
    for layer in model.get_layers_primitive():
        if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR):
            weights = getattr(layer, WEIGHTS_ATTR).data
            mask = getattr(layer, WEIGHTS_PRUNING_ATTR).data

            active_weights = weights[mask < 0]
            counter += active_weights.numel()
    
    print("COUNTER", counter)

def prune_model_globally(model: LayerComposite, pruning_rate: float = 0.1) -> float:
    """
    Prunes a specified percentage of the remaining weights in the model based on global magnitude.

    This function collects all weights that have not yet been pruned (where mask > 0),
    determines a global threshold for the lowest `pruning_rate` percentage of these weights,
    and then sets their corresponding mask values to -1.0.

    Args:
        model (LayerComposite): The model to be pruned. The model must have a
                                `get_layers_primitive` method.
        pruning_rate (float): The fraction (e.g., 0.1 for 10%) of the *remaining*
                              weights to prune. Must be in the range [0.0, 1.0].

    Returns:
        float: The magnitude threshold used for pruning. This is the largest magnitude
               of a weight that was pruned in this step.

    Raises:
        ValueError: If `pruning_rate` is not between 0.0 and 1.0.
        ValueError: If the model contains no layers with prunable weights.
        ValueError: If pruning is requested (rate > 0) but no unpruned weights remain.
        AttributeError: If `model` does not have a `get_layers_primitive` method.
    """
    print("Prune model globally!!!")

    # --- Step 1: Input and State Validation ---
    if not 0.0 <= pruning_rate <= 1.0:
        raise ValueError(f"pruning_rate must be between 0.0 and 1.0, but got {pruning_rate}")

    if not hasattr(model, 'get_layers_primitive'):
        raise AttributeError("The provided model object must have a `get_layers_primitive` method.")

    # If the rate is 0, no action is needed. Return 0.0 threshold.
    if pruning_rate == 0.0:
        return 0.0
    
    # count_pruned(model)

    # --- Step 2: Collect all unpruned weight magnitudes ---
    unpruned_weights = []
    found_prunable_layer = False
    for layer in model.get_layers_primitive():
        if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR):
            found_prunable_layer = True
            weights = getattr(layer, WEIGHTS_ATTR).data
            mask = getattr(layer, WEIGHTS_PRUNING_ATTR).data

            # Get weights that have not been pruned yet (where the mask is positive)
            # print(mask.shape)
            # print(mask.flatten()[:10])
            active_weights = weights[mask >= 0]
            # print("Active numel", active_weights.numel())
            if active_weights.numel() > 0:
                unpruned_weights.append(torch.abs(active_weights.flatten()))
        else: 
            raise("Has no WEIGHTS_PRUNING_ATTR or WEIGHTS_ATTR????")

    if not found_prunable_layer:
        raise ValueError("The provided model contains no prunable layers (i.e., no layers with both weight and weight_mask attributes).")

    if not unpruned_weights:
        # Prunable layers were found, but all their weights have already been pruned.
        raise ValueError("No unpruned weights remaining to be pruned, but a pruning_rate > 0 was requested.")

    # Concatenate all active weights into a single tensor
    all_active_weights = torch.cat(unpruned_weights)

    # --- Step 3: Determine the pruning threshold ---
    num_to_prune = int(all_active_weights.numel() * pruning_rate)
    
    # If the rate > 0 and there are weights, ensure at least one weight is pruned.
    if num_to_prune == 0:
        num_to_prune = 1
    
    # The k for kthvalue is 1-indexed and cannot be larger than the number of elements.
    k = min(num_to_prune, all_active_weights.numel())
    threshold = torch.kthvalue(all_active_weights, k).values.item()
    # print("Found threshold", threshold)

    # --- Step 4: Apply the new mask globally ---
    with torch.no_grad():
        for layer in model.get_layers_primitive():
            if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, WEIGHTS_PRUNING_ATTR):
                weights = getattr(layer, WEIGHTS_ATTR).data
                mask = getattr(layer, WEIGHTS_PRUNING_ATTR).data

                # Identify weights to be pruned: active (mask > 0) and magnitude <= threshold
                to_prune_indices = (mask >= 0) & (torch.abs(weights) <= threshold)

                # Update the mask to a non-positive value for pruned weights
                mask[to_prune_indices] = -1.0

    return threshold

def calculate_pruning_epochs(
    target_sparsity: float,
    pruning_rate: float,
    total_epochs: int,
    start_epoch: int = 0,
    end_epoch: int = -1
) -> list[int]:
    """
    Calculates the specific epochs at which to prune weights.

    This function determines when to prune a fixed percentage of the *remaining*
    unpruned weights to reach a target sparsity level over a specified
    number of training epochs.

    Args:
        target_sparsity: The desired final fraction of weights to be pruned.
                         (e.g., 0.9 for 90% sparsity). Must be between 0 and 1.
        pruning_rate: The fraction of *remaining* weights to prune at each step.
                      (e.g., 0.1 for 10%). Must be between 0 and 1.
        total_epochs: The total number of epochs for the training process.
        start_epoch: The first epoch where pruning can occur. Defaults to 0.
        end_epoch: The last epoch where pruning can occur.
                   Defaults to -1, which is interpreted as the last epoch.

    Returns:
        A list of integers representing the exact epochs to perform pruning.
        Returns an empty list if the inputs are invalid.
    """
    # --- Input Validation ---
    if not (0 < target_sparsity < 1):
        raise("Error: target_sparsity must be between 0 and 1.")
    if not (0 < pruning_rate < 1):
        raise("Error: pruning_rate must be between 0 and 1.")
    if total_epochs <= 0:
        raise("Error: total_epochs must be a positive integer.")

    # If end_epoch is the default -1, set it to the last epoch
    if end_epoch == -1:
        end_epoch = total_epochs - 1

    if start_epoch < 0 or end_epoch >= total_epochs or start_epoch > end_epoch:
        raise("Error: Invalid start_epoch or end_epoch.")

    # --- Calculation ---

    # The fraction of weights we want to remain after all pruning is done.
    # If target sparsity is 90% (0.9), remaining_weights_target is 10% (0.1).
    remaining_weights_target = 1.0 - target_sparsity

    # We need to find n, the number of pruning steps, such that:
    # (1 - pruning_rate)^n <= remaining_weights_target
    #
    # Taking the logarithm of both sides:
    # n * log(1 - pruning_rate) <= log(remaining_weights_target)
    #
    # Since log(1 - pruning_rate) is negative, we flip the inequality when dividing:
    # n >= log(remaining_weights_target) / log(1 - pruning_rate)
    #
    # We use math.ceil to ensure we perform enough steps to meet or exceed the target.
    try:
        num_pruning_steps = math.ceil(
            math.log(remaining_weights_target) / math.log(1.0 - pruning_rate)
        )
    except ValueError:
        raise("Error: Could not calculate the number of pruning steps. Check input values.")
        
    if num_pruning_steps <= 0:
        raise ("num_pruning_steps <= 0")

    # --- Epoch Distribution ---

    # Determine the span of epochs available for pruning.
    pruning_epoch_span = end_epoch - start_epoch
    
    # If there are more pruning steps than available epochs, we can't schedule them.
    # We return one pruning step per available epoch up to the required number.
    if num_pruning_steps > pruning_epoch_span + 1:
        print("Warning: More pruning steps required than available epochs in the given range.")
        print(f"Returning a schedule with one pruning event per epoch from {start_epoch} to {end_epoch}.")
        return list(range(start_epoch, end_epoch + 1))


    # Distribute the pruning steps evenly across the available epochs.
    # np.linspace creates an array of evenly spaced numbers over a specified interval.
    # We then round them to the nearest integer to get concrete epoch numbers.
    pruning_epochs_float = np.linspace(start_epoch, end_epoch, num_pruning_steps)
    pruning_epochs = [int(round(epoch)) for epoch in pruning_epochs_float]

    return sorted(list(set(pruning_epochs))) # Retu