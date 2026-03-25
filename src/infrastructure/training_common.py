from typing import TYPE_CHECKING
import torch

from src.infrastructure.constants import WEIGHTS_ATTR, BIAS_ATTR, WEIGHTS_PRUNING_ATTR
if TYPE_CHECKING:
    from src.infrastructure.layers import LayerComposite


def get_model_weights_params(model: 'LayerComposite') -> list[torch.Tensor]:
    weight_bias_params = []
    for name, param in model.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)

    return weight_bias_params

def get_model_flow_params_and_weights_params(model: 'LayerComposite') -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    weight_bias_params = []
    pruning_params = []

    for name, param in model.named_parameters():
        if WEIGHTS_ATTR in name or BIAS_ATTR in name:
            weight_bias_params.append(param)
        if WEIGHTS_PRUNING_ATTR in name:
            pruning_params.append(param)

    return weight_bias_params, pruning_params

def get_model_flow_params_and_weights_params_bn_separate(model: "LayerComposite") -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Splits model parameters into four groups:
      1) weight_params:    all conv/linear weights (for decay)
      2) pruning_params:   parameters used for pruning masks
      3) no_decay_params:  all biases and all normalization params (no weight decay)
    """
    weight_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []
    pruning_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if WEIGHTS_PRUNING_ATTR in name:
            pruning_params.append(param)
        if BIAS_ATTR in name or 'bn' in name.lower() or 'norm' in name.lower():
            no_decay_params.append(param)
        elif WEIGHTS_ATTR in name:
            weight_params.append(param)

    return weight_params, pruning_params, no_decay_params

