import torch
from src.infrastructure.layers import (
    ConfigsNetworkMask, LayerLinearMaskImportance, ConfigsLayerLinear,
    get_flow_params_loss, LayerPrimitive, LayerComposite, get_layers_primitive,
)
from src.infrastructure.constants import FULLY_CONNECTED_LAYER
from src.model_lenet.model_attributes import get_lenet_variable_attributes
from src.model_lenet.model_functions import forward_pass_lenet

class ModelLenetVariable(LayerComposite):
    def __init__(self, alpha: float, config_network_mask: ConfigsNetworkMask):
        super(ModelLenetVariable, self).__init__()
        self._layer_attributes = get_lenet_variable_attributes(alpha)
        self.registered_layers = []

        for layer_attr in self._layer_attributes:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == FULLY_CONNECTED_LAYER:
                layer = LayerLinearMaskImportance(
                    configs_linear=ConfigsLayerLinear(
                        in_features=layer_attr['in_features'],
                        out_features=layer_attr['out_features'],
                        bias_enabled=layer_attr['bias_enabled'],
                    ),
                    configs_network=config_network_mask,
                )
            else:
                raise ValueError(f"Unsupported layer type: {type_}")

            setattr(self, name, layer)
            self.registered_layers.append(layer)

    def get_hyperflux_loss(self) -> torch.Tensor:
        total, sigmoid = get_flow_params_loss(self)
        return sigmoid / total

    def forward(self, x):
        return forward_pass_lenet(self, x)
