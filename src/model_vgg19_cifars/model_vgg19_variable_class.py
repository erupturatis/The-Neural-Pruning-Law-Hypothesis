import torch
import torch.nn as nn

from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER, BATCH_NORM_2D_LAYER
from src.infrastructure.layers import (
    ConfigsNetworkMask, LayerConv2MaskImportance, ConfigsLayerConv2,
    LayerLinearMaskImportance, ConfigsLayerLinear, ModelCustom, get_flow_params_loss,
)
from src.model_vgg19_cifars.model_attributes import get_vgg19_variable_cifar_attributes
from src.model_vgg19_cifars.model_functions import forward_pass_vgg19_cifar


class ModelVGG19Variable(ModelCustom):
    def __init__(self, alpha: float, config_network_mask: ConfigsNetworkMask, num_classes: int = 10):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.registered_layers = []
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        reg_attrs, unreg_attrs = get_vgg19_variable_cifar_attributes(alpha, num_classes)
        self._registered_layer_attributes = reg_attrs
        self._unregistered_layer_attributes = unreg_attrs

        for attr in reg_attrs:
            name = attr['name']
            t = attr['type']
            if t == CONV2D_LAYER:
                layer = LayerConv2MaskImportance(
                    ConfigsLayerConv2(
                        in_channels=attr['in_channels'],
                        out_channels=attr['out_channels'],
                        kernel_size=attr['kernel_size'],
                        stride=attr['stride'],
                        padding=attr['padding'],
                        bias_enabled=attr['bias_enabled'],
                    ),
                    config_network_mask,
                )
            elif t == FULLY_CONNECTED_LAYER:
                layer = LayerLinearMaskImportance(
                    ConfigsLayerLinear(
                        in_features=attr['in_features'],
                        out_features=attr['out_features'],
                        bias_enabled=attr['bias_enabled'],
                    ),
                    config_network_mask,
                )
            else:
                raise ValueError(f"Unsupported registered layer type: {t}")
            setattr(self, name, layer)
            self.registered_layers.append(layer)

        for attr in unreg_attrs:
            name = attr['name']
            t = attr['type']
            if t == BATCH_NORM_2D_LAYER:
                layer = nn.BatchNorm2d(attr['num_features'])
            else:
                raise ValueError(f"Unsupported unregistered layer type: {t}")
            setattr(self, name, layer)

    def get_hyperflux_loss(self) -> torch.Tensor:
        total, remaining = get_flow_params_loss(self)
        return remaining / total

    def forward(self, x):
        return forward_pass_vgg19_cifar(self, x)
