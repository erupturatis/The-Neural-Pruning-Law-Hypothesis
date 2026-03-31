import torch
import torch.nn as nn

from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER, BATCH_NORM_2D_LAYER
from src.infrastructure.layers import (
    ConfigsNetworkMask, LayerConv2MaskImportance, ConfigsLayerConv2,
    LayerLinearMaskImportance, ConfigsLayerLinear, ModelCustom, get_flow_params_loss,
)
from src.model_resnet50_cifars.model_attributes import get_resnet50_variable_cifar10_attributes
from src.model_resnet50_cifars.model_functions import forward_pass_resnet50_cifar10


class ModelResnet50Variable(ModelCustom):
    def __init__(self, alpha: float, config_network_mask: ConfigsNetworkMask, num_classes: int = 10):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.registered_layers = []
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        reg_attrs, unreg_attrs = get_resnet50_variable_cifar10_attributes(alpha, num_classes)
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
        total, sigmoid = get_flow_params_loss(self)
        return sigmoid / total

    def forward(self, x):
        return forward_pass_resnet50_cifar10(self, x)
