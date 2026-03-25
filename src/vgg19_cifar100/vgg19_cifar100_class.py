from typing import List
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.common_files_experiments.forward_functions import forward_pass_vgg19_cifars_version2
from src.common_files_experiments.load_save import save_model_weights, load_model_weights
from src.vgg19_cifar100.vgg19_cifar100_attributes import (
    VGG19_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES,
    VGG19_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES,
    VGG19_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    VGG19_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
)
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER, N_SCALER, PRUNED_MODELS_PATH
from src.infrastructure.layers import (
    LayerConv2MaskImportance,
    ConfigsNetworkMasksImportance,
    LayerLinearMaskImportance,
    LayerComposite,
    LayerPrimitive,
    get_layers_primitive,
    get_flow_params_loss,
    get_layer_composite_flow_params_statistics,
    ConfigsLayerConv2,
    ConfigsLayerLinear,
    get_parameters_total_count
)


class VGG19Cifar100(LayerComposite):
    def __init__(self, configs_network_masks: ConfigsNetworkMasksImportance):
        super(VGG19Cifar100, self).__init__()
        self.registered_layers = []

        # Define ReLU and MaxPool layers
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Initialize registered layers
        for layer_attr in VGG19_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            # print(name)
            type_ = layer_attr['type']

            if type_ == CONV2D_LAYER:
                layer = LayerConv2MaskImportance(
                    ConfigsLayerConv2(
                        in_channels=layer_attr['in_channels'],
                        out_channels=layer_attr['out_channels'],
                        kernel_size=layer_attr['kernel_size'],
                        stride=layer_attr['stride'],
                        padding=layer_attr['padding'],
                        bias_enabled=layer_attr['bias_enabled']
                    ),
                    configs_network_masks
                )
            elif type_ == FULLY_CONNECTED_LAYER:
                layer = LayerLinearMaskImportance(
                    ConfigsLayerLinear(
                        in_features=layer_attr['in_features'],
                        out_features=layer_attr['out_features']
                    ),
                    configs_network_masks
                )
            else:
                raise ValueError(f"Unsupported registered layer type: {type_}")

            setattr(self, name, layer)
            self.registered_layers.append(layer)

        # Initialize unregistered layers (empty for VGG19)
        for layer_attr in VGG19_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == 'BatchNorm2d':
                layer = nn.BatchNorm2d(
                    num_features=layer_attr['num_features']
                )
            else:
                raise ValueError(f"Unsupported unregistered layer type: {type_}")

            setattr(self, name, layer)

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, remaining = get_flow_params_loss(self)
        return remaining / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_flow_params_statistics(self)

    def get_parameters_total_count(self) -> int:
        total = get_parameters_total_count(self)
        return total

    def forward(self, x):
        return forward_pass_vgg19_cifars_version2(self, x, VGG19_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES, VGG19_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES)

    def save(self, name: str, folder: str):
        save_model_weights(
            model=self,
            model_name=name,
            folder_name=folder,
            custom_to_standard_mapping=VGG19_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
            skip_array=[]
        )

    def load(self, path: str, folder: str):
        load_model_weights(
            model=self,
            model_name=path,
            folder_name=folder,
            standard_to_custom_mapping=VGG19_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
            skip_array=[]
        )
