from typing import List
import torch
import torch.nn as nn
from src.common_files_experiments.forward_functions import forward_pass_resnet50_cifars
from src.common_files_experiments.load_save import save_model_weights, load_model_weights
from src.infrastructure.parameters_mask_processors import get_weights_params_decay_all_
from src.resnet50_cifar100.resnet50_cifar100_attributes import RESNET50_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES, \
    RESNET50_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES, RESNET50_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING, \
    RESNET50_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.infrastructure.constants import N_SCALER, PRUNED_MODELS_PATH, CONV2D_LAYER, FULLY_CONNECTED_LAYER, \
    BATCH_NORM_2D_LAYER
from src.infrastructure.layers import LayerComposite, ConfigsNetworkMasksImportance, LayerConv2MaskImportance, \
    ConfigsLayerConv2, LayerLinearMaskImportance, ConfigsLayerLinear, LayerPrimitive, get_layers_primitive, \
    get_flow_params_loss, get_layer_composite_flow_params_statistics, \
    get_weight_decay_only_for_present, get_weight_decay_only_for_all


class Resnet50Cifar100(LayerComposite):
    def __init__(self, configs_network_masks: ConfigsNetworkMasksImportance):
        super(Resnet50Cifar100, self).__init__()
        self.registered_layers = []

        # Hardcoded activations
        self.relu = nn.ReLU(inplace=True)

        # Initialize registered layers
        for layer_attr in RESNET50_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
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

        # Initialize unregistered layers
        for layer_attr in RESNET50_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES:
            name = layer_attr['name']
            type_ = layer_attr['type']

            if type_ == BATCH_NORM_2D_LAYER:
                layer = nn.BatchNorm2d(
                    num_features=layer_attr['num_features']
                )
            elif type_ == 'MaxPool2d':
                layer = nn.MaxPool2d(
                    kernel_size=layer_attr['kernel_size'],
                    stride=layer_attr['stride'],
                    padding=layer_attr['padding']
                )
            elif type_ == 'AdaptiveAvgPool2d':
                layer = nn.AdaptiveAvgPool2d(
                    output_size=layer_attr['output_size']
                )
            else:
                raise ValueError(f"Unsupported unregistered layer type: {type_}")

            setattr(self, name, layer)

        # Initialize additional layers if any
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def get_weight_decay_all(self) -> torch.Tensor:
        weights = get_weight_decay_only_for_all(self)
        weight_decay = 5e-4
        return weights * weight_decay / 2

    def get_weight_decay_only_present(self) -> torch.Tensor:
        weights = get_weight_decay_only_for_present(self)
        weight_decay = 5e-4
        return weights * weight_decay / 2

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, flow_params_values = get_flow_params_loss(self)
        return flow_params_values / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_flow_params_statistics(self)

    def forward(self, x):
        return forward_pass_resnet50_cifars(
            self=self,
            x=x,
            registered_layer_attributes=RESNET50_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES,
            unregistered_layer_attributes=RESNET50_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES
        )

    def save(self, name: str, folder: str):
        save_model_weights(
            model=self,
            model_name=name,
            folder_name=folder,
            custom_to_standard_mapping=RESNET50_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
            skip_array=[]
        )

    def load(self, path: str, folder):
        load_model_weights(
            model=self,
            model_name=path,
            folder_name=folder,
            standard_to_custom_mapping=RESNET50_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
            skip_array=[]
        )

