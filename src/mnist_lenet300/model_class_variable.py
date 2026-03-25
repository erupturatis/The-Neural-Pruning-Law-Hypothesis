import torch
from typing import List

from src.common_files_experiments.load_save import save_model_weights, load_model_weights
from src.infrastructure.layers import (
    ConfigsNetworkMasksImportance, LayerLinearMaskImportance, ConfigsLayerLinear,
    get_flow_params_loss, get_layer_composite_flow_params_statistics,
    LayerPrimitive, LayerComposite, get_layers_primitive,
)
from src.mnist_lenet300.model_functions import forward_pass_lenet300

_LAYER_NAME_MAPPING = [
    {'custom_name': 'fc1', 'standard_name': 'fc1.weight'},
    {'custom_name': 'fc2', 'standard_name': 'fc2.weight'},
    {'custom_name': 'fc3', 'standard_name': 'fc3.weight'},
]

_LAYER_NAME_MAPPING_REVERSE = [
    {'standard_name': 'fc1.weight', 'custom_name': 'fc1'},
    {'standard_name': 'fc2.weight', 'custom_name': 'fc2'},
    {'standard_name': 'fc3.weight', 'custom_name': 'fc3'},
]


class ModelLenetVariable(LayerComposite):
    """Variable-width LeNet: 784 → hidden1 → hidden2 → 10 (MNIST)."""

    def __init__(self, hidden1: int, hidden2: int, config_network_mask: ConfigsNetworkMasksImportance):
        super(ModelLenetVariable, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.registered_layers = []

        for name, in_f, out_f in [
            ('fc1', 784, hidden1),
            ('fc2', hidden1, hidden2),
            ('fc3', hidden2, 10),
        ]:
            layer = LayerLinearMaskImportance(
                configs_linear=ConfigsLayerLinear(in_features=in_f, out_features=out_f, bias_enabled=True),
                configs_network=config_network_mask,
            )
            setattr(self, name, layer)
            self.registered_layers.append(layer)

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid = get_flow_params_loss(self)
        return sigmoid / total

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self):
        return get_layer_composite_flow_params_statistics(self)

    def forward(self, x, inference=False):
        return forward_pass_lenet300(self, x, inference)

    def save(self, name: str, folder: str):
        save_model_weights(
            model=self,
            model_name=name,
            folder_name=folder,
            custom_to_standard_mapping=_LAYER_NAME_MAPPING,
        )

    def load(self, name: str, folder: str):
        load_model_weights(
            model=self,
            model_name=name,
            folder_name=folder,
            standard_to_custom_mapping=_LAYER_NAME_MAPPING_REVERSE,
        )
