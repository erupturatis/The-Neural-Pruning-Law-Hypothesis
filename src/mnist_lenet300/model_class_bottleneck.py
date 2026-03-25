import torch
from typing import List
from src.common_files_experiments.load_save import save_model_weights, load_model_weights
from src.infrastructure.layers import ConfigsNetworkMasksImportance, LayerLinearMaskImportance, ConfigsLayerLinear, \
    get_flow_params_loss, get_layer_composite_flow_params_statistics, \
    LayerPrimitive, LayerComposite, get_layers_primitive
from src.infrastructure.constants import FULLY_CONNECTED_LAYER, N_SCALER
from src.mnist_lenet300.model_attributes import LENET300_MNIST_REGISTERED_LAYERS_ATTRIBUTES, \
    LENET300_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING, LENET300_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.mnist_lenet300.model_functions import forward_pass_lenet300

class ModelLenet300Bottleneck(LayerComposite):
    def __init__(self, config_network_mask: ConfigsNetworkMasksImportance):
        super(ModelLenet300Bottleneck, self).__init__()
        self.registered_layers = []

        for layer_attr in LENET300_MNIST_REGISTERED_LAYERS_ATTRIBUTES:
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
                raise ValueError(f"Unsupported registered layer type: {type_}")

            setattr(self, name, layer)
            if name == "fc3":
                self.registered_layers.append(layer)

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid = get_flow_params_loss(self)
        # act / total params, bottleneck reduces total but we need to use the same divider
        return sigmoid / 266200

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    def get_parameters_pruning_statistics(self) -> any:
        return get_layer_composite_flow_params_statistics(self)

    def forward(self, x, inference=False):
        return forward_pass_lenet300(self, x, inference)

    def save(self, name: str, folder: str):
        save_model_weights(
            model=self,
            model_name=name,
            folder_name=folder,
            custom_to_standard_mapping=LENET300_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
        )

    def load(self, name: str, folder:str):
        load_model_weights(
            model=self,
            model_name=name,
            folder_name=folder,
            standard_to_custom_mapping=LENET300_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
        )
