from typing import List, Dict
from torch import nn
import torch
from src.infrastructure.layers import LayerPrimitive
from src.infrastructure.others import prefix_path_with_root

def save_model_entire_dict(model: 'LayerComposite', model_name: str, folder_name:str):
    filepath = folder_name + "/" + model_name
    filepath = prefix_path_with_root(filepath)
        
    torch.save(model.state_dict(), filepath)
    

def load_model_entire_dict(model: 'LayerComposite', model_name: str, folder_name:str):
    filepath = folder_name + "/" + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    return model

def save_model_weights(model: 'LayerComposite', model_name: str, folder_name:str, custom_to_standard_mapping: Dict, skip_array: List = []):
    filepath = folder_name + "/" + model_name
    filepath = prefix_path_with_root(filepath)
    _save_model_weights(
        model=model,
        filepath=filepath,
        network_to_standard_mapping=custom_to_standard_mapping,
        skip_array=skip_array
    )


def _save_model_weights(model: 'LayerComposite', filepath: str, network_to_standard_mapping: Dict, skip_array: List = []):
    state_dict = {}

    for mapping in network_to_standard_mapping:
        custom_name = mapping['custom_name']
        standard_name = mapping['standard_name']
        if custom_name in skip_array:
            continue

        layer = getattr(model, custom_name, None)
        if layer is None:
            print(f"WARNING: Layer '{custom_name}' not found in the model.")
            continue

        if isinstance(layer, nn.BatchNorm2d):
            state_dict[f'{standard_name}.weight'] = layer.weight.data.clone()
            state_dict[f'{standard_name}.bias'] = layer.bias.data.clone()
            state_dict[f'{standard_name}.running_mean'] = layer.running_mean.data.clone()
            state_dict[f'{standard_name}.running_var'] = layer.running_var.data.clone()
            state_dict[f'{standard_name}.num_batches_tracked'] = layer.num_batches_tracked.clone()

        # Handle Conv2d and Linear layers
        elif isinstance(layer, LayerPrimitive):
            state_dict[standard_name] = layer.get_applied_weights().data.clone()
            if layer.get_bias_enabled():
                bias_name = standard_name.replace('.weight', '.bias')
                state_dict[bias_name] = layer.bias.data.clone()

        else:
            print(f"Unhandled layer type for layer '{custom_name}': {type(layer)}")

    torch.save(state_dict, filepath)
    print(f"Model weights saved to {filepath}.")

def _load_model_weights(model: 'LayerComposite', model_dict, standard_to_network_dict: Dict, skip_array: List = []):
    state_dict = model_dict

    for mapping in standard_to_network_dict:
        standard_name = mapping['standard_name']
        custom_name = mapping['custom_name']
        if custom_name in skip_array:
            continue

        layer = getattr(model, custom_name, None)
        if layer is None:
            print(f"Layer '{custom_name}' not found in the model.")
            continue

        # Handle BatchNorm layers
        if isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.copy_(state_dict[f'{standard_name}.weight'])
            layer.bias.data.copy_(state_dict[f'{standard_name}.bias'])
            layer.running_mean.data.copy_(state_dict[f'{standard_name}.running_mean'])
            layer.running_var.data.copy_(state_dict[f'{standard_name}.running_var'])
            layer.num_batches_tracked.copy_(state_dict[f'{standard_name}.num_batches_tracked'])

        # Handle Conv2d and Linear layers
        elif isinstance(layer, LayerPrimitive):
            layer.weights.data.copy_(state_dict[standard_name])
            if layer.get_bias_enabled():
                bias_name = standard_name.replace('.weight', '.bias')
                layer.bias.data.copy_(state_dict[bias_name])

        else:
            print(f"Unhandled layer type for layer '{custom_name}': {type(layer)}")

def load_model_weights(model: 'LayerComposite', model_name: str, folder_name:str, standard_to_custom_mapping: Dict, skip_array: List = []):
    filepath = folder_name + "/" + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)
    _load_model_weights(model, state_dict, standard_to_custom_mapping, skip_array)

