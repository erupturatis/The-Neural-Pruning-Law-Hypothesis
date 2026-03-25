from src.infrastructure.layer_initializations import kaiming_sqrt5, kaiming_relu, kaiming_sqrt0, bad_initialization
from src.infrastructure.mask_functions import MaskPruningFunctionConstant
from src.infrastructure.parameters_mask_processors import get_flow_params_statistics_raw_, \
    get_flow_params_loss_raw_
from typing import Dict

MaskPruningFunction = MaskPruningFunctionConstant

get_flow_params_loss_abstract = get_flow_params_loss_raw_
get_flow_params_statistics_abstract = get_flow_params_statistics_raw_

_configs_layers_init = {
    "fcn": kaiming_sqrt5,
    "conv2d": kaiming_sqrt5
}

def configs_get_layers_all_initialization(layer_name: str) -> Dict:
    return _configs_layers_init

def configs_get_layers_initialization(layer_name) -> callable:
    return _configs_layers_init[layer_name]

def configs_layers_initialization_all_kaiming_relu():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": kaiming_relu,
        "conv2d": kaiming_relu
    }

def configs_layers_initialization_all_kaiming_sqrt5():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": kaiming_sqrt5,
        "conv2d": kaiming_sqrt5
    }

def configs_layers_initialization_all_kaiming_sqrt0():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": kaiming_sqrt0,
        "conv2d": kaiming_sqrt0
    }

def configs_layers_initialization_all_bad():
    global _configs_layers_init
    _configs_layers_init = {
        "fcn": bad_initialization,
        "conv2d": bad_initialization
    }