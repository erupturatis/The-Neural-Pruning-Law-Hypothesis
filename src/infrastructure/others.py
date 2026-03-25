from calendar import month
from typing import List, Dict
import torch
import os
import uuid
from src.infrastructure.constants import EXPERIMENTS_RESULTS_PATH

def get_random_id():
    random_id = str(uuid.uuid4())
    return random_id

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_root_folder() -> str:
    """Return the absolute path to the project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # Root has the same parent directory
        if '.root' in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir

def prefix_path_with_root(path):
    root_path = get_root_folder()
    return root_path + "/" + path

def round_float(value: float, digits: int = 4) -> float:
    return round(value, digits)

import json
def save_array_experiment(filename, arr):
    path = prefix_path_with_root(EXPERIMENTS_RESULTS_PATH)
    path = path + "/" + filename
    with open(path, 'w') as file:
        json.dump(arr, file)


def get_custom_model_sparsity_percent(model) -> float:
    total, remaining = model.get_parameters_pruning_statistics()
    percent = remaining / total * 100
    return round_float(percent.item())

from torch import nn

def get_raw_model_sparsity_percent(model: nn.Module, standard_to_custom_attributes: Dict):
    total_weights = 0
    nonzero_weights = 0

    state_dict = model.state_dict()
    for layer_info in standard_to_custom_attributes:
        layer_name = layer_info['standard_name']

        if layer_name in state_dict:
            weights = state_dict[layer_name]
            total_weights += weights.numel()
            nonzero_weights += torch.count_nonzero(weights).item()
        else:
            print(f"Warning: {layer_name} not found in the model's state_dict")

    if total_weights == 0:
        sparsity = 0.0
    else:
        sparsity = (nonzero_weights / total_weights) * 100

    print(f"Total weights: {total_weights}")
    print(f"Non-zero weights: {nonzero_weights}")
    print(f"Sparsity (% of non-zero weights): {sparsity:.2f}%")

    return sparsity

from typing import TypedDict

class TrainingConfigsWithResume(TypedDict):
    pruning_end: int
    regrowing_end: int
    target_sparsity: float
    lr_flow_params_decay_regrowing: float
    start_lr_pruning: float
    end_lr_pruning: float
    reset_lr_pruning: float
    end_lr_regrowth: float
    reset_lr_flow_params_scaler: float
    weight_decay: float
    resume: str
    notes: str


class TrainingConfigsNPLHIMP(TypedDict):
    training_end: int
    target_sparsity: float
    start_lr_pruning: float
    end_lr_pruning: float
    weight_decay: float
    resume: str
    notes: str

class TrainingConfigsNPLHL0(TypedDict):
    pruning_end: int
    exponent_start: float
    exponent_end: float
    base: float
    epochs_raise: int
    learning_rate: float
    weight_decay: float
    resume: str
    notes: str