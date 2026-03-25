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