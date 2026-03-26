from typing import List, Dict
import os
from torch import nn
import torch
from src.infrastructure.layers import LayerPrimitive
from src.infrastructure.others import prefix_path_with_root

def save_model_entire_dict(model: 'LayerComposite', model_name: str, folder_name:str):
    filepath = folder_name + "/" + model_name
    filepath = prefix_path_with_root(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    
def load_model_entire_dict(model: 'LayerComposite', model_name: str, folder_name:str):
    filepath = folder_name + "/" + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    return model