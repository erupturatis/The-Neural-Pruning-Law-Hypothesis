from types import SimpleNamespace
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from src.infrastructure.constants import PRUNED_MODELS_PATH
from src.infrastructure.layers import LayerComposite, LayerPrimitive
from typing import List
from src.infrastructure.others import prefix_path_with_root
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from fontTools.config import Config

from src.mnist_lenet300.model_attributes import LENET300_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING, \
    LENET300_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING

def forward_pass_lenet300(self: 'LayerComposite', x: torch.Tensor, inference=False) -> torch.Tensor:
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x, inference=inference))
    x = F.relu(self.fc2(x, inference=inference))
    x = self.fc3(x, inference=inference)
    return x

