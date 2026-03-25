import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def bad_initialization(weights: torch.Tensor):
    nn.init.uniform_(weights, a=-1, b=1)

def kaiming_sqrt0(weights: torch.Tensor):
    nn.init.kaiming_uniform_(weights, a=math.sqrt(0))

def kaiming_sqrt5(weights: torch.Tensor):
    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))

def kaiming_relu(weights: torch.Tensor):
    nn.init.kaiming_uniform_(weights, a=0, nonlinearity='relu')
