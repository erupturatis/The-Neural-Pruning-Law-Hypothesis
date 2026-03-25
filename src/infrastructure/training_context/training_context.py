from enum import Enum
from typing import Tuple
import kornia.augmentation as K
from abc import ABC, abstractmethod
import torch
from src.infrastructure.others import get_device
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TrainingContextBaselineTrainArgs:
    optimizer_weights: torch.optim.Optimizer

class TrainingContextBaselineTrain:
    def __init__(self, args: TrainingContextBaselineTrainArgs):
        self.params = args

    def get_optimizer_weights(self) -> torch.optim.Optimizer:
        return self.params.optimizer_weights


@dataclass
class TrainingContextPrunedTrainArgs:
    lr_weights_reset: float
    lr_flow_params_reset: float

    l0_gamma_scaler: float

    optimizer_weights: torch.optim.Optimizer
    optimizer_flow_mask: torch.optim.Optimizer

class TrainingContextPrunedTrain:
    def __init__(self, params: TrainingContextPrunedTrainArgs):
        self.params = params

    def get_optimizer_weights(self) -> torch.optim.Optimizer:
        return self.params.optimizer_weights

    def get_optimizer_flow_mask(self) -> torch.optim.Optimizer:
        return self.params.optimizer_flow_mask

    def set_gamma(self, gamma: float) -> None:
        self.params.l0_gamma_scaler = gamma

    def reset_param_groups_to_defaults(self) -> None:
        for param_group in self.params.optimizer_weights.param_groups:
            param_group['lr'] = self.params.lr_weights_reset

        for param_group in self.params.optimizer_flow_mask.param_groups:
            param_group['lr'] = self.params.lr_flow_params_reset

        self.params.l0_gamma_scaler = 0


@dataclass
class TrainingContextNPLHL0Args:
    optimizer_weights: torch.optim.Optimizer
    optimizer_flow_mask: torch.optim.Optimizer
    l0_gamma_scaler: float

class TrainingContextNPLHL0:
    def __init__(self, params: TrainingContextNPLHL0Args):
        self.params = params

    def get_optimizer_weights(self) -> torch.optim.Optimizer:
        return self.params.optimizer_weights

    def get_optimizer_flow_mask(self) -> torch.optim.Optimizer:
        return self.params.optimizer_flow_mask
    
    def set_gamma(self, gamma: float) -> None:
        self.params.l0_gamma_scaler = gamma


@dataclass
class TrainingContextPrunedBottleneckTrainArgs:
    l0_gamma_scaler: float
    optimizer_weights: torch.optim.Optimizer
    optimizer_flow_mask: torch.optim.Optimizer

class TrainingContextPrunedBottleneckTrain:
    def __init__(self, params: TrainingContextPrunedBottleneckTrainArgs):

        self.params = params

    def get_optimizer_weights(self) -> torch.optim.Optimizer:
        return self.params.optimizer_weights

    def get_optimizer_flow_mask(self) -> torch.optim.Optimizer:
        return self.params.optimizer_flow_mask

    def set_gamma(self, gamma: float) -> None:
        self.params.l0_gamma_scaler = gamma
