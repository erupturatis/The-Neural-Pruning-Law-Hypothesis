from enum import Enum
import warnings
from typing import Tuple
import kornia.augmentation as K
from abc import ABC, abstractmethod
import torch
from src.infrastructure.configs_general import VERBOSE_STAGES, EXTRA_VERBOSE_STAGES
from src.infrastructure.others import get_device
import torch.nn as nn
from dataclasses import dataclass
from src.infrastructure.schedulers import PressureSchedulerPolicy1
from src.infrastructure.training_context.training_context import \
    TrainingContextBaselineTrain, TrainingContextPrunedTrain, TrainingContextNPLHL0


@dataclass
class StagesContextBaselineTrainArgs:
    scheduler_weights_lr_during_training: torch.optim.lr_scheduler.LRScheduler
    training_end: int

class StagesContextBaselineTrain:
    def __init__(self, args: StagesContextBaselineTrainArgs):
        self.epoch = 1
        self.args = args

        if self.args.scheduler_weights_lr_during_training == None:
            warnings.warn("Scheduler weights pruning is disabled")

    def update_context(self, epoch: int):
        self.epoch = epoch

    def step(self, training_context: TrainingContextBaselineTrain):
        if VERBOSE_STAGES:
            try:
                print("Learning rates init, Weights:", training_context.get_optimizer_weights().param_groups[0]['lr'], " Flow mask:", training_context.get_optimizer_flow_mask().param_groups[0]['lr'])
            except AttributeError:
                print("Attribute error", training_context.get_optimizer_weights().param_groups[0]['lr'])

        if self.args.scheduler_weights_lr_during_training is not None:
            self.args.scheduler_weights_lr_during_training.step()

        if VERBOSE_STAGES:
            try:
                print("Learning rates init, Weights:", training_context.get_optimizer_weights().param_groups[0]['lr'], " Flow mask:", training_context.get_optimizer_flow_mask().param_groups[0]['lr'])
            except AttributeError:
                print("Attribute error", training_context.get_optimizer_weights().param_groups[0]['lr'])

@dataclass
class StagesContextSparsityCurveArgs:
    scheduler_weights_lr_during_pruning: torch.optim.lr_scheduler.LRScheduler
    epoch_end: int

class StagesContextSparsityCurve:
    def __init__(self, args: StagesContextSparsityCurveArgs):
        self.epoch = 1
        self.args = args

        if self.args.scheduler_weights_lr_during_pruning is None:
            warnings.warn("Scheduler weights pruning is disabled")

    def update_context(self, epoch: int, sparsity_percent: float):
        self.epoch = epoch
        self.sparsity_percent = sparsity_percent

    def step(self, training_context: TrainingContextNPLHL0):
        if VERBOSE_STAGES:
            print("Learning rates init, Weights:", training_context.get_optimizer_weights().param_groups[0]['lr'], " Flow mask:", training_context.get_optimizer_flow_mask().param_groups[0]['lr'])

        if self.epoch <= self.args.epoch_end:
            if self.args.scheduler_weights_lr_during_pruning is not None:
                self.args.scheduler_weights_lr_during_pruning.step()

        if VERBOSE_STAGES:
            print("Learning rates end, Weights:", training_context.get_optimizer_weights().param_groups[0]['lr'], " Flow mask:", training_context.get_optimizer_flow_mask().param_groups[0]['lr'])


@dataclass
class StagesContextPrunedTrainArgs:
    scheduler_weights_lr_during_pruning: torch.optim.lr_scheduler.LRScheduler

    scheduler_weights_lr_during_regrowth: torch.optim.lr_scheduler.LRScheduler
    scheduler_flow_params_regrowth: torch.optim.lr_scheduler.LRScheduler

    scheduler_gamma: PressureSchedulerPolicy1

    pruning_epoch_end: int
    regrowth_epoch_end: int

class StagesContextPrunedTrain:
    def __init__(self, args: StagesContextPrunedTrainArgs):
        self.epoch = 1
        self.sparsity_percent = 100
        self.args = args

        if self.args.scheduler_weights_lr_during_pruning == None:
            warnings.warn("Scheduler weights pruning is disabled")
        if self.args.scheduler_weights_lr_during_regrowth == None:
            warnings.warn("Scheduler weights regrowth is disabled")
        if self.args.scheduler_flow_params_regrowth == None:
            warnings.warn("Scheduler flow params regrowth is disabled")
        if self.args.scheduler_gamma == None:
            warnings.warn("Scheduler gamma is disabled")



    def update_context(self, epoch: int, sparsity_percent: float):
        self.epoch = epoch
        self.sparsity_percent = sparsity_percent

    def step(self, training_context: TrainingContextPrunedTrain):
        if VERBOSE_STAGES:
            try:
                print("Learning rates init, Weights:", training_context.get_optimizer_weights().param_groups[0]['lr'], " Flow mask:", training_context.get_optimizer_flow_mask().param_groups[0]['lr'])
            except AttributeError:
                print("Attribute error", training_context.get_optimizer_weights().param_groups[0]['lr'])

        if self.epoch <= self.args.pruning_epoch_end:
            # run weights scheduler
            if self.args.scheduler_weights_lr_during_pruning is not None:
                self.args.scheduler_weights_lr_during_pruning.step()

            # run gamma scheduler
            self.args.scheduler_gamma.step(self.epoch, self.sparsity_percent)
            gamma = self.args.scheduler_gamma.get_multiplier()
            if VERBOSE_STAGES:
                print("Gamma:", gamma)
            # gamma = 0

            training_context.set_gamma(gamma)

        if self.epoch == self.args.pruning_epoch_end + 1:
            # sets optimizer parameters to reset values
            training_context.reset_param_groups_to_defaults()
            self.args.scheduler_weights_lr_during_regrowth.base_lrs[0] = training_context.params.lr_weights_reset
            self.args.scheduler_flow_params_regrowth.base_lrs[0] = training_context.params.lr_flow_params_reset

        # if EXTRA_VERBOSE_STAGES:
        #     print("Learning rates mid, Weights:", training_context.get_optimizer_weights().param_groups[0]['lr'], " Flow mask:", training_context.get_optimizer_flow_mask().param_groups[0]['lr'])

        if self.epoch >= self.args.pruning_epoch_end + 1 and self.epoch <= self.args.regrowth_epoch_end:
            if self.args.scheduler_weights_lr_during_regrowth is not None:
                self.args.scheduler_weights_lr_during_regrowth.step()

            # we decay flow params lr, to stop flipping and limit regrowing
            if self.args.scheduler_flow_params_regrowth is not None:
                self.args.scheduler_flow_params_regrowth.step()

            # we remove pressure
            training_context.set_gamma(0)

        if VERBOSE_STAGES:
            try:
                print("Learning rates end, Weights:", training_context.get_optimizer_weights().param_groups[0]['lr'], " Flow mask:", training_context.get_optimizer_flow_mask().param_groups[0]['lr'])
            except AttributeError:
                print("Attribute error", training_context.get_optimizer_weights().param_groups[0]['lr'])



@dataclass
class StagesContextBottleneckTrainArgs:
    training_end: int

class StagesContextBottleneckTrain:
    def __init__(self, args: StagesContextBottleneckTrainArgs):
        self.epoch = 1
        self.args = args

    def update_context(self, epoch: int):
        self.epoch = epoch

    def step(self, training_context: TrainingContextPrunedTrain):
        stepconst = 10
        if self.epoch % stepconst == 0:
            newgm = 5 ** (self.epoch//stepconst)
            training_context.set_gamma(newgm)
            print("SET NEW PRESSURE TO -----------------", newgm)








@dataclass
class StagesContextNPLHTrainArgs:
    scheduler_gamma: PressureSchedulerPolicy1
    pruning_epoch_end: int

class StagesContextNPLHTrain:
    def __init__(self, args: StagesContextNPLHTrainArgs):
        self.epoch = 1
        self.sparsity_percent = 100
        self.args = args

    def update_context(self, epoch: int, sparsity_percent: float):
        self.epoch = epoch
        self.sparsity_percent = sparsity_percent

    def step(self, training_context: TrainingContextPrunedTrain):
        self.args.scheduler_gamma.step(self.epoch, self.sparsity_percent)
        gamma = self.args.scheduler_gamma.get_multiplier()
        if VERBOSE_STAGES:
            print("Gamma:", gamma)

        training_context.set_gamma(gamma)