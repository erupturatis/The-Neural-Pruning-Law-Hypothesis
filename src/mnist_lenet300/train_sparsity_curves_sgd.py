import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
from .model_class import ModelLenet300
import wandb
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.constants import LR_FLOW_PARAMS_ADAM, LR_FLOW_PARAMS_ADAM_RESET, get_lr_flow_params, \
    get_lr_flow_params_reset, config_sgd_setup
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar10, dataset_context_configs_mnist
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent, save_array_experiment
from src.infrastructure.schedulers import PressureSchedulerPolicy1
from src.infrastructure.training_common import get_model_flow_params_and_weights_params
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish
from ..common_files_experiments.generate_sparsity_curves_commons import train_mixed_curves, test_curves
from ..infrastructure.stages_context.stages_context import StagesContextSparsityCurve, StagesContextSparsityCurveArgs
from ..infrastructure.training_context.training_context import TrainingContextNPLHL0, \
    TrainingContextNPLHL0Args


def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )
    MODEL = ModelLenet300(configs_network_masks).to(get_device())

def get_epoch() -> int:
    global epoch_global
    return epoch_global

def initalize_training_display():
    global training_display
    training_display = TrainingDisplay(
        args=ArgsTrainingDisplay(
            dataset_context=dataset_context,
            average_losses_names=["Loss Data", "Loss Remaining Weights"],
            model=MODEL,
            batch_print_rate=BATCH_PRINT_RATE,
            get_epoch= get_epoch
        )
    )

def initialize_dataset_context():
    global dataset_context
    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.MNIST, configs=dataset_context_configs_mnist())


def initialize_training_context():
    global training_context, MODEL

    lr_weights_training = 0.005
    lr_weights_finetuning = 0.001

    lr_weights = lr_weights_training
    lr_flow_params = get_lr_flow_params()

    weight_bias_params, flow_params = get_model_flow_params_and_weights_params(MODEL)
    optimizer_weights = torch.optim.Adam(lr=lr_weights, params=weight_bias_params, weight_decay=0)
    optimizer_flow_mask = torch.optim.SGD(lr=lr_flow_params, params=flow_params, weight_decay=0, momentum=0.9)

    # reset weights are applied after pruning and before regrowth, they are the starting point for the regrowth schedulers
    training_context = TrainingContextNPLHL0(
        TrainingContextNPLHL0Args(
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    pruning_end = 300
    scheduler_weights_lr_during_pruning = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=pruning_end, eta_min=1e-7)

    stages_context = StagesContextSparsityCurve(
        StagesContextSparsityCurveArgs(
            epoch_end=pruning_end,
            scheduler_weights_lr_during_pruning=scheduler_weights_lr_during_pruning,
        ),
    )


MODEL: ModelLenet300
training_context: TrainingContextNPLHL0
dataset_context: DatasetSmallContext
stages_context: StagesContextSparsityCurve
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

PRESSURE: float = 0
sparsity_levels_recording =[]
BATCH_RECORD_FREQ = 128

def _init_data_arrays():
    global sparsity_levels_recording
    sparsity_levels_recording = []

def run_lenet300_mnist_sgd_sparsity_curve(arg:float, power_start:int, power_end:int):
    global PRESSURE, sparsity_levels_recording
    for pw in range(power_start, power_end+1):
        PRESSURE = arg ** pw
        _init_data_arrays()
        _run_lenet300_mnist_sgd()
        save_array_experiment(f"mnist_lenet300_sgd_{PRESSURE}.json", sparsity_levels_recording)

def _run_lenet300_mnist_sgd():
    global epoch_global, sparsity_levels_recording
    configs_layers_initialization_all_kaiming_sqrt5()
    config_sgd_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    initialize_dataset_context()
    initalize_training_display()

    for epoch in range(1, stages_context.args.regrowth_epoch_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()

        sparsity_levels_recording = train_mixed_curves(
            dataset_context=dataset_context,
            model=MODEL,
            training_context=training_context,
            PRESSURE=PRESSURE,
            BATCH_RECORD_FREQ=BATCH_RECORD_FREQ,
            training_display=training_display,
            sparsity_levels_recording=sparsity_levels_recording
        )
        test_curves(
            model=MODEL,
            dataset_context=dataset_context,
        )

        stages_context.update_context(epoch_global, get_custom_model_sparsity_percent(MODEL))
        stages_context.step(training_context)