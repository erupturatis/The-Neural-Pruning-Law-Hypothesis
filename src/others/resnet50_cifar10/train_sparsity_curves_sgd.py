from src.common_files_experiments.generate_sparsity_curves_commons import test_curves, train_mixed_curves
import torch
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.constants import config_adam_setup, get_lr_flow_params, BASELINE_MODELS_PATH, config_sgd_setup
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar10
from src.infrastructure.others import get_device, save_array_experiment, get_custom_model_sparsity_percent
from src.infrastructure.stages_context.stages_context import StagesContextSparsityCurve, StagesContextSparsityCurveArgs
from src.infrastructure.training_utils import get_model_flow_params_and_weights_params
from src.infrastructure.training_context.training_context import TrainingContextNPLHL0Args, \
    TrainingContextNPLHL0
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMask
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.resnet50_cifar10.resnet50_cifar10_class import Resnet50Cifar10

def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMask(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )
    MODEL = Resnet50Cifar10(configs_network_masks).to(get_device())
    MODEL.load("resnet50_cifar10_accuracy94.91%", BASELINE_MODELS_PATH)

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
            get_epoch=get_epoch
        )
    )

def initialize_dataset_context():
    global dataset_context
    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR10, configs=dataset_context_configs_cifar10())

lr_weights_finetuning = 1e-3
def initialize_training_context():
    global training_context

    lr_flow_params = get_lr_flow_params()

    weight_bias_params, flow_params = get_model_flow_params_and_weights_params(MODEL)
    optimizer_weights = torch.optim.SGD(lr=lr_weights_finetuning, params= weight_bias_params, momentum=0.9, weight_decay=0)
    optimizer_flow_mask = torch.optim.SGD(lr=lr_flow_params, params=flow_params, weight_decay=0, momentum=0.9)

    training_context = TrainingContextNPLHL0(
        TrainingContextNPLHL0Args(
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask, 
            l0_gamma_scaler=PRESSURE
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    pruning_end = 1000
    scheduler_weights_lr_during_pruning = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=pruning_end, eta_min=lr_weights_finetuning)

    stages_context = StagesContextSparsityCurve(
        StagesContextSparsityCurveArgs(
            epoch_end=pruning_end,
            scheduler_weights_lr_during_pruning=scheduler_weights_lr_during_pruning,
        )
    )

MODEL: Resnet50Cifar10
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

def run_cifar10_resnet50_sgd_sparsity_curve(arg:float, power_start:int, power_end:int):
    global PRESSURE, sparsity_levels_recording
    for pw in range(power_start, power_end+1):
        PRESSURE = arg ** pw
        _init_data_arrays()
        _run_cifar10_resnet50_sgd()
        save_array_experiment(f"cifar10_resnet50_sgd_highlr_{PRESSURE}.json", sparsity_levels_recording)


def _run_cifar10_resnet50_sgd():
    global MODEL, epoch_global, sparsity_levels_recording
    configs_layers_initialization_all_kaiming_sqrt5()
    config_sgd_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    initialize_dataset_context()
    initalize_training_display()

    for epoch in range(1, stages_context.args.epoch_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed_curves(
            dataset_context=dataset_context,
            model=MODEL,
            training_context=training_context,
            PRESSURE=PRESSURE,
            BATCH_RECORD_FREQ=BATCH_RECORD_FREQ,
            training_display=training_display,
            sparsity_levels_recording=[]
        )
        test_curves(
            model=MODEL,
            dataset_context=dataset_context,
        )

        sparsity_levels_recording.append(get_custom_model_sparsity_percent(MODEL))
        stages_context.update_context(epoch_global, get_custom_model_sparsity_percent(MODEL))
        stages_context.step(training_context)
