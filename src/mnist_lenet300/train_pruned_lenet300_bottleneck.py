import torch
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.constants import LR_FLOW_PARAMS_ADAM, LR_FLOW_PARAMS_ADAM_RESET, get_lr_flow_params, \
    get_lr_flow_params_reset, config_adam_setup, PRUNED_MODELS_PATH, BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar10, dataset_context_configs_mnist
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent
from src.infrastructure.training_common import get_model_flow_params_and_weights_params
from src.infrastructure.training_context.training_context import TrainingContextPrunedTrain, \
    TrainingContextPrunedTrainArgs, TrainingContextPrunedBottleneckTrain, TrainingContextPrunedBottleneckTrainArgs
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags
from .model_class_bottleneck import ModelLenet300Bottleneck
from ..common_files_experiments.train_pruned_commons import train_mixed_pruned, test_pruned
from ..infrastructure.stages_context.stages_context import StagesContextBottleneckTrain, \
    StagesContextBottleneckTrainArgs


def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )
    MODEL = ModelLenet300Bottleneck(configs_network_masks).to(get_device())

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

    lr_weights_finetuning = 0.001
    lr_weights = lr_weights_finetuning
    lr_flow_params = get_lr_flow_params()

    weight_bias_params, flow_params = get_model_flow_params_and_weights_params(MODEL)
    optimizer_weights = torch.optim.Adam(lr=lr_weights, params=weight_bias_params, weight_decay=0)
    optimizer_flow_mask = torch.optim.Adam(lr=lr_flow_params, params=flow_params, weight_decay=0)

    training_context = TrainingContextPrunedBottleneckTrain(
        TrainingContextPrunedBottleneckTrainArgs(
            l0_gamma_scaler=0,
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask
        )
    )

def initialize_stages_context():
    global stages_context, training_context
    epochs = 100

    stages_context = StagesContextBottleneckTrain(
        StagesContextBottleneckTrainArgs(
            training_end=epochs
        ),
    )

MODEL: ModelLenet300Bottleneck
training_context: TrainingContextPrunedTrain
dataset_context: DatasetSmallContext
stages_context: StagesContextBottleneckTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

def train_pruned_lenet300_mnist_bottleneck():
    global epoch_global, MODEL
    configs_layers_initialization_all_kaiming_sqrt5()
    config_adam_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    initialize_dataset_context()
    initalize_training_display()

    wandb_initalize(
        experiment=Experiment.LENET300MNIST,
        type=Tags.TRAIN_PRUNING,
        configs=None,
        other_tags=["ADAM"]
    )

    acc = 0
    for epoch in range(1, stages_context.args.training_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed_pruned(
            dataset_context=dataset_context,
            training_context=training_context,
            model=MODEL,
            training_display=training_display,
        )
        acc = test_pruned(
            dataset_context=dataset_context,
            model=MODEL,
            epoch=get_epoch()
        )

        stages_context.update_context(epoch_global)
        stages_context.step(training_context)

    wandb_finish()
