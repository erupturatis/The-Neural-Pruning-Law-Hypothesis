import torch
from src.common_files_experiments.train_model_scratch_commons import train_mixed_baseline, test_baseline
from src.common_files_experiments.train_pruned_commons import train_mixed_pruned, test_pruned
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5, \
    configs_layers_initialization_all_kaiming_relu
from src.infrastructure.constants import config_adam_setup, get_lr_flow_params_reset, get_lr_flow_params, \
    PRUNED_MODELS_PATH, BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar100
from src.infrastructure.stages_context.stages_context import  \
    StagesContextBaselineTrain, StagesContextBaselineTrainArgs
from src.infrastructure.training_context.training_context import \
    TrainingContextBaselineTrain, TrainingContextBaselineTrainArgs
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent, get_random_id
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.infrastructure.schedulers import PressureSchedulerPolicy1
from src.infrastructure.training_common import get_model_weights_params
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags
from src.resnet50_cifar100.resnet50_cifar100_class import Resnet50Cifar100

def initialize_model():
    global MODEL
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )

    MODEL = Resnet50Cifar100(configs_network_masks).to(get_device())

def get_epoch() -> int:
    global epoch_global
    return epoch_global

def initalize_training_display():
    global training_display
    training_display = TrainingDisplay(
        args=ArgsTrainingDisplay(
            dataset_context=dataset_context,
            average_losses_names=["Loss Data"],
            model=MODEL,
            batch_print_rate=BATCH_PRINT_RATE,
            get_epoch= get_epoch
        )
    )

def initialize_dataset_context():
    global dataset_context
    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR100, configs=dataset_context_configs_cifar100())


def initialize_training_context():
    global training_context

    lr = 0.1
    weight_bias_params = get_model_weights_params(MODEL)
    optimizer_weights = torch.optim.SGD(lr=lr, params=weight_bias_params, momentum=0.9, weight_decay=5e-4, nesterov=True)

    training_context = TrainingContextBaselineTrain(
        TrainingContextBaselineTrainArgs(
            optimizer_weights=optimizer_weights,
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    training_end = 160
    scheduler_weights_lr_during_training = torch.optim.lr_scheduler.MultiStepLR(training_context.get_optimizer_weights(), milestones=[int(training_end/ 2), int(training_end * 3 / 4)], last_epoch=-1)

    stages_context = StagesContextBaselineTrain(
        StagesContextBaselineTrainArgs(
            training_end=training_end,
            scheduler_weights_lr_during_training=scheduler_weights_lr_during_training,
        )
    )

MODEL: Resnet50Cifar100
training_context: TrainingContextBaselineTrain
dataset_context: DatasetSmallContext
stages_context: StagesContextBaselineTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

def train_resnet50_cifar100_from_scratch_multistep():
    global MODEL, epoch_global
    configs_layers_initialization_all_kaiming_relu()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize(experiment=Experiment.RESNET50CIFAR100, type=Tags.BASELINE)
    initialize_dataset_context()
    initalize_training_display()

    acc = 0
    for epoch in range(1, stages_context.args.training_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed_baseline(
            model=MODEL,
            dataset_context=dataset_context,
            training_context=training_context,
            training_display=training_display,
        )
        acc = test_baseline(
            model=MODEL,
            dataset_context=dataset_context,
            epoch=epoch,
        )

        stages_context.update_context(epoch_global)
        stages_context.step(training_context)

    MODEL.save(
        name=f"resnet50_cifar100_accuracy{acc}%_{get_random_id()}",
        folder=BASELINE_MODELS_PATH
    )

    print("Training complete")
    wandb_finish()
