import torch

from src.infrastructure.stages_context.stages_context import (
    StagesContextPrunedTrain,
    StagesContextPrunedTrainArgs,
)
from src.infrastructure.training_context.training_context import (
    TrainingContextPrunedTrain,
    TrainingContextPrunedTrainArgs,
)
from src.common_files_experiments.train_pruned_commons import (
    train_mixed_pruned,
    test_pruned,
)
from src.infrastructure.configs_layers import (
    configs_layers_initialization_all_kaiming_sqrt5,
    configs_layers_initialization_all_kaiming_relu,
)
from src.infrastructure.constants import (
    config_adam_setup,
    get_lr_flow_params_reset,
    get_lr_flow_params,
    PRUNED_MODELS_PATH,
    BASELINE_MODELS_PATH,
)
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext,
    DatasetSmallType,
    dataset_context_configs_cifar100,
)
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.others import (
    get_device,
    get_custom_model_sparsity_percent,
    get_random_id,
    TrainingConfigsWithResume,
)
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.infrastructure.schedulers import PressureSchedulerPolicy1
from src.infrastructure.training_common import get_model_flow_params_and_weights_params
from src.infrastructure.wandb_functions import (
    wandb_initalize,
    wandb_finish,
    Experiment,
    Tags,
)
from src.vit.prunable_vit import VisionTransformerPrunable


def initialize_model():
    global MODEL, training_configs
    configs_network_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=True,
        weights_training_enabled=True,
    )
    MODEL = VisionTransformerPrunable(
        configs_network_masks=configs_network_masks,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=100,
        embed_dim=384,  # DeiT-S
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
    ).to(get_device())
    if "resume" in training_configs:
        MODEL.load(training_configs["resume"], BASELINE_MODELS_PATH)


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
            get_epoch=get_epoch,
        )
    )


def initialize_dataset_context():
    global dataset_context
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.CIFAR100, configs=dataset_context_configs_cifar100()
    )


def initialize_training_context():
    global training_context

    lr_weights_finetuning = training_configs["start_lr_pruning"]
    lr_flow_params = get_lr_flow_params()

    weight_bias_params, flow_params = get_model_flow_params_and_weights_params(MODEL)
    optimizer_weights = torch.optim.SGD(
        lr=lr_weights_finetuning,
        params=weight_bias_params,
        momentum=0.9,
        weight_decay=training_configs["weight_decay"],
    )
    optimizer_flow_mask = torch.optim.Adam(
        lr=lr_flow_params, params=flow_params, weight_decay=0
    )

    training_context = TrainingContextPrunedTrain(
        TrainingContextPrunedTrainArgs(
            lr_weights_reset=training_configs["reset_lr_pruning"],
            lr_flow_params_reset=get_lr_flow_params()
            * training_configs["reset_lr_flow_params_scaler"],
            l0_gamma_scaler=0,
            optimizer_weights=optimizer_weights,
            optimizer_flow_mask=optimizer_flow_mask,
        )
    )


def initialize_stages_context():
    global stages_context, training_context

    pruning_end = training_configs["pruning_end"]
    regrowing_end = training_configs["regrowing_end"]
    regrowth_stage_length = regrowing_end - pruning_end

    pruning_scheduler = PressureSchedulerPolicy1(
        pressure_exponent_constant=1.5,
        sparsity_target=training_configs["target_sparsity"],
        epochs_target=pruning_end,
        step_size=0.2,
    )
    scheduler_decay_after_pruning = training_configs["lr_flow_params_decay_regrowing"]

    scheduler_weights_lr_during_pruning = CosineAnnealingLR(
        training_context.get_optimizer_weights(),
        T_max=pruning_end,
        eta_min=training_configs["end_lr_pruning"],
    )
    scheduler_weights_lr_during_regrowth = CosineAnnealingLR(
        training_context.get_optimizer_weights(),
        T_max=regrowth_stage_length,
        eta_min=training_configs["end_lr_regrowth"],
    )
    scheduler_flow_params_lr_during_regrowth = LambdaLR(
        training_context.get_optimizer_flow_mask(),
        lr_lambda=lambda iter: scheduler_decay_after_pruning**iter if iter < 50 else 0,
    )

    stages_context = StagesContextPrunedTrain(
        StagesContextPrunedTrainArgs(
            pruning_epoch_end=pruning_end,
            regrowth_epoch_end=regrowing_end,
            scheduler_gamma=pruning_scheduler,
            scheduler_weights_lr_during_pruning=scheduler_weights_lr_during_pruning,
            scheduler_flow_params_regrowth=scheduler_flow_params_lr_during_regrowth,
            scheduler_weights_lr_during_regrowth=scheduler_weights_lr_during_regrowth,
        )
    )


MODEL: VisionTransformerPrunable
training_context: TrainingContextPrunedTrain
dataset_context: DatasetSmallContext
stages_context: StagesContextPrunedTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

training_configs: TrainingConfigsWithResume


def train_vit_cifar100_sparse_model(sparsity_configs_aux: TrainingConfigsWithResume):
    global MODEL, epoch_global, training_configs
    sparsity_configs = sparsity_configs_aux
    training_configs = sparsity_configs_aux

    configs_layers_initialization_all_kaiming_relu()
    config_adam_setup()

    initialize_model()
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize(
        Experiment.VGG19CIFAR100,
        type=Tags.TRAIN_PRUNING,
        configs=sparsity_configs,
        other_tags=["ADAM"],
    )
    initialize_dataset_context()
    initalize_training_display()

    acc = 0
    id = get_random_id()
    for epoch in range(1, stages_context.args.regrowth_epoch_end + 1):
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed_pruned(
            dataset_context=dataset_context,
            training_context=training_context,
            model=MODEL,
            training_display=training_display,
        )
        acc = test_pruned(
            dataset_context=dataset_context, model=MODEL, epoch=get_epoch()
        )

        stages_context.update_context(
            epoch_global, get_custom_model_sparsity_percent(MODEL)
        )
        stages_context.step(training_context)

    MODEL.save(
        name=f"vit_cifar100_sparsity{get_custom_model_sparsity_percent(MODEL)}_acc{acc}_{id}",
        folder=PRUNED_MODELS_PATH,
    )
    print("Training complete")
    wandb_finish()
