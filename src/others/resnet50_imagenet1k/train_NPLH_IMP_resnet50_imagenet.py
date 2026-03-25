import torch
from src.common_files_experiments.train_pruned_commons import train_mixed_pruned, test_pruned, \
    train_mixed_pruned_imagenet, test_pruned_imagenet, train_mixed_pruned_imagenet_IMP
from src.infrastructure.stages_context.stages_context import StagesContextPrunedTrain, StagesContextPrunedTrainArgs
from src.infrastructure.training_context.training_context import TrainingContextPrunedTrain, \
    TrainingContextPrunedTrainArgs
from src.resnet50_imagenet1k.resnet50_imagenet_class import Resnet50Imagenet
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5, \
    configs_layers_initialization_all_kaiming_relu
from src.infrastructure.constants import config_adam_setup, get_lr_flow_params_reset, get_lr_flow_params, \
    PRUNED_MODELS_PATH, BASELINE_MODELS_PATH
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar10, DatasetImageNetContext, DatasetImageNetContextConfigs
from src.infrastructure.training_display import TrainingDisplay, ArgsTrainingDisplay
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent, TrainingConfigsWithResume
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from src.infrastructure.training_utils import \
    get_model_flow_params_and_weights_params_bn_separate
from src.infrastructure.wandb_functions import wandb_initalize, wandb_finish, Experiment, Tags
from torch import nn
from src.infrastructure.stages_context.stages_context import StagesContextBaselineTrain, StagesContextBaselineTrainArgs
from src.infrastructure.training_context.training_context import TrainingContextBaselineTrain, TrainingContextBaselineTrainArgs
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device, TrainingConfigsNPLHIMP
from src.infrastructure.layers import prune_model_globally, calculate_pruning_epochs
from src.infrastructure.read_write import save_dict_to_csv

def initialize_model():
    global MODEL, MODEL_MODULE, training_configs
    configs_network_masks = ConfigsNetworkMask(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    MODEL = Resnet50Imagenet(configs_network_masks)
    if "resume" in training_configs:
        MODEL.load(training_configs["resume"], BASELINE_MODELS_PATH)

    print(f"Number of available CUDA devices: {torch.cuda.device_count()}")
    print("LOADED IMAGENET")
    MODEL = MODEL.to(get_device())
    if torch.cuda.device_count() > 1:
        MODEL = nn.DataParallel(MODEL, device_ids=[0,1,2])
        MODEL_MODULE = MODEL.module
    else:
        # IF YOU USE A SINGLE GPU, JUST UNCOMMENT THE FOLLOWING !! AND REMOVE THE nn.DataParallel from above
        MODEL_MODULE = MODEL


def get_epoch() -> int:
    global epoch_global
    return epoch_global

def initalize_training_display():
    global training_display
    training_display = TrainingDisplay(
        args=ArgsTrainingDisplay(
            dataset_context=dataset_context,
            average_losses_names=["Loss Data", "Loss Remaining Weights"],
            model=MODEL_MODULE,
            batch_print_rate=BATCH_PRINT_RATE,
            get_epoch= get_epoch
        )
    )

def initialize_dataset_context():
    global dataset_context
    configs = DatasetImageNetContextConfigs(
        batch_size= 512
    )
    dataset_context = DatasetImageNetContext(configs)

def initialize_training_context():
    global training_context

    lr_weights_finetuning = training_configs["start_lr_pruning"]
    weight_params, _, no_decay_params = get_model_flow_params_and_weights_params_bn_separate(MODEL)
    optimizer_weights = torch.optim.SGD(
        [
            { "params": weight_params,     "weight_decay": training_configs["weight_decay"]},  # conv/linear weights
            { "params": no_decay_params,   "weight_decay": 0.0          },  # biases & norm‑affines
        ],
        lr=lr_weights_finetuning,
        momentum=0.9,
    )

    training_context = TrainingContextBaselineTrain(
        TrainingContextBaselineTrainArgs(
            optimizer_weights=optimizer_weights,
        )
    )

def initialize_stages_context():
    global stages_context, training_context

    training_end = training_configs["training_end"]
    scheduler_weights_lr_during_pruning = CosineAnnealingLR(training_context.get_optimizer_weights(), T_max=training_end, eta_min=training_configs["end_lr_pruning"])

    stages_context = StagesContextBaselineTrain(
        StagesContextBaselineTrainArgs(
            training_end=training_end,
            scheduler_weights_lr_during_training=scheduler_weights_lr_during_pruning,
        )
    )

    
MODEL: Resnet50Imagenet
MODEL_MODULE: any
training_context: TrainingContextBaselineTrain
dataset_context: DatasetImageNetContext
stages_context: StagesContextBaselineTrain
training_display: TrainingDisplay
epoch_global: int = 0
BATCH_PRINT_RATE = 100

training_configs: TrainingConfigsNPLHIMP

def train_resnet50_imagenet_NPLH_IMP(sparsity_configs_aux: TrainingConfigsNPLHIMP):
    global epoch_global, MODEL_MODULE, training_configs
    sparsity_configs = sparsity_configs_aux
    training_configs = sparsity_configs_aux

    configs_layers_initialization_all_kaiming_relu()
    config_adam_setup()

    initialize_model()
    print("LOADED SUCCESS")
    initialize_training_context()
    initialize_stages_context()
    wandb_initalize(Experiment.RESNET50IMAGENET, type=Tags.TRAIN_PRUNING, configs=sparsity_configs,other_tags=["ADAM"])
    MODEL_MODULE.save_entire_dict("intialisation_resnet50_imagenet")
    initialize_dataset_context()
    initalize_training_display()


    pruning_rate = 0.1
    epochs_to_prune = []
    epochs_to_prune = calculate_pruning_epochs(
        target_sparsity=training_configs["target_sparsity"]/100, 
        pruning_rate=pruning_rate, 
        total_epochs=training_configs["training_end"], 
        start_epoch=1
    )
    print("EPOCHS TO PRUNE", epochs_to_prune)

    acc = 0
    thresholds = []
    remaining_params = []
    pruned_epochs = []
    accuracies = []

    acc = 0
    for epoch in range(1, training_configs["training_end"] + 1):
    
        epoch_global = epoch
        dataset_context.init_data_split()
        train_mixed_pruned_imagenet_IMP(
            dataset_context=dataset_context,
            training_context=training_context,
            model=MODEL,
            model_module= MODEL_MODULE,
            training_display=training_display,
        )
        acc = test_pruned_imagenet(
            dataset_context=dataset_context,
            model=MODEL,
            model_module= MODEL_MODULE,
            epoch=get_epoch()
        )

        stages_context.update_context(epoch_global)
        stages_context.step(training_context)

        if epoch in epochs_to_prune: 
            val = prune_model_globally(MODEL_MODULE, pruning_rate)
            rem = get_custom_model_sparsity_percent(MODEL_MODULE)
            thresholds.append(val)
            remaining_params.append(rem)
            pruned_epochs.append(epoch)
            accuracies.append(acc)

            print(thresholds)
            print(remaining_params)
            print(pruned_epochs)
            print(accuracies)

            MODEL_MODULE.save(f"/resnet50_imagenet_sparsity{get_custom_model_sparsity_percent(MODEL_MODULE)}_acc{acc}_{epoch}", "networks_pruned")
            MODEL_MODULE.save_entire_dict(f"/resnet50_entire_imagenet_sparsity{get_custom_model_sparsity_percent(MODEL_MODULE)}_acc{acc}_{epoch}")
        
        save_dict_to_csv({
            "Epoch": pruned_epochs, 
            "Saliency": thresholds, 
            "Remaining": remaining_params,
            "Accuracy": accuracies
        },
        filename="impr50ImageNet.csv" 
        )

    print("Training complete")
    wandb_finish()
