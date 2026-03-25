import enum
from enum import Enum
from typing import List, Dict
import wandb
from src.infrastructure.configs_general import WANDB_REGISTER

class Experiment(Enum):
    LENET300MNIST = "lenet300_mnist"

    RESNET18CIFAR10 = "resnet18_cifar10"
    RESNET50CIFAR10 = "resnet50_cifar10"
    VGG19CIFAR10 = "vgg19_cifar10"

    RESNET18CIFAR100 = "resnet18_cifar100"
    RESNET50CIFAR100 = "resnet50_cifar100"
    VGG19CIFAR100 = "vgg19_cifar100"

    RESNET50IMAGENET = "resnet50_imagenet"

class Tags(Enum):
    BASELINE = "baseline"
    IMP = "IMP"
    TRAIN_PRUNING = "train_pruning"


def wandb_initalize(experiment: Experiment, type: Tags, configs: Dict = None, other_tags: List[str] = [], note = '') -> None:
    if WANDB_REGISTER:
        wandb.init(
            project=experiment.value,
            config=configs,
            tags=[type.value, *other_tags],
            notes=note
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

def wandb_snapshot_baseline(epoch:int, accuracy: float, test_loss:float, others: Dict = None):
    if WANDB_REGISTER:
        # print("REGISERED DATA")
        wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy, "others": others})

def wandb_snapshot(epoch: int, accuracy: float, test_loss: float, sparsity: float, others: Dict = None):
    if WANDB_REGISTER:
        wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy, "sparsity": sparsity, "others": others})

def wandb_finish():
    if WANDB_REGISTER:
        wandb.finish()
