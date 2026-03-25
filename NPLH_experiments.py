import argparse
import sys
from src.infrastructure.constants import INITIAL_LR 
from src.infrastructure.others import TrainingConfigsNPLHIMP 
from src.resnet50_cifar10.train_NPLH_IMP_resnet50_cifar10 import train_resnet50_cifar10_IMP
from src.vgg19_cifar100.train_NPLH_IMP_vgg19_cifar100 import train_vgg19_cifar100_IMP
from src.resnet50_imagenet1k.train_NPLH_IMP_resnet50_imagenet import train_resnet50_imagenet_NPLH_IMP
from src.resnet50_cifar10.train_sparsity_curves_adam import generate_cifar10_resnet50_adam_sparsity_curve
from src.resnet50_cifar10.train_sparsity_curves_sgd import run_cifar10_resnet50_sgd_sparsity_curve
from src.vgg19_cifar100.train_NPLH_L0_vgg19_cifar100 import generate_cifar100_vgg19_adam_sparsity_curve

def traing_r50c10_IMP(): 
   defaults: TrainingConfigsNPLHIMP = {
      "training_end": 500,
      "start_lr_pruning": INITIAL_LR / 10,
      "end_lr_pruning": INITIAL_LR / 10,
      "weight_decay": 5e-4,
      "target_sparsity": 99.975,
      "resume": "resnet50_cifar10_accuracy94.91%", 
   }

   train_resnet50_cifar10_IMP(defaults)

def train_vgg19_c100_IMP(): 
   defaults: TrainingConfigsNPLHIMP = {
      "training_end": 500,
      "start_lr_pruning": INITIAL_LR / 10,
      "end_lr_pruning": INITIAL_LR / 10,
      "weight_decay": 5e-4,
      "target_sparsity": 99.95,
      "resume": "vgg19_cifar100_accuracy72.9%", 
   }

   train_vgg19_cifar100_IMP(defaults)

def train_r50c10_NPLH_Hyperflux():
   # generate curves either with SGD or ADAM optimizer for flow params

   generate_cifar10_resnet50_adam_sparsity_curve(arg=2, power_start=-15, power_end=10)
   # run_cifar10_resnet50_sgd_sparsity_curve(arg=2, power_start=-15, power_end=10)

def train_vgg19c100_NPLH_Hyperflux():
   generate_cifar100_vgg19_adam_sparsity_curve(arg=2, power_start=-15, power_end=10)


if __name__ == "__main__":
   pass