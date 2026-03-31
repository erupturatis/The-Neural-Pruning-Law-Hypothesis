import torch
from src.experiments.resnet50_variable_cifar100_train_dense import train_dense_resnet50_cifar100
from src.model_resnet50_cifars.model_resnet50_variable_class import ModelResnet50Variable
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device

def run():
    device = get_device()
    print(f"Running ResNet50 CIFAR-100 on {device}")
    
    cfg: ConfigsNetworkMask = {
        "mask_apply_enabled": False,
        "mask_training_enabled": False,
        "weights_training_enabled": True
    }
    model = ModelResnet50Variable(alpha=1.0, config_network_mask=cfg, num_classes=100).to(device)
    train_dense_resnet50_cifar100(model)

if __name__ == "__main__":
    run()
