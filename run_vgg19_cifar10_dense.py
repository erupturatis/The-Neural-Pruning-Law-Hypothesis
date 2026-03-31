import torch
from src.experiments.vgg19_variable_cifar10_train_dense import train_dense_vgg19_cifar10
from src.model_vgg19_cifars.model_vgg19_variable_class import ModelVGG19Variable
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device

def run():
    device = get_device()
    print(f"Running VGG19 CIFAR-10 on {device}")
    
    cfg: ConfigsNetworkMask = {
        "mask_apply_enabled": False,
        "mask_training_enabled": False,
        "weights_training_enabled": True
    }
    model = ModelVGG19Variable(alpha=1.0, config_network_mask=cfg, num_classes=10).to(device)
    train_dense_vgg19_cifar10(model)

if __name__ == "__main__":
    run()
