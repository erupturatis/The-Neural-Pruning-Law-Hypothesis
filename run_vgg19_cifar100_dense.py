import torch
from src.experiments.vgg19_variable_cifar100_train_dense import train_dense_vgg19_cifar100
from src.model_vgg19_cifars.model_vgg19_variable_class import ModelVGG19Variable
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_relu

def run():
    device = get_device()
    print(f"Running VGG19 CIFAR-100 on {device}")

    configs_layers_initialization_all_kaiming_relu()

    cfg: ConfigsNetworkMask = {
        "mask_apply_enabled": False,
        "mask_training_enabled": False,
        "weights_training_enabled": True
    }
    model = ModelVGG19Variable(alpha=1.0, config_network_mask=cfg, num_classes=100).to(device)
    train_dense_vgg19_cifar100(model)

if __name__ == "__main__":
    run()
