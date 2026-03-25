import torchvision.models as models
import torch

from src.vgg19_cifar10.vgg19_cifar10_attributes import VGG19_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.common_files_experiments.test_existing_model import test_existing_model
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar10
from src.infrastructure.others import prefix_path_with_root, get_device, get_raw_model_sparsity_percent
from torch import nn


def run_cifar10_vgg19_existing_model(model_name:str, folder: str):
    dataset_context: DatasetSmallContext
    filepath = folder + '/' + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)

    model = models.vgg19()

    model.classifier[0] = nn.Linear(512, 4096)
    model.classifier[6] = nn.Linear(4096, 10)


    model.load_state_dict(state_dict)
    model = model.to(get_device())
    get_raw_model_sparsity_percent(
        model=model,
        standard_to_custom_attributes=VGG19_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
    )

    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.CIFAR10, configs=dataset_context_configs_cifar10())

    num_epochs = 10
    for epoch in range(1, num_epochs):
        dataset_context.init_data_split()
        test_existing_model(model, dataset_context)


