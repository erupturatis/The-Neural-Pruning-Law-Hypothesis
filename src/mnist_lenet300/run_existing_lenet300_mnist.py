import torchvision.models as models
import torch

from src.resnet18_cifar10.resnet18_cifar10_attributes import RESNET18_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING
from src.common_files_experiments.test_existing_model import test_existing_model
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_cifar10, dataset_context_configs_mnist
from src.infrastructure.others import prefix_path_with_root, get_device, get_raw_model_sparsity_percent
from torch import nn

from src.mnist_lenet300.model_attributes import LENET300_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING


class LeNet300100(nn.Module):
    def __init__(self):
        super(LeNet300100, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)  # Input size for MNIST images (28x28)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)      # Output size for 10 classes (digits 0-9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)    # Optional dropout for regularization

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def run_mnist_lenet300_existing_model(model_name:str, folder: str):
    dataset_context: DatasetSmallContext

    filepath = folder + '/' + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)

    model = LeNet300100() # no lenet in pytorch

    model.load_state_dict(state_dict)
    model = model.to(get_device())
    get_raw_model_sparsity_percent(model, LENET300_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING)

    dataset_context = DatasetSmallContext(dataset=DatasetSmallType.MNIST, configs=dataset_context_configs_mnist())
    num_epochs = 10

    for epoch in range(1, num_epochs):
        dataset_context.init_data_split()
        test_existing_model(model, dataset_context)


