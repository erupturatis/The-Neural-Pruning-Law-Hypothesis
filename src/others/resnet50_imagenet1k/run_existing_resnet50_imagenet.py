import torchvision.models as models
import torch
from src.common_files_experiments.test_existing_model import test_existing_model
from src.infrastructure.dataset_context.dataset_context import DatasetImageNetContext, DatasetImageNetContextConfigs
from src.infrastructure.others import prefix_path_with_root, get_device


def run_imagenet_resnet50_existing_model(model_name:str, folder: str):
    dataset_context: DatasetImageNetContext

    filepath = folder + '/' + model_name
    filepath = prefix_path_with_root(filepath)
    state_dict = torch.load(filepath)

    model = models.resnet50()
    model.load_state_dict(state_dict)
    model = model.to(get_device())

    configs = DatasetImageNetContextConfigs(batch_size=128 * 4)
    dataset_context = DatasetImageNetContext(configs=configs, cache_dir = "")

    num_epochs = 10
    for epoch in range(1, num_epochs):
        dataset_context.init_data_split()
        test_existing_model(model, dataset_context)


