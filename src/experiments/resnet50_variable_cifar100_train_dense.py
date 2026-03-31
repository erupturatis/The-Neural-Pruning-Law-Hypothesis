import torch
import torch.nn as nn

from src.common_files_experiments.load_save import save_model_entire_dict
from src.infrastructure.context_factory import make_training_context
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar100,
)
from src.infrastructure.constants import BASELINE_MODELS_PATH
from src.model_resnet50_cifars.model_resnet50_variable_class import ModelResnet50Variable


def train_dense_resnet50_cifar100(model: ModelResnet50Variable) -> ModelResnet50Variable:
    dataset = DatasetSmallContext(dataset=DatasetSmallType.CIFAR100, configs=dataset_context_configs_cifar100())
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120])

    ctx = make_training_context(model, dataset, optimizer, criterion)

    EPOCHS = 160
    acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        ctx.train_one_epoch()
        acc, _ = ctx.evaluate()
        scheduler.step()
        print(f"  Epoch {epoch}/{EPOCHS}  acc={acc:.2f}%")

    save_model_entire_dict(model, f"resnet50_cifar100_alpha{model.alpha}_acc{acc:.1f}", BASELINE_MODELS_PATH)
    print(f"Saved dense model  acc={acc:.2f}%")
    return model
