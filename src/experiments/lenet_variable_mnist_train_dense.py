import torch
import torch.nn as nn

from src.common_files_experiments.load_save import save_model_entire_dict
from src.infrastructure.context_factory import make_training_context
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, dataset_context_configs_mnist,
)

from src.infrastructure.constants import BASELINE_MODELS_PATH, PRUNED_MODELS_PATH
from src.model_lenet.model_lenetVariable_class import ModelLenetVariable

def train_dense_lenet_mnist(model: ModelLenetVariable) -> ModelLenetVariable:
    dataset = DatasetSmallContext(dataset=DatasetSmallType.MNIST, configs=dataset_context_configs_mnist())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    ctx = make_training_context(model, dataset, optimizer, criterion)
    acc = 0.0
    EPOCHS = 25

    for epoch in range(1, EPOCHS + 1):
        ctx.train_one_epoch()
        acc, _ = ctx.evaluate()
        print(f"  Epoch {epoch}/{EPOCHS}  acc={acc:.2f}%")

    save_model_entire_dict(model, f"lenet_alpha{model.alpha}_acc{acc:.1f}", BASELINE_MODELS_PATH)

    print(f"Saved dense model  acc={acc:.2f}%")
    return model

