import torch
import torch.nn as nn
from src.infrastructure.dataset_context.dataset_context import DatasetContextAbstract
from src.infrastructure.layers import LayerComposite
from src.infrastructure.others import get_custom_model_sparsity_percent
from torch.amp import GradScaler, autocast
from src.infrastructure.training_context.training_context import \
    TrainingContextBaselineTrain
from src.infrastructure.training_display import TrainingDisplay
from src.infrastructure.wandb_functions import wandb_snapshot, wandb_snapshot_baseline


def train_mixed_baseline_debug(model: nn.Module, dataset_context: DatasetContextAbstract, training_context: TrainingContextBaselineTrain, training_display: TrainingDisplay):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()

    scaler = GradScaler('cuda')
    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()

        with autocast('cuda'):
            output = model(data)
            loss_data = criterion(output, target)

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Before backward: {name} grad: {param.grad}")

        scaler.scale(loss_data).backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"After backward: {name} grad: {param.grad}")

        scaler.step(optimizer_weights)
        scaler.update()

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Updated parameter: {name}, value: {param.data}")

        training_display.record_losses([loss_data.item()])


def train_mixed_baseline_weight_decay(model: nn.Module, dataset_context: DatasetContextAbstract, training_context: TrainingContextBaselineTrain, training_display: TrainingDisplay):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer_weights = training_context.get_optimizer_weights()

    scaler = GradScaler('cuda')
    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()

        with autocast('cuda'):
            output = model(data)
            loss_data = criterion(output, target)
            loss_decay = model.get_weight_decay()
            loss_data += loss_decay

        scaler.scale(loss_data).backward()
        scaler.step(optimizer_weights)
        scaler.update()


        training_display.record_losses([loss_data.item()])
def train_mixed_baseline(model: nn.Module, dataset_context: DatasetContextAbstract, training_context: TrainingContextBaselineTrain, training_display: TrainingDisplay):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer_weights = training_context.get_optimizer_weights()

    scaler = GradScaler('cuda')
    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()

        with autocast('cuda'):
            output = model(data)
            loss_data = criterion(output, target)

        scaler.scale(loss_data).backward()
        scaler.step(optimizer_weights)
        scaler.update()

        training_display.record_losses([loss_data.item()])

def test_baseline(model: 'LayerComposite', dataset_context: DatasetContextAbstract, epoch):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_loss = 0
    correct = 0

    with torch.no_grad():
        while dataset_context.any_data_testing_available():
            data, target = dataset_context.get_testing_data_and_labels()

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_data_len = dataset_context.get_data_testing_length()
    test_loss /= total_data_len
    accuracy = 100.0 * correct / total_data_len

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)"
    )
    wandb_snapshot_baseline(
        epoch=epoch,
        accuracy=accuracy,
        test_loss=test_loss,
    )
    return accuracy

