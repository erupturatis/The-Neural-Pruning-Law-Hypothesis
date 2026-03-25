import torch
import time
import torch.nn as nn
from src.infrastructure.dataset_context.dataset_context import DatasetContextAbstract
from src.infrastructure.layers import LayerComposite, get_accumulated_flops
from src.infrastructure.others import get_custom_model_sparsity_percent, get_device
from torch.amp import GradScaler, autocast
from src.infrastructure.training_context.training_context import  \
    TrainingContextPrunedTrain
from src.infrastructure.training_display import TrainingDisplay
from src.infrastructure.wandb_functions import wandb_snapshot

def train_mixed_pruned_with_decay(model: LayerComposite, dataset_context: DatasetContextAbstract, training_context: TrainingContextPrunedTrain, training_display: TrainingDisplay):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()
    optimizer_pruning = training_context.get_optimizer_flow_mask()

    scaler = GradScaler('cuda')

    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()

        with autocast('cuda'):
            output = model(data)
            loss_remaining_weights = model.get_remaining_parameters_loss() * training_context.params.l0_gamma_scaler
            loss_weight_decay = model.get_weight_decay_only_present()

            loss_data = criterion(output, target)
            loss = loss_remaining_weights + loss_data + loss_weight_decay

        scaler.scale(loss).backward()
        scaler.step(optimizer_weights)
        scaler.step(optimizer_pruning)
        scaler.update()

        training_display.record_losses([loss_data.item(), loss_remaining_weights.item()])


def train_mixed_pruned_separated(model: LayerComposite, dataset_context: DatasetContextAbstract, training_context: TrainingContextPrunedTrain, training_display: TrainingDisplay):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()
    optimizer_pruning = training_context.get_optimizer_flow_mask()

    scaler = GradScaler('cuda')

    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()

        with autocast('cuda'):
            output = model(data)
            pruned_ts, present_ts = model.get_remaining_parameters_loss_separated()

            loss_pruned_weights = pruned_ts * training_context.params.l0_gamma_scaler
            loss_present_weights = present_ts

            loss_data = criterion(output, target)
            loss = loss_pruned_weights + loss_present_weights + loss_data

        scaler.scale(loss).backward()
        scaler.step(optimizer_weights)
        scaler.step(optimizer_pruning)
        scaler.update()

        training_display.record_losses([loss_data.item(), loss_pruned_weights.item()])


def train_mixed_pruned_imagenet_IMP(model: LayerComposite, model_module : any, dataset_context: DatasetContextAbstract, training_context: TrainingContextPrunedTrain, training_display: TrainingDisplay):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()

    scaler = GradScaler('cuda')
    epoch_start_time = time.time()
    batch_count = 0


    while dataset_context.any_data_training_available():

        iteration_start = time.time()
        data_load_start = time.time()
        data, target = dataset_context.get_training_data_and_labels()
        data_load_end = time.time()

        zero_grad_start = time.time()
        optimizer_weights.zero_grad()
        zero_grad_end = time.time()


        fwd_start = time.time()
        with autocast('cuda'):
            output = model(data)
            loss_data = criterion(output, target)
            loss = loss_data
        fwd_end = time.time()

        bwd_start = time.time()
        scaler.scale(loss).backward()
        scaler.step(optimizer_weights)
        scaler.update()
        bwd_end = time.time()
        log_start = time.time()
        training_display.record_losses([loss_data.item()])
        log_end = time.time()

        iteration_end = time.time()
        if iteration_end - iteration_start > 2.0:
            print(f"\n--- Batch {batch_count} Timings ---")
            print(f"Data load time:       {data_load_end - data_load_start:.4f} s")
            print(f"Zero grad time:       {zero_grad_end - zero_grad_start:.4f} s")
            print(f"Forward pass time:    {fwd_end - fwd_start:.4f} s")
            print(f"Backward pass time:   {bwd_end - bwd_start:.4f} s")
            print(f"Logging time:         {log_end - log_start:.4f} s")
            print(f"Iteration time:       {iteration_end - iteration_start:.4f} s")
            print("----------------------------------")

        batch_count += 1


    total_epoch_time = time.time() - epoch_start_time
    print(f"==> Finished epoch with {batch_count} batches. Epoch time: {total_epoch_time:.2f} s")



def train_mixed_pruned_imagenet(model: LayerComposite, model_module : any, dataset_context: DatasetContextAbstract, training_context: TrainingContextPrunedTrain, training_display: TrainingDisplay):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()
    optimizer_pruning = training_context.get_optimizer_flow_mask()

    scaler = GradScaler('cuda')
    epoch_start_time = time.time()
    batch_count = 0


    while dataset_context.any_data_training_available():

        iteration_start = time.time()
        data_load_start = time.time()
        data, target = dataset_context.get_training_data_and_labels()
        data_load_end = time.time()

        zero_grad_start = time.time()
        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()
        zero_grad_end = time.time()


        fwd_start = time.time()
        with autocast('cuda'):
            output = model(data)
            loss_remaining_weights = model_module.get_remaining_parameters_loss() * training_context.params.l0_gamma_scaler
            loss_data = criterion(output, target)
            loss = loss_remaining_weights + loss_data
        fwd_end = time.time()

        bwd_start = time.time()
        scaler.scale(loss).backward()
        scaler.step(optimizer_weights)
        scaler.step(optimizer_pruning)
        scaler.update()
        bwd_end = time.time()
        log_start = time.time()
        training_display.record_losses([loss_data.item(), loss_remaining_weights.item()])
        log_end = time.time()

        iteration_end = time.time()
        if iteration_end - iteration_start > 2.0:
            print(f"\n--- Batch {batch_count} Timings ---")
            print(f"Data load time:       {data_load_end - data_load_start:.4f} s")
            print(f"Zero grad time:       {zero_grad_end - zero_grad_start:.4f} s")
            print(f"Forward pass time:    {fwd_end - fwd_start:.4f} s")
            print(f"Backward pass time:   {bwd_end - bwd_start:.4f} s")
            print(f"Logging time:         {log_end - log_start:.4f} s")
            print(f"Iteration time:       {iteration_end - iteration_start:.4f} s")
            print("----------------------------------")

        batch_count += 1


    total_epoch_time = time.time() - epoch_start_time
    print(f"==> Finished epoch with {batch_count} batches. Epoch time: {total_epoch_time:.2f} s")


def test_pruned_imagenet(model: nn.Module, model_module: any, dataset_context: DatasetContextAbstract, epoch: int):
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

    remain_percent = get_custom_model_sparsity_percent(model_module)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.3f}%)"
    )
    print(
        f"Remaining parameters: {remain_percent:.2f}%"
    )
    wandb_snapshot(epoch=epoch, accuracy=accuracy, test_loss=test_loss, sparsity=remain_percent)
    return accuracy

def train_mixed_pruned(model: LayerComposite, dataset_context: DatasetContextAbstract, training_context: TrainingContextPrunedTrain, training_display: TrainingDisplay):
    model.train()
    model.to(get_device())

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_weights = training_context.get_optimizer_weights()
    optimizer_pruning = training_context.get_optimizer_flow_mask()

    scaler = GradScaler('cuda')

    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()

        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()

        with autocast('cuda'):
            output = model(data)
            loss_remaining_weights = model.get_remaining_parameters_loss() * training_context.params.l0_gamma_scaler
            loss_data = criterion(output, target)
            loss = loss_remaining_weights + loss_data

        scaler.scale(loss).backward()
        scaler.step(optimizer_weights)
        scaler.step(optimizer_pruning)
        scaler.update()

        training_display.record_losses([loss_data.item(), loss_remaining_weights.item()])

    acc = get_accumulated_flops()
    print("ACCUMULATED FLOPS", acc["counter_dense"], acc["counter_sparse"])
    print("PERCENTAGE", acc["counter_sparse"] / acc["counter_dense"] * 100 )


def test_pruned(model: nn.Module, dataset_context: DatasetContextAbstract, epoch: int, aux = None):
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

    remain_percent = get_custom_model_sparsity_percent(model)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{total_data_len} ({accuracy:.0f}%)"
    )
    print(
        f"Remaining parameters: {remain_percent:.2f}%"
    )
    wandb_snapshot(epoch=epoch, accuracy=accuracy, test_loss=test_loss, sparsity=100-remain_percent, others=aux)
    return accuracy
