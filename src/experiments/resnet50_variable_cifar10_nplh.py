import torch
import torch.nn as nn
from dataclasses import dataclass

from src.common_files_experiments.load_save import load_model_entire_dict
from src.experiments.resnet50_variable_cifar10_train_dense import train_dense_resnet50_cifar10
from src.infrastructure.context_factory import make_training_context
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar10,
)
from src.infrastructure.layers import ConfigsNetworkMask, get_total_and_remaining_params
from src.infrastructure.others import get_device
from src.infrastructure.policies.pruning_policy import PruningPolicy
from src.infrastructure.policies.saliency_measurement_policy import SaliencyMeasurementPolicy, compute_network_state
from src.infrastructure.policies.training_convergence_policy import TrainingConvergencePolicy
from src.infrastructure.policies.nplh_stopping_policy import NPLHStoppingPolicy
from src.infrastructure.constants import BASELINE_MODELS_PATH
from src.model_resnet50_cifars.model_resnet50_variable_class import ModelResnet50Variable
import time
from src.experiments.utils import get_model_density, timed, log_layer_densities
from src.plots.nplh_data import NplhSeries


def nplh_resnet50_cifar10(
    model: ModelResnet50Variable,
    pruning_policy: PruningPolicy,
    convergence_policy: TrainingConvergencePolicy,
    saliency_policies: list[SaliencyMeasurementPolicy],
    stopping_policy: NPLHStoppingPolicy,
    experiment_name: str,
    warmup_policy: TrainingConvergencePolicy | None = None,
):
    LR_FINETUNE = 1e-3
    MAX_ROUNDS = 1000

    dataset = DatasetSmallContext(dataset=DatasetSmallType.CIFAR10, configs=dataset_context_configs_cifar10())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    criterion = nn.CrossEntropyLoss()

    ctx = make_training_context(model, dataset, optimizer, criterion)
    total_params, _ = get_total_and_remaining_params(model)

    series_map = {
        policy: NplhSeries(
            f"resnet50_cifar10_alpha{model.alpha}_{type(policy).__name__}",
            experiment_folder=experiment_name,
        )
        for policy in saliency_policies
    }

    # ── Warmup ────────────────────────────────────────────────────────────────
    _warmup = warmup_policy if warmup_policy is not None else convergence_policy
    print(f"\n=== Warmup  |  {time.strftime('%H:%M:%S')} ===")
    with timed() as t:
        _warmup.train_until_convergence(ctx)
        acc_w, _ = ctx.evaluate()
    print(f"=== Warmup done — acc={acc_w:.4f}  ({t}) ===\n")

    for round_idx in range(1, MAX_ROUNDS + 1):
        density = get_model_density(model)
        print(f"\n=== Round {round_idx}  |  {time.strftime('%H:%M:%S')}  |  remaining={density:.3f}% ===")

        if stopping_policy.stop_experiment(ctx):
            print("Stopping policy triggered.")
            break

        print(f"  [Pruning]   applying {type(pruning_policy).__name__}...")
        with timed() as t:
            pruning_policy.apply_pruning(ctx)
        density = get_model_density(model)
        print(f"  [Pruning]   done — density now {density:.3f}%  ({t})")
        log_layer_densities(model)

        print(f"  [Training]  running {type(convergence_policy).__name__}...")
        with timed() as t:
            acc = convergence_policy.train_until_convergence(ctx)
            accuracy, test_loss = ctx.evaluate()
            _, train_loss = ctx.evaluate_train()
        print(f"  [Training]  done — accuracy={acc:.4f}  test_loss={test_loss:.6f}  train_loss={train_loss:.6f}  ({t})")

        print(f"  [Saliency]  computing network state...")
        with timed() as t:
            network_state = compute_network_state(ctx)
        contributing = round(network_state.active_count / total_params * 100, 4)
        print(f"  [Saliency]  present={round(network_state.present_count / total_params * 100, 4):.4f}%  active={contributing:.4f}%  ({t})")

        print(f"  [Saliency]  measuring {len(saliency_policies)} policy/policies...")
        with timed() as t:
            for policy in saliency_policies:
                result = policy.measure_saliency(ctx, network_state)
                series = series_map[policy]
                series.record(
                    density=density,
                    contributing=contributing,
                    avg_saliency=result.avg_saliency,
                    avg_saliency_contributing=result.avg_saliency_contributing,
                    min_saliency=result.min_saliency,
                    min_saliency_contributing=result.min_saliency_contributing,
                    accuracy=acc,
                    test_loss=test_loss,
                    train_loss=train_loss,
                    epoch=ctx.epoch_count,
                )
                series.save()
        print(f"  [Saliency]  done — recorded and saved  ({t})")


@dataclass
class ModelSpec:
    alpha: float
    loaded_model_name: str | None = None


def experiment_resnet50_variable_cifar10_NPLH(
    models_to_run: list[ModelSpec],
    pruning_policy: PruningPolicy,
    convergence_policy: TrainingConvergencePolicy,
    saliency_policies: list[SaliencyMeasurementPolicy],
    stopping_policy: NPLHStoppingPolicy,
    experiment_name: str,
    warmup_policy: TrainingConvergencePolicy | None = None,
) -> None:
    """Prepares each model and runs the NPLH experiment on CIFAR-10."""
    for spec in models_to_run:
        if spec.alpha is None:
            raise ValueError("Alpha cannot be None. Please provide a valid alpha value.")

        cfg = ConfigsNetworkMask(mask_apply_enabled=True, mask_training_enabled=False, weights_training_enabled=True)
        model = ModelResnet50Variable(spec.alpha, cfg, num_classes=10).to(get_device())

        if spec.loaded_model_name is None:
            train_dense_resnet50_cifar10(model)
        else:
            load_model_entire_dict(model, spec.loaded_model_name, BASELINE_MODELS_PATH)

        nplh_resnet50_cifar10(
            model=model,
            pruning_policy=pruning_policy,
            convergence_policy=convergence_policy,
            saliency_policies=saliency_policies,
            stopping_policy=stopping_policy,
            experiment_name=experiment_name,
            warmup_policy=warmup_policy,
        )
