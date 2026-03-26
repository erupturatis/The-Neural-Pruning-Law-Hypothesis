import torch
import torch.nn as nn
from dataclasses import dataclass

from src.common_files_experiments.load_save import load_model_entire_dict
from src.experiments.resnet50_variable_cifar10_train_dense import train_dense_resnet50_cifar10
from src.infrastructure.context_factory import make_training_context
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, dataset_context_configs_cifar10,
)
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device
from src.infrastructure.policies.pruning_policy import PruningPolicy
from src.infrastructure.policies.saliency_measurement_policy import SaliencyMeasurementPolicy
from src.infrastructure.policies.training_convergence_policy import TrainingConvergencePolicy
from src.infrastructure.policies.nplh_stopping_policy import NPLHStoppingPolicy
from src.infrastructure.constants import BASELINE_MODELS_PATH
from src.model_resnet50_cifar10.model_resnet50_variable_class import ModelResnet50Variable
from src.experiments.utils import get_model_density
from src.plots.nplh_data import NplhSeries


def nplh_resnet50_cifar10(
    model: ModelResnet50Variable,
    pruning_policy: PruningPolicy,
    convergence_policy: TrainingConvergencePolicy,
    saliency_policies: list[SaliencyMeasurementPolicy],
    stopping_policy: NPLHStoppingPolicy,
):
    LR_FINETUNE = 1e-3
    MAX_ROUNDS = 1000

    dataset = DatasetSmallContext(dataset=DatasetSmallType.CIFAR10, configs=dataset_context_configs_cifar10())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    criterion = nn.CrossEntropyLoss()

    ctx = make_training_context(model, dataset, optimizer, criterion)

    series_map = {
        policy: NplhSeries(f"resnet50_cifar10_alpha{model.alpha}_{type(policy).__name__}")
        for policy in saliency_policies
    }

    for round_idx in range(1, MAX_ROUNDS + 1):
        density = get_model_density(model)
        print(f"\n=== Round {round_idx}  |  remaining={density:.3f}% ===")

        if stopping_policy.stop_experiment(ctx):
            print("Stopping policy triggered.")
            break

        print(f"  [Pruning]   applying {type(pruning_policy).__name__}...")
        pruning_policy.apply_pruning(ctx)
        density = get_model_density(model)
        print(f"  [Pruning]   done — density now {density:.3f}%")

        print(f"  [Training]  running {type(convergence_policy).__name__}...")
        acc = convergence_policy.train_until_convergence(ctx)
        print(f"  [Training]  done — accuracy={acc:.4f}")

        print(f"  [Saliency]  measuring {len(saliency_policies)} policy/policies...")
        for policy in saliency_policies:
            min_sal, avg_sal = policy.measure_saliency(ctx)
            series = series_map[policy]
            series.record(density, avg_sal, min_saliency=min_sal, accuracy=acc)
            series.save()
        print(f"  [Saliency]  done — recorded and saved")


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
) -> None:
    """Prepares each model and runs the NPLH experiment on CIFAR-10."""
    for spec in models_to_run:
        if spec.alpha is None:
            raise ValueError("Alpha cannot be None. Please provide a valid alpha value.")

        cfg = ConfigsNetworkMask(mask_apply_enabled=True, mask_training_enabled=False, weights_training_enabled=True)
        model = ModelResnet50Variable(spec.alpha, cfg).to(get_device())

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
        )
