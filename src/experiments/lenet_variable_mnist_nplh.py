import torch
import torch.nn as nn

from src.common_files_experiments.load_save import load_model_entire_dict
from src.experiments.lenet_variable_mnist_train_dense import train_dense_lenet_mnist
from src.infrastructure.context_factory import make_training_context
from src.infrastructure.dataset_context.dataset_context import DatasetSmallContext, DatasetSmallType, \
    dataset_context_configs_mnist
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device
from src.infrastructure.policies.pruning_policy import PruningPolicy
from src.infrastructure.policies.saliency_measurement_policy import SaliencyMeasurementPolicy
from src.infrastructure.policies.training_convergence_policy import TrainingConvergencePolicy
from src.infrastructure.policies.nplh_stopping_policy import NPLHStoppingPolicy
from src.infrastructure.constants import BASELINE_MODELS_PATH, PRUNED_MODELS_PATH
from src.model_lenet.model_lenetVariable_class import ModelLenetVariable
from src.experiments.utils import get_model_density


# ── Config ────────────────────────────────────────────────────────────────────

LR_FINETUNE   = 1e-3
DENSE_EPOCHS  = 60
MAX_ROUNDS    = 20

# ─── NPLH experiment ──────────────────────────────────────────────────────────

def nplh_lenet_mnist(
    model: ModelLenetVariable,
    pruning_policy: PruningPolicy,
    convergence_policy: TrainingConvergencePolicy,
    saliency_policy: SaliencyMeasurementPolicy,
    stopping_policy: NPLHStoppingPolicy,
):

    dataset = DatasetSmallContext(dataset=DatasetSmallType.MNIST, configs=dataset_context_configs_mnist())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    criterion = nn.CrossEntropyLoss()

    ctx = make_training_context(model, dataset, optimizer, criterion)

    for round_idx in range(1, MAX_ROUNDS + 1):
        remaining = get_model_density(model)
        print(f"\n=== Round {round_idx}/{MAX_ROUNDS}  |  remaining={remaining:.1f}% ===")

        if stopping_policy.stop_experiment(ctx):
            print("Stopping policy triggered.")
            break

        pruning_policy.apply_pruning(ctx)
        acc = convergence_policy.train_until_convergence(ctx)
        saliencies = saliency_policy.measure_saliency(ctx)


from dataclasses import dataclass
@dataclass
class ModelSpec:
    alpha: float
    loaded_model_name: str | None = None

def experiment_lenet_variable_NPLH(
    models_to_run: list[ModelSpec],
    pruning_policy: PruningPolicy,
    convergence_policy: TrainingConvergencePolicy,
    saliency_policy: SaliencyMeasurementPolicy,
    stopping_policy: NPLHStoppingPolicy,
) -> None:
    """Wrapper that prepares the model based on alphas/model_name and runs the experiment."""
    models = []
    for spec in models_to_run:
        if spec.alpha == None:
            raise ValueError("Alpha cannot be None. Please provide a valid alpha value.")

        cfg = ConfigsNetworkMask(mask_apply_enabled=True, mask_training_enabled=False, weights_training_enabled=True)
        model = ModelLenetVariable(spec.alpha, cfg).to(get_device())

        if spec.loaded_model_name == None:
            train_dense_lenet_mnist(model)
        else:
            load_model_entire_dict(model, spec.loaded_model_name, BASELINE_MODELS_PATH)

        models.append(model)

    for model in models:
        nplh_lenet_mnist(
            model=model,
            pruning_policy=pruning_policy,
            convergence_policy=convergence_policy,
            saliency_policy=saliency_policy,
            stopping_policy=stopping_policy
        )
