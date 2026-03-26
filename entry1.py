from src.experiments.lenet_variable_mnist_nplh import experiment_lenet_variable_NPLH, ModelSpec
from src.infrastructure.policies.pruning_policy import RandomPruningPolicy
from src.infrastructure.policies.training_convergence_policy import FixedEpochsConvergencePolicy
from src.infrastructure.policies.saliency_measurement_policy import (
    MagnitudeSaliencyMeasurementPolicy,
    TaylorSaliencyMeasurementPolicy,
    GradientSaliencyMeasurementPolicy,
)
from src.infrastructure.policies.nplh_stopping_policy import NPLHDensityLimitStoppingPolicy


def experiment_random_pruning_lenet_mnist():
    models_to_run = [
        ModelSpec(alpha=0.5, loaded_model_name="lenet_alpha0.5_acc97.7"),
        ModelSpec(alpha=1.0, loaded_model_name="lenet_alpha1.0_acc98.0"),
        ModelSpec(alpha=2.0, loaded_model_name="lenet_alpha2.0_acc98.1"),
    ]
    experiment_lenet_variable_NPLH(
        models_to_run=models_to_run,
        pruning_policy=RandomPruningPolicy(pruning_rate=0.1),
        convergence_policy=FixedEpochsConvergencePolicy(epochs=5),
        saliency_policies=[
            MagnitudeSaliencyMeasurementPolicy(),
            TaylorSaliencyMeasurementPolicy(),
            GradientSaliencyMeasurementPolicy(),
        ],
        stopping_policy=NPLHDensityLimitStoppingPolicy(density_threshold=0.1),
    )


if __name__ == "__main__":
    experiment_random_pruning_lenet_mnist()
