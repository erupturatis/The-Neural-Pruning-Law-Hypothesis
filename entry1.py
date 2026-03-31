from src.experiments.lenet_variable_mnist_nplh import experiment_lenet_variable_mnist_NPLH, ModelSpec as LeNetModelSpec
from src.experiments.resnet50_variable_cifar10_nplh import experiment_resnet50_variable_cifar10_NPLH, ModelSpec as ResNet10ModelSpec
from src.experiments.resnet50_variable_cifar100_nplh import experiment_resnet50_variable_cifar100_NPLH, ModelSpec as ResNet100ModelSpec
from src.experiments.vgg19_variable_cifar10_nplh import experiment_vgg19_variable_cifar10_NPLH, ModelSpec as VGG10ModelSpec
from src.experiments.vgg19_variable_cifar100_nplh import experiment_vgg19_variable_cifar100_NPLH, ModelSpec as VGG100ModelSpec

from src.infrastructure.policies.pruning_policy import RandomPruningPolicy, GradientPruningPolicy
from src.infrastructure.policies.training_convergence_policy import FixedEpochsConvergencePolicy
from src.infrastructure.policies.saliency_measurement_policy import (
    MagnitudeSaliencyMeasurementPolicy,
    TaylorSaliencyMeasurementPolicy,
    GradientSaliencyMeasurementPolicy,
    HyperfluxSampleEstimationSaliencyMeasurementPolicy,
)
from src.infrastructure.policies.nplh_stopping_policy import NPLHDensityLimitStoppingPolicy


# ── LeNet MNIST ───────────────────────────────────────────────────────────────

def experiment_random_pruning_lenet_mnist():
    models_to_run = [
        LeNetModelSpec(alpha=0.5, loaded_model_name="lenet_alpha0.5_acc97.7"),
    ]
    experiment_lenet_variable_mnist_NPLH(
        models_to_run=models_to_run,
        pruning_policy=RandomPruningPolicy(pruning_rate=10),
        convergence_policy=FixedEpochsConvergencePolicy(epochs=5),
        saliency_policies=[
            MagnitudeSaliencyMeasurementPolicy(),
            TaylorSaliencyMeasurementPolicy(),
            GradientSaliencyMeasurementPolicy(),
        ],
        stopping_policy=NPLHDensityLimitStoppingPolicy(density_threshold=1),
    )


def experiment_gradient_pruning_lenet_mnist():
    models_to_run = [
        LeNetModelSpec(alpha=0.5, loaded_model_name="lenet_alpha0.5_acc97.7"),
    ]
    experiment_lenet_variable_mnist_NPLH(
        models_to_run=models_to_run,
        pruning_policy=GradientPruningPolicy(pruning_rate=5),
        convergence_policy=FixedEpochsConvergencePolicy(epochs=5),
        saliency_policies=[
            GradientSaliencyMeasurementPolicy(),
        ],
        stopping_policy=NPLHDensityLimitStoppingPolicy(density_threshold=10),
    )


# ── ResNet-50 CIFAR ───────────────────────────────────────────────────────────

def experiment_resnet50_cifar10_nplh_hyperflux():
    models_to_run = [
        ResNet10ModelSpec(alpha=0.5, loaded_model_name="resnet50_cifar10_alpha0.5_acc93.5"),
    ]
    experiment_resnet50_variable_cifar10_NPLH(
        models_to_run=models_to_run,
        pruning_policy=RandomPruningPolicy(pruning_rate=10),
        convergence_policy=FixedEpochsConvergencePolicy(epochs=5),
        saliency_policies=[
            MagnitudeSaliencyMeasurementPolicy(),
            GradientSaliencyMeasurementPolicy(),
            HyperfluxSampleEstimationSaliencyMeasurementPolicy(n_samples=3, sample_fraction=0.05),
        ],
        stopping_policy=NPLHDensityLimitStoppingPolicy(density_threshold=5),
    )


def experiment_resnet50_cifar100_nplh():
    models_to_run = [
        ResNet100ModelSpec(alpha=0.25), 
    ]
    experiment_resnet50_variable_cifar100_NPLH(
        models_to_run=models_to_run,
        pruning_policy=RandomPruningPolicy(pruning_rate=20),
        convergence_policy=FixedEpochsConvergencePolicy(epochs=5),
        saliency_policies=[
            MagnitudeSaliencyMeasurementPolicy(),
            GradientSaliencyMeasurementPolicy(),
            HyperfluxSampleEstimationSaliencyMeasurementPolicy(n_samples=3, sample_fraction=0.05),
        ],
        stopping_policy=NPLHDensityLimitStoppingPolicy(density_threshold=10),
    )


# ── VGG19 CIFAR ───────────────────────────────────────────────────────────────

def experiment_vgg19_cifar10_nplh():
    models_to_run = [
        VGG10ModelSpec(alpha=0.5),
    ]
    experiment_vgg19_variable_cifar10_NPLH(
        models_to_run=models_to_run,
        pruning_policy=RandomPruningPolicy(pruning_rate=20),
        convergence_policy=FixedEpochsConvergencePolicy(epochs=5),
        saliency_policies=[
            MagnitudeSaliencyMeasurementPolicy(),
            GradientSaliencyMeasurementPolicy(),
            HyperfluxSampleEstimationSaliencyMeasurementPolicy(n_samples=3, sample_fraction=0.05),
        ],
        stopping_policy=NPLHDensityLimitStoppingPolicy(density_threshold=10),
    )


def experiment_vgg19_cifar100_nplh():
    models_to_run = [
        VGG100ModelSpec(alpha=0.5),
    ]
    experiment_vgg19_variable_cifar100_NPLH(
        models_to_run=models_to_run,
        pruning_policy=RandomPruningPolicy(pruning_rate=20),
        convergence_policy=FixedEpochsConvergencePolicy(epochs=5),
        saliency_policies=[
            MagnitudeSaliencyMeasurementPolicy(),
            GradientSaliencyMeasurementPolicy(),
            HyperfluxSampleEstimationSaliencyMeasurementPolicy(n_samples=3, sample_fraction=0.05),
        ],
        stopping_policy=NPLHDensityLimitStoppingPolicy(density_threshold=10),
    )


if __name__ == "__main__":
    # experiment_random_pruning_lenet_mnist()
    # experiment_resnet50_cifar10_nplh_hyperflux()
    # experiment_resnet50_cifar100_nplh()
    experiment_vgg19_cifar10_nplh()
    # experiment_vgg19_cifar100_nplh()
