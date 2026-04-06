"""
Fixed-epoch NPLH experiments: converge once, then prune + fine-tune for N epochs.

Flow per experiment:
  1. Load / train-from-scratch a dense baseline.
  2. Warmup: train until convergence (UntilConvergencePolicy, same as retrain runner).
  3. Pruning loop: prune → FixedEpochsConvergencePolicy(N, lr=LR_FIXED) → measure.

LR notes:
  - LeNet    loop LR = 1e-4  (10× lower than warmup; LeNet training is noisy)
  - ResNet50 loop LR = unset  (Adam state carries over from warmup at its final LR)
  - VGG-19   loop LR = unset  (same)

Experiment names use the suffix _fixedepoch to avoid colliding with _retrain data.

    python runners/run_nplh_fixedepoch_experiments.py
    python runners/run_nplh_fixedepoch_experiments.py --gpus 0,1,2
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import datetime
import os
import random
import string
import time

from src.experiments.lenet_variable_mnist_nplh import experiment_lenet_variable_NPLH
from src.experiments.lenet_variable_mnist_nplh import ModelSpec as LenetSpec
from src.experiments.resnet50_variable_cifar10_nplh import experiment_resnet50_variable_cifar10_NPLH
from src.experiments.resnet50_variable_cifar10_nplh import ModelSpec as R50Spec
from src.experiments.vgg19_variable_cifar100_nplh import experiment_vgg19_variable_cifar100_NPLH
from src.experiments.vgg19_variable_cifar100_nplh import ModelSpec as VGGSpec
from src.infrastructure.policies.nplh_stopping_policy import NPLHDensityLimitStoppingPolicy
from src.infrastructure.policies.pruning_policy import (
    MagnitudePruningPolicy,
    RandomPruningPolicy,
    TaylorPruningPolicy,
)
from src.infrastructure.policies.saliency_measurement_policy import (
    GradientSaliencyMeasurementPolicy,
    HessianSaliencyMeasurementPolicy,
    MagnitudeSaliencyMeasurementPolicy,
    NeuronActivationFrequencyPolicy,
    TaylorSaliencyMeasurementPolicy,
)
from src.infrastructure.policies.training_convergence_policy import (
    FixedEpochsConvergencePolicy,
    UntilConvergencePolicy,
)
from src.plots.nplh_data import write_experiment_details
from src.infrastructure.experiment_runner_persistent import ExperimentSpec, run_experiments

ROOT = Path(__file__).parent.parent

# ── Shared constants ───────────────────────────────────────────────────────────

LENET_RATE  = 10.0
CIFAR_RATE  = 20.0
LENET_STOP  = 0.5    # stop when density ≤ 0.5%
CIFAR_STOP  = 0.05   # stop when density ≤ 0.05%

# Warmup: same convergence config as the retrain runner
LR_WARMUP       = 1e-3
CONV_WINDOW     = 5
CONV_MAX_EPOCHS = 100
CONV_REL_TOL    = 0.25

# Fixed-epoch loop
FIXED_EPOCHS        = 5
LR_FIXED_LENET      = 1e-4   # 10× lower — LeNet training is noisy
LR_FIXED_CIFAR      = None   # no override — Adam state carries over from warmup

R50_CIFAR10_PRETRAINED  = "resnet50_cifar10_alpha1.0_acc94.2"
VGG_CIFAR100_PRETRAINED = "vgg19_cifar100_alpha1.0_acc73.8"


# ── Policy factories ───────────────────────────────────────────────────────────

def _all_saliency():
    return [
        MagnitudeSaliencyMeasurementPolicy(),
        GradientSaliencyMeasurementPolicy(),
        TaylorSaliencyMeasurementPolicy(),
        HessianSaliencyMeasurementPolicy(),
        NeuronActivationFrequencyPolicy(),
    ]


def _warmup_policy():
    """Shared warmup: train until convergence at LR_WARMUP (same as retrain runner)."""
    return UntilConvergencePolicy(
        window=CONV_WINDOW,
        max_epochs=CONV_MAX_EPOCHS,
        initial_lr=LR_WARMUP,
        rel_tol=CONV_REL_TOL,
    )


def _lenet_fixed_policy():
    """LeNet loop: 5 epochs at 1e-4 (lower LR to stabilise noisy LeNet training)."""
    return FixedEpochsConvergencePolicy(FIXED_EPOCHS, lr=LR_FIXED_LENET)


def _cifar_fixed_policy():
    """ResNet50 / VGG-19 loop: 5 epochs, Adam LR left as-is from warmup."""
    return FixedEpochsConvergencePolicy(FIXED_EPOCHS, lr=LR_FIXED_CIFAR)


# ── Details formatter ──────────────────────────────────────────────────────────

def _fmt_details(
    exp_name: str,
    model: str,
    dataset: str,
    pruning_cls: str,
    rate: float,
    stop_threshold: float,
    fixed_lr: float | None,
    description: str,
) -> str:
    lr_note = f"{fixed_lr:.0e}" if fixed_lr is not None else "unchanged from warmup"
    return (
        f"Experiment:   {exp_name}\n"
        f"Date:         {datetime.date.today()}\n"
        f"Model:        {model} (alpha=1.0)\n"
        f"Dataset:      {dataset}\n"
        f"Pruning:      {pruning_cls}  rate={rate}%\n"
        f"Warmup:       UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS}, lr={LR_WARMUP:.0e})\n"
        f"Loop:         FixedEpochsConvergencePolicy(epochs={FIXED_EPOCHS}, lr={lr_note})\n"
        f"Stopping:     NPLHDensityLimitStoppingPolicy  threshold={stop_threshold}%\n"
        f"Saliency:     Magnitude, Gradient, Taylor, Hessian, NeuronActivationFreq\n"
        f"Description:  {description}\n"
    )


# ── Per-architecture helpers ───────────────────────────────────────────────────

def _run_lenet(exp_name: str, pruning, details: str) -> None:
    write_experiment_details(exp_name, details)
    experiment_lenet_variable_NPLH(
        models_to_run=[LenetSpec(alpha=1.0, loaded_model_name=None)],
        pruning_policy=pruning,
        convergence_policy=_lenet_fixed_policy(),
        warmup_policy=_warmup_policy(),
        saliency_policies=_all_saliency(),
        stopping_policy=NPLHDensityLimitStoppingPolicy(LENET_STOP),
        experiment_name=exp_name,
    )


def _run_resnet50(exp_name: str, pruning, details: str) -> None:
    write_experiment_details(exp_name, details)
    experiment_resnet50_variable_cifar10_NPLH(
        models_to_run=[R50Spec(alpha=1.0, loaded_model_name=R50_CIFAR10_PRETRAINED)],
        pruning_policy=pruning,
        convergence_policy=_cifar_fixed_policy(),
        warmup_policy=_warmup_policy(),
        saliency_policies=_all_saliency(),
        stopping_policy=NPLHDensityLimitStoppingPolicy(CIFAR_STOP),
        experiment_name=exp_name,
    )


def _run_vgg19(exp_name: str, pruning, details: str) -> None:
    write_experiment_details(exp_name, details)
    experiment_vgg19_variable_cifar100_NPLH(
        models_to_run=[VGGSpec(alpha=1.0, loaded_model_name=VGG_CIFAR100_PRETRAINED)],
        pruning_policy=pruning,
        convergence_policy=_cifar_fixed_policy(),
        warmup_policy=_warmup_policy(),
        saliency_policies=_all_saliency(),
        stopping_policy=NPLHDensityLimitStoppingPolicy(CIFAR_STOP),
        experiment_name=exp_name,
    )


# ── 9 experiment callables (module-level, picklable) ──────────────────────────

def _lenet_random_fixedepoch() -> None:
    _run_lenet("lenet_random_fixedepoch", RandomPruningPolicy(LENET_RATE),
        _fmt_details("lenet_random_fixedepoch", "LeNet-300", "MNIST",
            "RandomPruningPolicy", LENET_RATE, LENET_STOP, LR_FIXED_LENET,
            "Random pruning, warmup to convergence then 5-epoch fixed fine-tuning at 1e-4."))

def _lenet_magnitude_fixedepoch() -> None:
    _run_lenet("lenet_magnitude_fixedepoch", MagnitudePruningPolicy(LENET_RATE),
        _fmt_details("lenet_magnitude_fixedepoch", "LeNet-300", "MNIST",
            "MagnitudePruningPolicy", LENET_RATE, LENET_STOP, LR_FIXED_LENET,
            "Magnitude pruning, warmup to convergence then 5-epoch fixed fine-tuning at 1e-4."))

def _lenet_taylor_fixedepoch() -> None:
    _run_lenet("lenet_taylor_fixedepoch", TaylorPruningPolicy(LENET_RATE),
        _fmt_details("lenet_taylor_fixedepoch", "LeNet-300", "MNIST",
            "TaylorPruningPolicy", LENET_RATE, LENET_STOP, LR_FIXED_LENET,
            "Taylor pruning, warmup to convergence then 5-epoch fixed fine-tuning at 1e-4."))


def _resnet50_random_fixedepoch() -> None:
    _run_resnet50("resnet50_random_fixedepoch", RandomPruningPolicy(CIFAR_RATE),
        _fmt_details("resnet50_random_fixedepoch", "ResNet-50", "CIFAR-10",
            "RandomPruningPolicy", CIFAR_RATE, CIFAR_STOP, LR_FIXED_CIFAR,
            "Random pruning on ResNet-50/CIFAR-10, warmup to convergence then 5-epoch fixed fine-tuning."))

def _resnet50_magnitude_fixedepoch() -> None:
    _run_resnet50("resnet50_magnitude_fixedepoch", MagnitudePruningPolicy(CIFAR_RATE),
        _fmt_details("resnet50_magnitude_fixedepoch", "ResNet-50", "CIFAR-10",
            "MagnitudePruningPolicy", CIFAR_RATE, CIFAR_STOP, LR_FIXED_CIFAR,
            "Magnitude pruning on ResNet-50/CIFAR-10, warmup to convergence then 5-epoch fixed fine-tuning."))

def _resnet50_taylor_fixedepoch() -> None:
    _run_resnet50("resnet50_taylor_fixedepoch", TaylorPruningPolicy(CIFAR_RATE),
        _fmt_details("resnet50_taylor_fixedepoch", "ResNet-50", "CIFAR-10",
            "TaylorPruningPolicy", CIFAR_RATE, CIFAR_STOP, LR_FIXED_CIFAR,
            "Taylor pruning on ResNet-50/CIFAR-10, warmup to convergence then 5-epoch fixed fine-tuning."))


def _vgg19_random_fixedepoch() -> None:
    _run_vgg19("vgg19_random_fixedepoch", RandomPruningPolicy(CIFAR_RATE),
        _fmt_details("vgg19_random_fixedepoch", "VGG-19", "CIFAR-100",
            "RandomPruningPolicy", CIFAR_RATE, CIFAR_STOP, LR_FIXED_CIFAR,
            "Random pruning on VGG-19/CIFAR-100, warmup to convergence then 5-epoch fixed fine-tuning."))

def _vgg19_magnitude_fixedepoch() -> None:
    _run_vgg19("vgg19_magnitude_fixedepoch", MagnitudePruningPolicy(CIFAR_RATE),
        _fmt_details("vgg19_magnitude_fixedepoch", "VGG-19", "CIFAR-100",
            "MagnitudePruningPolicy", CIFAR_RATE, CIFAR_STOP, LR_FIXED_CIFAR,
            "Magnitude pruning on VGG-19/CIFAR-100, warmup to convergence then 5-epoch fixed fine-tuning."))

def _vgg19_taylor_fixedepoch() -> None:
    _run_vgg19("vgg19_taylor_fixedepoch", TaylorPruningPolicy(CIFAR_RATE),
        _fmt_details("vgg19_taylor_fixedepoch", "VGG-19", "CIFAR-100",
            "TaylorPruningPolicy", CIFAR_RATE, CIFAR_STOP, LR_FIXED_CIFAR,
            "Taylor pruning on VGG-19/CIFAR-100, warmup to convergence then 5-epoch fixed fine-tuning."))


# ── Ordered experiment list ────────────────────────────────────────────────────

_ALL_FNS = [
    ("lenet_random_fixedepoch",     _lenet_random_fixedepoch,     "LeNet  random    fixedepoch"),
    ("lenet_magnitude_fixedepoch",  _lenet_magnitude_fixedepoch,  "LeNet  magnitude fixedepoch"),
    ("lenet_taylor_fixedepoch",     _lenet_taylor_fixedepoch,     "LeNet  taylor    fixedepoch"),
    ("resnet50_random_fixedepoch",  _resnet50_random_fixedepoch,  "R50    random    fixedepoch"),
    ("resnet50_magnitude_fixedepoch", _resnet50_magnitude_fixedepoch, "R50  magnitude fixedepoch"),
    ("resnet50_taylor_fixedepoch",  _resnet50_taylor_fixedepoch,  "R50    taylor    fixedepoch"),
    ("vgg19_random_fixedepoch",     _vgg19_random_fixedepoch,     "VGG19  random    fixedepoch"),
    ("vgg19_magnitude_fixedepoch",  _vgg19_magnitude_fixedepoch,  "VGG19  magnitude fixedepoch"),
    ("vgg19_taylor_fixedepoch",     _vgg19_taylor_fixedepoch,     "VGG19  taylor    fixedepoch"),
]


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs, e.g. '0,1,2'.  Defaults to all detected GPUs.",
    )
    args = parser.parse_args()

    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]

    process_id = time.strftime('%Y%m%d_%H%M_') + ''.join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )
    os.environ["NPLH_PROCESS_ID"] = process_id

    nplh_base = ROOT / "nplh_data" / process_id

    experiments = [
        ExperimentSpec(
            name        = name,
            fn          = fn,
            description = desc,
            log_path    = str(nplh_base / name / "experiment.log"),
        )
        for name, fn, desc in _ALL_FNS
    ]

    run_experiments(
        experiments = experiments,
        gpu_ids     = gpu_ids,
        log_dir     = nplh_base,
        main_log    = str(nplh_base / "main.log"),
    )


if __name__ == "__main__":
    main()
