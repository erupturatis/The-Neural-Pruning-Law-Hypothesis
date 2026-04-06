"""
Run all 30 NPLH experiments in parallel across GPU slots.

    python runners/run_nplh_experiments.py                  # all detected GPUs
    python runners/run_nplh_experiments.py --gpus 0,1,2     # restrict to 3 GPUs

30 experiments = 10 pruning configurations × 3 architectures:
  - lenet_mnist        (LeNet-300,   MNIST,    rate=5%,  stop=0.1%)
  - resnet50_cifar10   (ResNet-50,   CIFAR-10,  rate=10%, stop=0.01%)
  - vgg19_cifar100     (VGG-19,      CIFAR-100, rate=10%, stop=0.01%)

Each configuration:
  random_static, random_retrain,
  magnitude_static, magnitude_retrain,
  fisher_static, fisher_retrain,
  taylor_static, taylor_retrain,
  gradient_static, gradient_retrain

Output lands in  nplh_data/<process_id>/<experiment_name>/
  details.txt       — experiment parameters and description
  experiment.log    — full stdout/stderr for that run
  *.csv             — one CSV per saliency policy

nplh_data/<process_id>/main.log captures the dispatcher overview.

All 30 experiments share the same <process_id> (set via NPLH_PROCESS_ID env
var before child processes are spawned) so output is grouped in one folder.
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
    GradientPruningPolicy,
    HessianPruningPolicy,
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

LR_FINETUNE = 1e-3   # Adam LR reset at the start of every convergence round

# Convergence: 4 phases (3 LR step-downs by /10); patience on train-loss
CONV_WINDOW     = 5
CONV_MAX_EPOCHS = 100
CONV_REL_TOL    = 0.25   # 10% relative improvement required to reset patience

# Pre-trained baselines (CIFAR networks only; LeNet trains from scratch)
R50_CIFAR10_PRETRAINED  = "resnet50_cifar10_alpha1.0_acc94.2"
VGG_CIFAR100_PRETRAINED = "vgg19_cifar100_alpha1.0_acc73.8"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _all_saliency():
    return [
        MagnitudeSaliencyMeasurementPolicy(),
        GradientSaliencyMeasurementPolicy(),
        TaylorSaliencyMeasurementPolicy(),
        HessianSaliencyMeasurementPolicy(),
        NeuronActivationFrequencyPolicy(),
    ]


def _static_policy():
    return FixedEpochsConvergencePolicy(0)


def _retrain_policy():
    return UntilConvergencePolicy(
        window=CONV_WINDOW,
        max_epochs=CONV_MAX_EPOCHS, initial_lr=LR_FINETUNE,
        rel_tol=CONV_REL_TOL,
    )


def _fmt_details(
    exp_name: str,
    model: str,
    dataset: str,
    pruning_cls: str,
    rate: float,
    convergence_desc: str,
    stop_threshold: float,
    description: str,
) -> str:
    return (
        f"Experiment:   {exp_name}\n"
        f"Date:         {datetime.date.today()}\n"
        f"Model:        {model} (alpha=1.0)\n"
        f"Dataset:      {dataset}\n"
        f"Pruning:      {pruning_cls}  rate={rate}%\n"
        f"Convergence:  {convergence_desc}\n"
        f"Stopping:     NPLHDensityLimitStoppingPolicy  threshold={stop_threshold}%\n"
        f"Saliency:     Magnitude, Gradient, Taylor, Hessian, NeuronActivationFreq\n"
        f"Description:  {description}\n"
    )


# ── Per-architecture helpers ───────────────────────────────────────────────────

def _run_lenet(exp_name: str, pruning, convergence, details: str) -> None:
    write_experiment_details(exp_name, details)
    experiment_lenet_variable_NPLH(
        models_to_run=[LenetSpec(alpha=1.0, loaded_model_name=None)],
        pruning_policy=pruning,
        convergence_policy=convergence,
        saliency_policies=_all_saliency(),
        stopping_policy=NPLHDensityLimitStoppingPolicy(LENET_STOP),
        experiment_name=exp_name,
    )


def _run_resnet50(exp_name: str, pruning, convergence, details: str) -> None:
    write_experiment_details(exp_name, details)
    experiment_resnet50_variable_cifar10_NPLH(
        models_to_run=[R50Spec(alpha=1.0, loaded_model_name=R50_CIFAR10_PRETRAINED)],
        pruning_policy=pruning,
        convergence_policy=convergence,
        saliency_policies=_all_saliency(),
        stopping_policy=NPLHDensityLimitStoppingPolicy(CIFAR_STOP),
        experiment_name=exp_name,
    )


def _run_vgg19(exp_name: str, pruning, convergence, details: str) -> None:
    write_experiment_details(exp_name, details)
    experiment_vgg19_variable_cifar100_NPLH(
        models_to_run=[VGGSpec(alpha=1.0, loaded_model_name=VGG_CIFAR100_PRETRAINED)],
        pruning_policy=pruning,
        convergence_policy=convergence,
        saliency_policies=_all_saliency(),
        stopping_policy=NPLHDensityLimitStoppingPolicy(CIFAR_STOP),
        experiment_name=exp_name,
    )


# ── 30 experiment callables ────────────────────────────────────────────────────
# Each must be a module-level function (picklable by multiprocessing.spawn).

# ── LeNet / MNIST ──────────────────────────────────────────────────────────────

def _lenet_random_static() -> None:
    _run_lenet("lenet_random_static", RandomPruningPolicy(LENET_RATE), _static_policy(),
        _fmt_details("lenet_random_static", "LeNet-300", "MNIST",
            "RandomPruningPolicy", LENET_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", LENET_STOP,
            "Random pruning without retraining. Saliency measured at each density level as weights are randomly removed with no fine-tuning."))

def _lenet_random_retrain() -> None:
    _run_lenet("lenet_random_retrain", RandomPruningPolicy(LENET_RATE), _retrain_policy(),
        _fmt_details("lenet_random_retrain", "LeNet-300", "MNIST",
            "RandomPruningPolicy", LENET_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", LENET_STOP,
            "Random pruning with retraining to convergence after each round."))

def _lenet_magnitude_static() -> None:
    _run_lenet("lenet_magnitude_static", MagnitudePruningPolicy(LENET_RATE), _static_policy(),
        _fmt_details("lenet_magnitude_static", "LeNet-300", "MNIST",
            "MagnitudePruningPolicy", LENET_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", LENET_STOP,
            "Magnitude pruning without retraining. Smallest-weight pruning schedule measured at each density without fine-tuning."))

def _lenet_magnitude_retrain() -> None:
    _run_lenet("lenet_magnitude_retrain", MagnitudePruningPolicy(LENET_RATE), _retrain_policy(),
        _fmt_details("lenet_magnitude_retrain", "LeNet-300", "MNIST",
            "MagnitudePruningPolicy", LENET_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", LENET_STOP,
            "Magnitude pruning with retraining to convergence after each round."))

# def _lenet_fisher_static() -> None:
#     _run_lenet("lenet_fisher_static", HessianPruningPolicy(LENET_RATE), _static_policy(),
#         _fmt_details("lenet_fisher_static", "LeNet-300", "MNIST",
#             "HessianPruningPolicy (diagonal Fisher)", LENET_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", LENET_STOP,
#             "Fisher/Hessian pruning without retraining. Diagonal Fisher approximation (0.5 * H_ii * w_i^2) with no fine-tuning."))

# def _lenet_fisher_retrain() -> None:
#     _run_lenet("lenet_fisher_retrain", HessianPruningPolicy(LENET_RATE), _retrain_policy(),
#         _fmt_details("lenet_fisher_retrain", "LeNet-300", "MNIST",
#             "HessianPruningPolicy (diagonal Fisher)", LENET_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", LENET_STOP,
#             "Fisher/Hessian pruning with retraining to convergence after each round."))

def _lenet_taylor_static() -> None:
    _run_lenet("lenet_taylor_static", TaylorPruningPolicy(LENET_RATE), _static_policy(),
        _fmt_details("lenet_taylor_static", "LeNet-300", "MNIST",
            "TaylorPruningPolicy", LENET_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", LENET_STOP,
            "First-order Taylor pruning (|w * grad|) without retraining."))

def _lenet_taylor_retrain() -> None:
    _run_lenet("lenet_taylor_retrain", TaylorPruningPolicy(LENET_RATE), _retrain_policy(),
        _fmt_details("lenet_taylor_retrain", "LeNet-300", "MNIST",
            "TaylorPruningPolicy", LENET_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", LENET_STOP,
            "First-order Taylor pruning (|w * grad|) with retraining to convergence after each round."))

# def _lenet_gradient_static() -> None:
#     _run_lenet("lenet_gradient_static", GradientPruningPolicy(LENET_RATE), _static_policy(),
#         _fmt_details("lenet_gradient_static", "LeNet-300", "MNIST",
#             "GradientPruningPolicy", LENET_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", LENET_STOP,
#             "Gradient-magnitude pruning (|grad|) without retraining."))

# def _lenet_gradient_retrain() -> None:
#     _run_lenet("lenet_gradient_retrain", GradientPruningPolicy(LENET_RATE), _retrain_policy(),
#         _fmt_details("lenet_gradient_retrain", "LeNet-300", "MNIST",
#             "GradientPruningPolicy", LENET_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", LENET_STOP,
#             "Gradient-magnitude pruning (|grad|) with retraining to convergence after each round."))


# ── ResNet-50 / CIFAR-10 ───────────────────────────────────────────────────────

def _resnet50_random_static() -> None:
    _run_resnet50("resnet50_random_static", RandomPruningPolicy(CIFAR_RATE), _static_policy(),
        _fmt_details("resnet50_random_static", "ResNet-50", "CIFAR-10",
            "RandomPruningPolicy", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
            "Random pruning without retraining on pretrained ResNet-50/CIFAR-10."))

def _resnet50_random_retrain() -> None:
    _run_resnet50("resnet50_random_retrain", RandomPruningPolicy(CIFAR_RATE), _retrain_policy(),
        _fmt_details("resnet50_random_retrain", "ResNet-50", "CIFAR-10",
            "RandomPruningPolicy", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
            "Random pruning with retraining to convergence on pretrained ResNet-50/CIFAR-10."))

def _resnet50_magnitude_static() -> None:
    _run_resnet50("resnet50_magnitude_static", MagnitudePruningPolicy(CIFAR_RATE), _static_policy(),
        _fmt_details("resnet50_magnitude_static", "ResNet-50", "CIFAR-10",
            "MagnitudePruningPolicy", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
            "Magnitude pruning without retraining on pretrained ResNet-50/CIFAR-10."))

def _resnet50_magnitude_retrain() -> None:
    _run_resnet50("resnet50_magnitude_retrain", MagnitudePruningPolicy(CIFAR_RATE), _retrain_policy(),
        _fmt_details("resnet50_magnitude_retrain", "ResNet-50", "CIFAR-10",
            "MagnitudePruningPolicy", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
            "Magnitude pruning with retraining to convergence on pretrained ResNet-50/CIFAR-10."))

# def _resnet50_fisher_static() -> None:
#     _run_resnet50("resnet50_fisher_static", HessianPruningPolicy(CIFAR_RATE), _static_policy(),
#         _fmt_details("resnet50_fisher_static", "ResNet-50", "CIFAR-10",
#             "HessianPruningPolicy (diagonal Fisher)", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
#             "Fisher/Hessian pruning without retraining on pretrained ResNet-50/CIFAR-10."))

# def _resnet50_fisher_retrain() -> None:
#     _run_resnet50("resnet50_fisher_retrain", HessianPruningPolicy(CIFAR_RATE), _retrain_policy(),
#         _fmt_details("resnet50_fisher_retrain", "ResNet-50", "CIFAR-10",
#             "HessianPruningPolicy (diagonal Fisher)", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
#             "Fisher/Hessian pruning with retraining to convergence on pretrained ResNet-50/CIFAR-10."))

def _resnet50_taylor_static() -> None:
    _run_resnet50("resnet50_taylor_static", TaylorPruningPolicy(CIFAR_RATE), _static_policy(),
        _fmt_details("resnet50_taylor_static", "ResNet-50", "CIFAR-10",
            "TaylorPruningPolicy", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
            "Taylor pruning without retraining on pretrained ResNet-50/CIFAR-10."))

def _resnet50_taylor_retrain() -> None:
    _run_resnet50("resnet50_taylor_retrain", TaylorPruningPolicy(CIFAR_RATE), _retrain_policy(),
        _fmt_details("resnet50_taylor_retrain", "ResNet-50", "CIFAR-10",
            "TaylorPruningPolicy", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
            "Taylor pruning with retraining to convergence on pretrained ResNet-50/CIFAR-10."))

# def _resnet50_gradient_static() -> None:
#     _run_resnet50("resnet50_gradient_static", GradientPruningPolicy(CIFAR_RATE), _static_policy(),
#         _fmt_details("resnet50_gradient_static", "ResNet-50", "CIFAR-10",
#             "GradientPruningPolicy", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
#             "Gradient pruning without retraining on pretrained ResNet-50/CIFAR-10."))

# def _resnet50_gradient_retrain() -> None:
#     _run_resnet50("resnet50_gradient_retrain", GradientPruningPolicy(CIFAR_RATE), _retrain_policy(),
#         _fmt_details("resnet50_gradient_retrain", "ResNet-50", "CIFAR-10",
#             "GradientPruningPolicy", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
#             "Gradient pruning with retraining to convergence on pretrained ResNet-50/CIFAR-10."))


# ── VGG-19 / CIFAR-100 ────────────────────────────────────────────────────────

def _vgg19_random_static() -> None:
    _run_vgg19("vgg19_random_static", RandomPruningPolicy(CIFAR_RATE), _static_policy(),
        _fmt_details("vgg19_random_static", "VGG-19", "CIFAR-100",
            "RandomPruningPolicy", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
            "Random pruning without retraining on pretrained VGG-19/CIFAR-100."))

def _vgg19_random_retrain() -> None:
    _run_vgg19("vgg19_random_retrain", RandomPruningPolicy(CIFAR_RATE), _retrain_policy(),
        _fmt_details("vgg19_random_retrain", "VGG-19", "CIFAR-100",
            "RandomPruningPolicy", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
            "Random pruning with retraining to convergence on pretrained VGG-19/CIFAR-100."))

def _vgg19_magnitude_static() -> None:
    _run_vgg19("vgg19_magnitude_static", MagnitudePruningPolicy(CIFAR_RATE), _static_policy(),
        _fmt_details("vgg19_magnitude_static", "VGG-19", "CIFAR-100",
            "MagnitudePruningPolicy", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
            "Magnitude pruning without retraining on pretrained VGG-19/CIFAR-100."))

def _vgg19_magnitude_retrain() -> None:
    _run_vgg19("vgg19_magnitude_retrain", MagnitudePruningPolicy(CIFAR_RATE), _retrain_policy(),
        _fmt_details("vgg19_magnitude_retrain", "VGG-19", "CIFAR-100",
            "MagnitudePruningPolicy", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
            "Magnitude pruning with retraining to convergence on pretrained VGG-19/CIFAR-100."))

# def _vgg19_fisher_static() -> None:
#     _run_vgg19("vgg19_fisher_static", HessianPruningPolicy(CIFAR_RATE), _static_policy(),
#         _fmt_details("vgg19_fisher_static", "VGG-19", "CIFAR-100",
#             "HessianPruningPolicy (diagonal Fisher)", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
#             "Fisher/Hessian pruning without retraining on pretrained VGG-19/CIFAR-100."))

# def _vgg19_fisher_retrain() -> None:
#     _run_vgg19("vgg19_fisher_retrain", HessianPruningPolicy(CIFAR_RATE), _retrain_policy(),
#         _fmt_details("vgg19_fisher_retrain", "VGG-19", "CIFAR-100",
#             "HessianPruningPolicy (diagonal Fisher)", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
#             "Fisher/Hessian pruning with retraining to convergence on pretrained VGG-19/CIFAR-100."))

def _vgg19_taylor_static() -> None:
    _run_vgg19("vgg19_taylor_static", TaylorPruningPolicy(CIFAR_RATE), _static_policy(),
        _fmt_details("vgg19_taylor_static", "VGG-19", "CIFAR-100",
            "TaylorPruningPolicy", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
            "Taylor pruning without retraining on pretrained VGG-19/CIFAR-100."))

def _vgg19_taylor_retrain() -> None:
    _run_vgg19("vgg19_taylor_retrain", TaylorPruningPolicy(CIFAR_RATE), _retrain_policy(),
        _fmt_details("vgg19_taylor_retrain", "VGG-19", "CIFAR-100",
            "TaylorPruningPolicy", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
            "Taylor pruning with retraining to convergence on pretrained VGG-19/CIFAR-100."))

# def _vgg19_gradient_static() -> None:
#     _run_vgg19("vgg19_gradient_static", GradientPruningPolicy(CIFAR_RATE), _static_policy(),
#         _fmt_details("vgg19_gradient_static", "VGG-19", "CIFAR-100",
#             "GradientPruningPolicy", CIFAR_RATE, "FixedEpochsConvergencePolicy(epochs=0) — no retraining", CIFAR_STOP,
#             "Gradient pruning without retraining on pretrained VGG-19/CIFAR-100."))

# def _vgg19_gradient_retrain() -> None:
#     _run_vgg19("vgg19_gradient_retrain", GradientPruningPolicy(CIFAR_RATE), _retrain_policy(),
#         _fmt_details("vgg19_gradient_retrain", "VGG-19", "CIFAR-100",
#             "GradientPruningPolicy", CIFAR_RATE, f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})", CIFAR_STOP,
#             "Gradient pruning with retraining to convergence on pretrained VGG-19/CIFAR-100."))


# ── Ordered experiment list ────────────────────────────────────────────────────
# Grouped by architecture so similar-cost experiments interleave across GPUs.

_ALL_FNS = [
    # LeNet (cheap — can share a GPU or run on CPU)
    ("lenet_random_static",       _lenet_random_static,       "LeNet  random    static"),
    ("lenet_random_retrain",      _lenet_random_retrain,      "LeNet  random    retrain"),
    ("lenet_magnitude_static",    _lenet_magnitude_static,    "LeNet  magnitude static"),
    ("lenet_magnitude_retrain",   _lenet_magnitude_retrain,   "LeNet  magnitude retrain"),
    # ("lenet_fisher_static",       _lenet_fisher_static,       "LeNet  fisher    static"),
    # ("lenet_fisher_retrain",      _lenet_fisher_retrain,      "LeNet  fisher    retrain"),
    ("lenet_taylor_static",       _lenet_taylor_static,       "LeNet  taylor    static"),
    ("lenet_taylor_retrain",      _lenet_taylor_retrain,      "LeNet  taylor    retrain"),
    # ("lenet_gradient_static",     _lenet_gradient_static,     "LeNet  gradient  static"),
    # ("lenet_gradient_retrain",    _lenet_gradient_retrain,    "LeNet  gradient  retrain"),
    # ResNet-50 (medium cost)
    ("resnet50_random_static",    _resnet50_random_static,    "R50    random    static"),
    ("resnet50_random_retrain",   _resnet50_random_retrain,   "R50    random    retrain"),
    ("resnet50_magnitude_static", _resnet50_magnitude_static, "R50    magnitude static"),
    ("resnet50_magnitude_retrain",_resnet50_magnitude_retrain,"R50    magnitude retrain"),
    # ("resnet50_fisher_static",    _resnet50_fisher_static,    "R50    fisher    static"),
    # ("resnet50_fisher_retrain",   _resnet50_fisher_retrain,   "R50    fisher    retrain"),
    ("resnet50_taylor_static",    _resnet50_taylor_static,    "R50    taylor    static"),
    ("resnet50_taylor_retrain",   _resnet50_taylor_retrain,   "R50    taylor    retrain"),
    # ("resnet50_gradient_static",  _resnet50_gradient_static,  "R50    gradient  static"),
    # ("resnet50_gradient_retrain", _resnet50_gradient_retrain, "R50    gradient  retrain"),
    # VGG-19 (heaviest)
    ("vgg19_random_static",       _vgg19_random_static,       "VGG19  random    static"),
    ("vgg19_random_retrain",      _vgg19_random_retrain,      "VGG19  random    retrain"),
    ("vgg19_magnitude_static",    _vgg19_magnitude_static,    "VGG19  magnitude static"),
    ("vgg19_magnitude_retrain",   _vgg19_magnitude_retrain,   "VGG19  magnitude retrain"),
    # ("vgg19_fisher_static",       _vgg19_fisher_static,       "VGG19  fisher    static"),
    # ("vgg19_fisher_retrain",      _vgg19_fisher_retrain,      "VGG19  fisher    retrain"),
    ("vgg19_taylor_static",       _vgg19_taylor_static,       "VGG19  taylor    static"),
    ("vgg19_taylor_retrain",      _vgg19_taylor_retrain,      "VGG19  taylor    retrain"),
    # ("vgg19_gradient_static",     _vgg19_gradient_static,     "VGG19  gradient  static"),
    # ("vgg19_gradient_retrain",    _vgg19_gradient_retrain,    "VGG19  gradient  retrain"),
]


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs to use, e.g. '0,1,2'.  Defaults to all detected GPUs.",
    )
    args = parser.parse_args()

    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]

    # Generate one process ID shared by all 30 child processes so their output
    # lands under a single nplh_data/<process_id>/ folder.
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
        experiments    = experiments,
        gpu_ids        = gpu_ids,
        log_dir        = nplh_base,
        main_log       = str(nplh_base / "main.log"),
    )


if __name__ == "__main__":
    main()
