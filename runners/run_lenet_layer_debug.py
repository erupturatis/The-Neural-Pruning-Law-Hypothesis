"""
6 LeNet/MNIST experiments to diagnose per-layer density imbalance.

2 pruning methods × 3 convergence modes:
  random_static          — random pruning, no retraining
  random_fixed5          — random pruning, 5 fixed epochs per round
  random_retrain         — random pruning, until convergence
  magnitude_static       — magnitude pruning, no retraining
  magnitude_fixed5       — magnitude pruning, 5 fixed epochs per round
  magnitude_retrain      — magnitude pruning, until convergence

All measure only MagnitudeSaliencyMeasurementPolicy.
Per-layer remaining/active % is printed each round (added to nplh_lenet_mnist).

Usage:
    python runners/run_lenet_layer_debug.py [--gpus 0,1]
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
from src.infrastructure.policies.nplh_stopping_policy import NPLHDensityLimitStoppingPolicy
from src.infrastructure.policies.pruning_policy import MagnitudePruningPolicy, RandomPruningPolicy
from src.infrastructure.policies.saliency_measurement_policy import MagnitudeSaliencyMeasurementPolicy
from src.infrastructure.policies.training_convergence_policy import (
    FixedEpochsConvergencePolicy,
    UntilConvergencePolicy,
)
from src.plots.nplh_data import write_experiment_details
from src.infrastructure.experiment_runner_persistent import ExperimentSpec, run_experiments

ROOT = Path(__file__).parent.parent

LENET_RATE = 10.0
LENET_STOP = 3   # stop at 0.2% density

LR_FINETUNE     = 1e-3
CONV_WINDOW     = 5
CONV_MAX_EPOCHS = 200
CONV_REL_TOL    = 0.2


def _saliency():
    return [MagnitudeSaliencyMeasurementPolicy()]


def _static():
    return FixedEpochsConvergencePolicy(0)


def _fixed5():
    return FixedEpochsConvergencePolicy(5)


def _retrain():
    return UntilConvergencePolicy(
        window=CONV_WINDOW,
        max_epochs=CONV_MAX_EPOCHS,
        initial_lr=LR_FINETUNE,
        rel_tol=CONV_REL_TOL,
    )


def _stopping():
    return NPLHDensityLimitStoppingPolicy(LENET_STOP)


def _fmt(exp_name, pruning_cls, rate, convergence_desc):
    return (
        f"Experiment:   {exp_name}\n"
        f"Date:         {datetime.date.today()}\n"
        f"Model:        LeNet-300 (alpha=1.0)\n"
        f"Dataset:      MNIST\n"
        f"Pruning:      {pruning_cls}  rate={rate}%\n"
        f"Convergence:  {convergence_desc}\n"
        f"Stopping:     NPLHDensityLimitStoppingPolicy  threshold={LENET_STOP}%\n"
        f"Saliency:     Magnitude only\n"
        f"Purpose:      Per-layer density debug run\n"
    )


def _run(exp_name, pruning, convergence, details):
    write_experiment_details(exp_name, details)
    experiment_lenet_variable_NPLH(
        models_to_run=[LenetSpec(alpha=1.0, loaded_model_name=None)],
        pruning_policy=pruning,
        convergence_policy=convergence,
        saliency_policies=_saliency(),
        stopping_policy=_stopping(),
        experiment_name=exp_name,
    )


def _random_static():
    _run("lenet_debug_random_static", RandomPruningPolicy(LENET_RATE), _static(),
         _fmt("lenet_debug_random_static", "RandomPruningPolicy", LENET_RATE,
              "FixedEpochsConvergencePolicy(epochs=0) — no retraining"))


def _random_fixed5():
    _run("lenet_debug_random_fixed5", RandomPruningPolicy(LENET_RATE), _fixed5(),
         _fmt("lenet_debug_random_fixed5", "RandomPruningPolicy", LENET_RATE,
              "FixedEpochsConvergencePolicy(epochs=5)"))


def _random_retrain():
    _run("lenet_debug_random_retrain", RandomPruningPolicy(LENET_RATE), _retrain(),
         _fmt("lenet_debug_random_retrain", "RandomPruningPolicy", LENET_RATE,
              f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})"))


def _magnitude_static():
    _run("lenet_debug_magnitude_static", MagnitudePruningPolicy(LENET_RATE), _static(),
         _fmt("lenet_debug_magnitude_static", "MagnitudePruningPolicy", LENET_RATE,
              "FixedEpochsConvergencePolicy(epochs=0) — no retraining"))


def _magnitude_fixed5():
    _run("lenet_debug_magnitude_fixed5", MagnitudePruningPolicy(LENET_RATE), _fixed5(),
         _fmt("lenet_debug_magnitude_fixed5", "MagnitudePruningPolicy", LENET_RATE,
              "FixedEpochsConvergencePolicy(epochs=5)"))


def _magnitude_retrain():
    _run("lenet_debug_magnitude_retrain", MagnitudePruningPolicy(LENET_RATE), _retrain(),
         _fmt("lenet_debug_magnitude_retrain", "MagnitudePruningPolicy", LENET_RATE,
              f"UntilConvergencePolicy(window={CONV_WINDOW}, max_epochs={CONV_MAX_EPOCHS})"))


_ALL_FNS = [
    ("lenet_debug_random_static",    _random_static,    "LeNet debug  random    static"),
    ("lenet_debug_random_fixed5",    _random_fixed5,    "LeNet debug  random    fixed5"),
    ("lenet_debug_random_retrain",   _random_retrain,   "LeNet debug  random    retrain"),
    ("lenet_debug_magnitude_static", _magnitude_static, "LeNet debug  magnitude static"),
    ("lenet_debug_magnitude_fixed5", _magnitude_fixed5, "LeNet debug  magnitude fixed5"),
    ("lenet_debug_magnitude_retrain",_magnitude_retrain,"LeNet debug  magnitude retrain"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1'. Defaults to all detected GPUs.")
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
            name=name,
            fn=fn,
            description=desc,
            log_path=str(nplh_base / name / "experiment.log"),
        )
        for name, fn, desc in _ALL_FNS
    ]

    run_experiments(
        experiments=experiments,
        gpu_ids=gpu_ids,
        log_dir=nplh_base,
        main_log=str(nplh_base / "main.log"),
    )


if __name__ == "__main__":
    main()
