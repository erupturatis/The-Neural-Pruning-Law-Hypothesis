"""
Train all dense baseline networks in parallel, one per GPU slot.

Usage:
    python run_baseline_networks.py                  # uses all detected GPUs
    python run_baseline_networks.py --gpus 0,1       # restrict to specific GPUs

Logs are written to neural_pruning_law/baselines/<exp_name>.log.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.experiments.resnet50_variable_cifar10_train_dense import train_dense_resnet50_cifar10
from src.experiments.resnet50_variable_cifar100_train_dense import train_dense_resnet50_cifar100
from src.experiments.vgg19_variable_cifar10_train_dense import train_dense_vgg19_cifar10
from src.experiments.vgg19_variable_cifar100_train_dense import train_dense_vgg19_cifar100
from src.model_resnet50_cifars.model_resnet50_variable_class import ModelResnet50Variable
from src.model_vgg19_cifars.model_vgg19_variable_class import ModelVGG19Variable
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_relu
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.others import get_device
from src.infrastructure.experiment_runner import ExperimentSpec, run_experiments

ROOT     = Path(__file__).parent.parent
LOG_DIR  = ROOT / "neural_pruning_law" / "baselines"


# ── Experiment callables ───────────────────────────────────────────────────────
# Must be module-level so multiprocessing.spawn can pickle and re-import them.

def _run_resnet50_cifar10() -> None:
    device = get_device()
    print(f"Running ResNet50 CIFAR-10 on {device}")
    cfg = ConfigsNetworkMask(mask_apply_enabled=False, mask_training_enabled=False, weights_training_enabled=True)
    model = ModelResnet50Variable(alpha=1.0, config_network_mask=cfg, num_classes=10).to(device)
    train_dense_resnet50_cifar10(model)


def _run_resnet50_cifar100() -> None:
    device = get_device()
    print(f"Running ResNet50 CIFAR-100 on {device}")
    cfg = ConfigsNetworkMask(mask_apply_enabled=False, mask_training_enabled=False, weights_training_enabled=True)
    model = ModelResnet50Variable(alpha=1.0, config_network_mask=cfg, num_classes=100).to(device)
    train_dense_resnet50_cifar100(model)


def _run_vgg19_cifar10() -> None:
    device = get_device()
    print(f"Running VGG19 CIFAR-10 on {device}")
    cfg = ConfigsNetworkMask(mask_apply_enabled=False, mask_training_enabled=False, weights_training_enabled=True)
    model = ModelVGG19Variable(alpha=1.0, config_network_mask=cfg, num_classes=10).to(device)
    train_dense_vgg19_cifar10(model)


def _run_vgg19_cifar100() -> None:
    device = get_device()
    print(f"Running VGG19 CIFAR-100 on {device}")
    configs_layers_initialization_all_kaiming_relu()
    cfg = ConfigsNetworkMask(mask_apply_enabled=False, mask_training_enabled=False, weights_training_enabled=True)
    model = ModelVGG19Variable(alpha=1.0, config_network_mask=cfg, num_classes=100).to(device)
    train_dense_vgg19_cifar100(model)


# ── Experiment registry ────────────────────────────────────────────────────────

EXPERIMENTS: list[ExperimentSpec] = [
    ExperimentSpec(
        name        = "resnet50_cifar10",
        fn          = _run_resnet50_cifar10,
        description = "ResNet-50 dense baseline — CIFAR-10",
        log_path    = str(LOG_DIR / "resnet50_cifar10.log"),
    ),
    ExperimentSpec(
        name        = "resnet50_cifar100",
        fn          = _run_resnet50_cifar100,
        description = "ResNet-50 dense baseline — CIFAR-100",
        log_path    = str(LOG_DIR / "resnet50_cifar100.log"),
    ),
    ExperimentSpec(
        name        = "vgg19_cifar10",
        fn          = _run_vgg19_cifar10,
        description = "VGG-19 dense baseline — CIFAR-10",
        log_path    = str(LOG_DIR / "vgg19_cifar10.log"),
    ),
    ExperimentSpec(
        name        = "vgg19_cifar100",
        fn          = _run_vgg19_cifar100,
        description = "VGG-19 dense baseline — CIFAR-100",
        log_path    = str(LOG_DIR / "vgg19_cifar100.log"),
    ),
]


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs to use, e.g. '0,1'.  Defaults to all detected GPUs.",
    )
    args = parser.parse_args()

    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]

    run_experiments(
        experiments = EXPERIMENTS,
        gpu_ids     = gpu_ids,
        log_dir     = LOG_DIR,
    )


if __name__ == "__main__":
    main()
