"""
Static IMP experiment on a torchvision pretrained ResNet50 (ImageNet1k).

No training, no dataset, no fine-tuning between steps.
We simply load the pretrained weights, then iteratively prune 10% of the
remaining weights by global magnitude and record both:
  - min saliency  (the pruning threshold = magnitude of the largest pruned weight)
  - avg saliency  (mean absolute magnitude of all active weights before pruning)

Each run creates a new timestamped folder with a description txt file.

Run from the project root:
    python -m src.resnet50_imagenet1k.nplh_static_imp_resnet50_imagenet
"""

import numpy as np
import torch
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights

from src.common_files_experiments.load_save import _load_model_weights
from src.common_files_experiments.vanilla_attributes_resnet50 import (
    RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
)
from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.constants import WEIGHTS_ATTR, MASK_ATTR
from src.infrastructure.layers import ConfigsNetworkMask, get_layers_primitive
from src.infrastructure.pruning_policy import MagnitudePruningPolicy
from src.infrastructure.nplh_run_context import (
    NplhRunContext, COL_STEP, COL_REMAINING, COL_SALIENCY,
    SAL_MIN, SAL_AVG, METHOD_IMP_STATIC,
)
from src.infrastructure.others import get_custom_model_sparsity_percent, get_device
from src.infrastructure.read_write import save_dict_to_csv
from src.resnet50_imagenet1k.resnet50_imagenet_class import Resnet50Imagenet

PRUNING_RATE    = 0.10
TARGET_SPARSITY = 0.999   # stop when >= 99.9% of weights have been pruned


def _collect_active_weights(model: Resnet50Imagenet) -> np.ndarray:
    parts = []
    for layer in get_layers_primitive(model):
        if hasattr(layer, WEIGHTS_ATTR) and hasattr(layer, MASK_ATTR):
            w = getattr(layer, WEIGHTS_ATTR).data
            m = getattr(layer, MASK_ATTR).data
            parts.append(w[m >= 0].detach().cpu().float().numpy())
    return np.concatenate(parts) if parts else np.array([])


def run_static_imp() -> NplhRunContext:
    run_ctx = NplhRunContext.create(
        run_name="resnet50_imagenet_static",
        description={
            "model":           "ResNet50",
            "dataset":         "ImageNet1k",
            "weights":         "torchvision IMAGENET1K_V1",
            "method":          METHOD_IMP_STATIC,
            "pruning_rate":    PRUNING_RATE,
            "target_sparsity": TARGET_SPARSITY,
            "retraining":      "none – pure magnitude cuts, no fine-tuning",
        },
    )

    configs_layers_initialization_all_kaiming_sqrt5()

    configs_network_masks = ConfigsNetworkMask(
        mask_pruning_enabled=False,
        weights_training_enabled=False,
    )
    model = Resnet50Imagenet(configs_network_masks).to(get_device())

    print("Downloading / loading torchvision ResNet50 (IMAGENET1K_V1) ...")
    tv_model   = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    state_dict = tv_model.state_dict()

    print("Transferring weights to custom model ...")
    _load_model_weights(
        model=model,
        model_dict=state_dict,
        standard_to_network_dict=RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
        skip_array=[],
    )
    del tv_model

    csv_min_path = run_ctx.csv_path("resnet50", "imagenet", SAL_MIN, METHOD_IMP_STATIC)
    csv_avg_path = run_ctx.csv_path("resnet50", "imagenet", SAL_AVG, METHOD_IMP_STATIC)

    policy = MagnitudePruningPolicy()

    steps_min = []; sal_min = []; rem_min = []
    steps_avg = []; sal_avg = []; rem_avg = []
    step = 0

    while True:
        remaining = get_custom_model_sparsity_percent(model)
        if 1.0 - remaining / 100.0 >= TARGET_SPARSITY:
            print(f"Reached target sparsity. Stopping.")
            break

        active = _collect_active_weights(model)
        if len(active) == 0:
            print("No active weights remaining. Stopping.")
            break

        try:
            result = policy.prune_step(model, PRUNING_RATE)
        except (ValueError, RuntimeError) as exc:
            print(f"Pruning stopped at step {step}: {exc}")
            break

        step += 1
        remaining_after = get_custom_model_sparsity_percent(model)

        steps_min.append(step); sal_min.append(result.threshold);   rem_min.append(remaining_after)
        steps_avg.append(step); sal_avg.append(result.avg_saliency); rem_avg.append(remaining_after)

        print(
            f"[step {step:3d}]  remaining={remaining_after:.4f}%  "
            f"min_sal={threshold:.6e}  avg_sal={avg_sal_val:.6e}"
        )

        # Save incrementally with standardised column names
        save_dict_to_csv(
            {COL_STEP: steps_min, COL_REMAINING: rem_min, COL_SALIENCY: sal_min},
            filename=csv_min_path,
        )
        save_dict_to_csv(
            {COL_STEP: steps_avg, COL_REMAINING: rem_avg, COL_SALIENCY: sal_avg},
            filename=csv_avg_path,
        )

    print(f"\nDone. {step} pruning steps performed.")
    print(f"  Min-saliency CSV → {csv_min_path}")
    print(f"  Avg-saliency CSV → {csv_avg_path}")
    return run_ctx


if __name__ == "__main__":
    run_static_imp()
