"""
train_lenet300_activation_saliency.py
======================================
Random-pruning experiment on the canonical LeNet-300-100 (MNIST) that
records two saliency metrics at each pruning step:

  1. Magnitude saliency  — mean |w| of all active weights (existing metric).
  2. APoZ saliency — Average Percentage of Zeros: for each hidden neuron
                     that still has at least one active incoming weight,
                     the fraction of batch samples for which that neuron
                     does NOT fire (output == 0 after ReLU).  Averaged
                     over all such active neurons.  Higher APoZ → more
                     dead neurons → less capable network.

Pruning is uniformly random (no score-based selection).  After each pruning
step the model is fine-tuned for FINETUNE_EPOCHS epochs.  The loop stops
when test accuracy drops below COLLAPSE_THRESHOLD.

Only neurons with at least one unpruned incoming weight are counted in APoZ
— fully disconnected neurons are excluded because they are structurally dead
and would artificially inflate the score.

Outputs (2 CSVs written incrementally to run_ctx):
  lenet_300_100_mnist_avg_random_retrain_magnitude.csv
  lenet_300_100_mnist_avg_random_retrain_apoz.csv
"""

import torch
import torch.nn.functional as F

from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, dataset_context_configs_mnist,
)
from src.infrastructure.layers import ConfigsNetworkMasksImportance
from src.infrastructure.nplh_run_context import (
    NplhRunContext,
    COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
    SAL_AVG,
    METHOD_RANDOM_RETRAIN_MAGNITUDE,
)
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent
from src.infrastructure.pruning_policy import RandomPruningPolicy
from src.infrastructure.read_write import save_dict_to_csv
from src.infrastructure.training_common import get_model_weights_params
from src.mnist_lenet300.model_class_variable import ModelLenetVariable
from src.mnist_lenet300.train_NPLH_IMP_lenet_variable import _train_epoch, _test_epoch

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
HIDDEN1 = 300
HIDDEN2 = 100

TRAIN_EPOCHS       = 25     # dense training epochs before pruning
PRUNING_RATE       = 0.10   # fraction of remaining weights pruned per step
FINETUNE_EPOCHS    = 3      # fine-tuning epochs between pruning steps
FINETUNE_LR        = 0.001  # constant Adam LR
COLLAPSE_THRESHOLD = 15.0   # % — stop when accuracy drops below this

ARCH_NAME    = f"lenet_{HIDDEN1}_{HIDDEN2}"
DATASET_NAME = "mnist"

# APoZ-saliency has no entry in nplh_run_context yet; define locally.
METHOD_RANDOM_RETRAIN_APOZ = "random_retrain_apoz"


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _build_model() -> ModelLenetVariable:
    configs_layers_initialization_all_kaiming_sqrt5()
    configs_masks = ConfigsNetworkMasksImportance(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    return ModelLenetVariable(HIDDEN1, HIDDEN2, configs_masks).to(get_device())


def _train_dense(
    model: ModelLenetVariable,
    dataset_context: DatasetSmallContext,
    n_epochs: int,
) -> float:
    """Train n_epochs with Adam; return final test accuracy."""
    optimizer = torch.optim.Adam(
        get_model_weights_params(model), lr=FINETUNE_LR, weight_decay=1e-4,
    )
    acc = 0.0
    for epoch in range(1, n_epochs + 1):
        dataset_context.init_data_split()
        _train_epoch(model, dataset_context, optimizer)
        acc = _test_epoch(model, dataset_context, epoch)
    return acc


def _measure_apoz(
    model: ModelLenetVariable,
    data: torch.Tensor,
) -> float:
    """
    APoZ (Average Percentage of Zeros) over active hidden neurons only.

    For each hidden neuron j that has at least one unpruned incoming weight,
    compute the fraction of batch samples for which that neuron does not fire:
        apoz_j = mean_i( relu(fc(x_i))_j == 0 )

    Return the mean of apoz_j across all such active neurons in fc1 and fc2.
    Neurons whose every incoming weight is pruned are excluded — they are
    structurally dead and would artificially inflate the score.

    Parameters
    ----------
    model : ModelLenetVariable — in eval mode with masks applied.
    data  : torch.Tensor — mini-batch already on the correct device,
            shape [B, 1, 28, 28] or [B, 784].

    Returns
    -------
    float — scalar APoZ in [0, 1].  0 = all active neurons always fire;
            1 = all active neurons never fire.
    """
    model.eval()
    with torch.no_grad():
        x    = data.view(-1, 28 * 28)
        act1 = F.relu(model.fc1(x))    # [B, hidden1]
        act2 = F.relu(model.fc2(act1)) # [B, hidden2]

        # Active neurons: at least one unpruned incoming weight.
        # mask_pruning shape matches weights: [out_features, in_features].
        active_fc1 = (model.fc1.mask_pruning.data >= 0).any(dim=1)  # [hidden1]
        active_fc2 = (model.fc2.mask_pruning.data >= 0).any(dim=1)  # [hidden2]

        # APoZ per neuron = fraction of batch where output is 0
        apoz1 = (act1 == 0).float().mean(dim=0)  # [hidden1]
        apoz2 = (act2 == 0).float().mean(dim=0)  # [hidden2]

        parts = []
        if active_fc1.any():
            parts.append(apoz1[active_fc1])
        if active_fc2.any():
            parts.append(apoz2[active_fc2])

        if not parts:
            return 1.0  # all neurons dead — maximally zero

        return float(torch.cat(parts).mean().item())


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------

def run_lenet300_random_activation(run_ctx: NplhRunContext) -> None:
    """
    Train LeNet-300-100 from scratch, then randomly prune + fine-tune
    until accuracy < COLLAPSE_THRESHOLD.

    At each pruning step, both saliency metrics are measured on the
    same time-point (before weights are removed):
      - magnitude saliency via RandomPruningPolicy.avg_saliency
      - APoZ saliency      via _measure_apoz() (active neurons only)

    Two CSVs are saved incrementally to run_ctx.

    Parameters
    ----------
    run_ctx : NplhRunContext — shared output folder for this run.
    """
    print(f"\n[{ARCH_NAME}] Random pruning — magnitude + APoZ saliency")

    model           = _build_model()
    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.MNIST,
        configs=dataset_context_configs_mnist(),
    )

    # ── Phase 1: dense training ──────────────────────────────────────────
    print(f"[{ARCH_NAME}] Dense training ({TRAIN_EPOCHS} epochs) ...")
    baseline_acc = _train_dense(model, dataset_context, TRAIN_EPOCHS)
    print(f"[{ARCH_NAME}] Baseline accuracy: {baseline_acc:.2f}%")

    # ── Phase 2: random prune → fine-tune → repeat ───────────────────────
    policy    = RandomPruningPolicy()
    optimizer = torch.optim.Adam(
        get_model_weights_params(model), lr=FINETUNE_LR, weight_decay=1e-4,
    )

    csv_mag  = run_ctx.csv_path(ARCH_NAME, DATASET_NAME, SAL_AVG, METHOD_RANDOM_RETRAIN_MAGNITUDE)
    csv_apoz = run_ctx.csv_path(ARCH_NAME, DATASET_NAME, SAL_AVG, METHOD_RANDOM_RETRAIN_APOZ)

    steps    = []
    mag_sal  = []
    apoz_sal = []
    rem      = []
    acc_list = []

    step         = 0
    global_epoch = TRAIN_EPOCHS

    while True:
        # ── Measure APoZ BEFORE pruning ──────────────────────────────────
        dataset_context.init_data_split()
        data_batch, _ = dataset_context.get_training_data_and_labels()
        apoz          = _measure_apoz(model, data_batch)

        # ── Random prune (internally records mean |w| before pruning) ────
        try:
            result = policy.prune_step(model, PRUNING_RATE)
        except (ValueError, RuntimeError) as exc:
            print(f"[{ARCH_NAME}] Pruning stopped at step {step}: {exc}")
            break

        remaining = get_custom_model_sparsity_percent(model)
        step += 1

        # ── Fine-tune ────────────────────────────────────────────────────
        acc = 0.0
        for _ in range(FINETUNE_EPOCHS):
            global_epoch += 1
            dataset_context.init_data_split()
            _train_epoch(model, dataset_context, optimizer)
            acc = _test_epoch(model, dataset_context, global_epoch)

        print(
            f"[{ARCH_NAME}] step {step:3d}  remaining={remaining:.4f}%  "
            f"mag_sal={result.avg_saliency:.4e}  "
            f"apoz={apoz:.4f}  acc={acc:.2f}%"
        )

        steps.append(step)
        mag_sal.append(result.avg_saliency)
        apoz_sal.append(apoz)
        rem.append(remaining)
        acc_list.append(acc)

        # Incremental CSV writes — safe to interrupt mid-run
        save_dict_to_csv(
            {COL_STEP: steps, COL_REMAINING: rem,
             COL_SALIENCY: mag_sal, COL_ACCURACY: acc_list},
            filename=csv_mag,
        )
        save_dict_to_csv(
            {COL_STEP: steps, COL_REMAINING: rem,
             COL_SALIENCY: apoz_sal, COL_ACCURACY: acc_list},
            filename=csv_apoz,
        )

        if acc < COLLAPSE_THRESHOLD:
            print(f"\n[{ARCH_NAME}] *** Collapse at step {step}: "
                  f"acc={acc:.2f}%  remaining={remaining:.4f}% ***")
            break

    print(f"[{ARCH_NAME}] Done. Pruning steps: {step}")
    print(f"  Magnitude CSV → {csv_mag}")
    print(f"  APoZ CSV      → {csv_apoz}")
