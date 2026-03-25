"""
train_NPLH_control_lenet_variable.py
=====================================
Control-group experiment: train to convergence once, then prune iteratively
WITHOUT any retraining between steps.  This isolates whether the NPLH
saliency/density relationship is a product of the pruning+training loop
or simply a property of magnitude-based pruning applied to a fixed set of
weights.

For each architecture:
  Phase 1 (epochs 1 .. BASELINE_EPOCHS):
      Dense training until convergence.

  Phase 2 (no training):
      Repeatedly prune PRUNING_RATE of remaining weights by magnitude, then
      evaluate.  No gradient steps between pruning events.
      Records at each step:
        - min saliency  (pruning threshold = largest magnitude pruned)
        - avg saliency  (mean |w| of active weights BEFORE the cut)
        - remaining %   (after the cut)
        - accuracy      (evaluated immediately after the cut)
      Also saves weight-distribution histograms and companion data CSVs.

Outputs (inside neural_pruning_law/final_data/lenet_variable/):
  lenet_{h1}_{h2}_control_imp.csv       — min-saliency control data
  lenet_{h1}_{h2}_control_avg.csv       — avg-saliency control data
  weight_distributions/lenet_{h1}_{h2}_control/
      step_NNN_epochCUT.png + _data.csv — per-step histograms

Run from the project root:
    python -m src.mnist_lenet300.train_NPLH_control_lenet_variable
"""

import os
import csv as _csv_mod
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.infrastructure.configs_layers import configs_layers_initialization_all_kaiming_sqrt5
from src.infrastructure.dataset_context.dataset_context import (
    DatasetSmallContext, DatasetSmallType, dataset_context_configs_mnist,
)
from src.infrastructure.layers import ConfigsNetworkMask
from src.infrastructure.nplh_run_context import (
    NplhRunContext, COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
    SAL_MIN, SAL_AVG,
)
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent
from src.infrastructure.pruning_policy import MagnitudePruningPolicy
from src.infrastructure.read_write import save_dict_to_csv
from src.infrastructure.training_utils import get_model_weights_params
from src.mnist_lenet300.model_class_variable import ModelLenetVariable

# ------------------------------------------------------------------
# Hyper-parameters  (defaults – all overridable via function args)
# ------------------------------------------------------------------
BASELINE_EPOCHS = 50
PRUNING_RATE    = 0.05
TARGET_SPARSITY = 0.999


# ------------------------------------------------------------------
# Training / evaluation helpers
# ------------------------------------------------------------------

def _train_epoch(model, dataset_context, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()


def _evaluate(model, dataset_context) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        while dataset_context.any_data_testing_available():
            data, target = dataset_context.get_testing_data_and_labels()
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total = dataset_context.get_data_testing_length()
    return 100.0 * correct / total


# ------------------------------------------------------------------
# Weight-distribution helpers (shared logic with IMP script)
# ------------------------------------------------------------------

def _collect_active_weights(model) -> np.ndarray:
    parts = []
    for layer in model.get_layers_primitive():
        if hasattr(layer, "weights") and hasattr(layer, "mask_pruning"):
            w = layer.weights.data
            m = layer.mask_pruning.data
            parts.append(w[m >= 0].detach().cpu().float().numpy())
    return np.concatenate(parts) if parts else np.array([])


def _save_weight_distribution(active_before, active_after, threshold,
                               step, arch_name, out_path):
    abs_before = np.abs(active_before)
    abs_after  = np.abs(active_after)
    abs_pruned = abs_before[abs_before <= threshold]

    num_bins   = min(200, max(50, len(abs_before) // 200))
    mag_min    = max(float(abs_before.min()), 1e-10)
    mag_max    = float(abs_before.max())
    bins       = np.logspace(np.log10(mag_min), np.log10(mag_max), num_bins + 1)
    bin_widths = np.diff(bins)
    bin_mids   = np.sqrt(bins[:-1] * bins[1:])

    cnt_before, _ = np.histogram(abs_before, bins=bins)
    cnt_after,  _ = np.histogram(abs_after,  bins=bins)
    cnt_pruned, _ = np.histogram(abs_pruned, bins=bins)

    n_before    = max(len(abs_before), 1)
    dens_before = cnt_before / (n_before * bin_widths)
    dens_after  = cnt_after  / (n_before * bin_widths)
    dens_pruned = cnt_pruned / (n_before * bin_widths)
    y_max = max(dens_before.max(), dens_after.max(),
                dens_pruned.max() if len(dens_pruned) else 0) * 1.15

    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(14, 5))

    def _bar(ax, dens, color, alpha, label):
        ax.bar(bin_mids, dens, width=bin_widths,
               color=color, alpha=alpha, align="center", label=label)

    _bar(ax_b, dens_before, "steelblue", 0.80, f"Active ({len(abs_before):,})")
    ax_b.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                 label=f"threshold ({threshold:.2e})")
    ax_b.set_xscale("log"); ax_b.set_xlim(bins[0], bins[-1]); ax_b.set_ylim(0, y_max)
    ax_b.set_title(f"Before pruning  (step {step})\n{len(abs_before):,} active weights")
    ax_b.set_xlabel("|Weight| magnitude"); ax_b.set_ylabel("Density (norm. to before-set)")
    ax_b.legend(fontsize=8)

    _bar(ax_a, dens_after,  "steelblue", 0.80, f"Remaining  ({len(abs_after):,})")
    if len(abs_pruned) > 0:
        _bar(ax_a, dens_pruned, "red", 0.60, f"Just pruned  ({len(abs_pruned):,})")
    ax_a.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                 label=f"threshold ({threshold:.2e})")
    ax_a.set_xscale("log"); ax_a.set_xlim(bins[0], bins[-1]); ax_a.set_ylim(0, y_max)
    ax_a.set_title(f"After pruning  (step {step})\n"
                   f"{len(abs_after):,} remaining / {len(abs_pruned):,} cut")
    ax_a.set_xlabel("|Weight| magnitude"); ax_a.set_ylabel("Density (norm. to before-set)")
    ax_a.legend(fontsize=8)

    fig.suptitle(f"{arch_name} [CONTROL]  –  pruning step {step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)

    # Companion data CSV
    data_path = out_path.replace(".png", "_data.csv")
    with open(data_path, "w", newline="") as f:
        writer = _csv_mod.writer(f)
        writer.writerow([
            "bin_lower", "bin_upper", "bin_mid",
            "count_active_before", "count_active_after", "count_just_pruned",
            "density_active_before", "density_active_after", "density_just_pruned",
        ])
        for i in range(len(bins) - 1):
            writer.writerow([
                f"{bins[i]:.6e}", f"{bins[i+1]:.6e}", f"{bin_mids[i]:.6e}",
                int(cnt_before[i]), int(cnt_after[i]), int(cnt_pruned[i]),
                f"{dens_before[i]:.6e}", f"{dens_after[i]:.6e}", f"{dens_pruned[i]:.6e}",
            ])


# ------------------------------------------------------------------
# Single-architecture control experiment
# ------------------------------------------------------------------

def train_lenet_control(
    hidden1: int,
    hidden2: int,
    run_ctx: NplhRunContext,
    baseline_epochs: int = BASELINE_EPOCHS,
    pruning_rate: float = PRUNING_RATE,
    target_sparsity: float = TARGET_SPARSITY,
    save_weight_distributions: bool = False,
) -> float:
    """
    Train once to convergence, then prune iteratively with NO retraining.

    Uses MagnitudePruningPolicy throughout.  No gradient steps are taken
    in Phase 2 — weights are purely pruned and evaluated.

    Returns
    -------
    baseline_accuracy : float
    """
    prunable_params = 784 * hidden1 + hidden1 * hidden2 + hidden2 * 10
    arch_name = f"lenet_{hidden1}_{hidden2}"
    print(f"\n[Control  LeNet ({hidden1}, {hidden2})]  prunable weights: {prunable_params:,}")

    configs_layers_initialization_all_kaiming_sqrt5()
    configs_network_masks = ConfigsNetworkMask(
        mask_pruning_enabled=False,
        weights_training_enabled=True,
    )
    model = ModelLenetVariable(hidden1, hidden2, configs_network_masks).to(get_device())

    dataset_context = DatasetSmallContext(
        dataset=DatasetSmallType.MNIST,
        configs=dataset_context_configs_mnist(),
    )
    optimizer = torch.optim.Adam(
        get_model_weights_params(model), lr=1e-3, weight_decay=1e-4,
    )

    # ── Phase 1: dense training ────────────────────────────────────────────
    baseline_accuracy = 0.0
    for epoch in range(1, baseline_epochs + 1):
        dataset_context.init_data_split()
        _train_epoch(model, dataset_context, optimizer)
        acc = _evaluate(model, dataset_context)
        print(f"  [epoch {epoch:3d}] accuracy: {acc:.2f}%")
        if epoch == baseline_epochs:
            baseline_accuracy = acc
            print(f"  >>> Baseline accuracy: {baseline_accuracy:.2f}%")

    # ── Output paths via run context ───────────────────────────────────────
    csv_min_path = run_ctx.csv_path(arch_name, "mnist", SAL_MIN, "control")
    csv_avg_path = run_ctx.csv_path(arch_name, "mnist", SAL_AVG, "control")

    dist_dir = None
    if save_weight_distributions:
        dist_dir = os.path.join(run_ctx.folder_path, "weight_distributions", f"{arch_name}_control")
        os.makedirs(dist_dir, exist_ok=True)

    # ── Phase 2: prune without retraining ─────────────────────────────────
    policy = MagnitudePruningPolicy()

    steps_min = []; sal_min = []; rem_min = []; acc_min = []
    steps_avg = []; sal_avg = []; rem_avg = []; acc_avg = []
    pruning_step = 0

    while get_custom_model_sparsity_percent(model) > (1.0 - target_sparsity) * 100:
        try:
            pruning_step += 1

            active_before = _collect_active_weights(model) if save_weight_distributions else None

            result    = policy.prune_step(model, pruning_rate)
            threshold = result.threshold
            avg_sal   = result.avg_saliency
            remaining = get_custom_model_sparsity_percent(model)

            if save_weight_distributions:
                active_after = _collect_active_weights(model)
                hist_path = os.path.join(dist_dir, f"step_{pruning_step:03d}.png")
                _save_weight_distribution(active_before, active_after, threshold,
                                          pruning_step, arch_name, hist_path)

            dataset_context.init_data_split()
            acc = _evaluate(model, dataset_context)

            steps_min.append(pruning_step); sal_min.append(threshold)
            rem_min.append(remaining);      acc_min.append(acc)
            steps_avg.append(pruning_step); sal_avg.append(avg_sal)
            rem_avg.append(remaining);      acc_avg.append(acc)

            print(
                f"  [control step {pruning_step:3d}] threshold={threshold:.4e}  "
                f"avg_sal={avg_sal:.4e}  remaining={remaining:.2f}%  acc={acc:.2f}%"
            )

            save_dict_to_csv(
                {COL_STEP: steps_min, COL_REMAINING: rem_min,
                 COL_SALIENCY: sal_min, COL_ACCURACY: acc_min},
                filename=csv_min_path,
            )
            save_dict_to_csv(
                {COL_STEP: steps_avg, COL_REMAINING: rem_avg,
                 COL_SALIENCY: sal_avg, COL_ACCURACY: acc_avg},
                filename=csv_avg_path,
            )

        except (ValueError, RuntimeError) as exc:
            print(f"  Pruning stopped at step {pruning_step}: {exc}")
            break

    print(f"  Saved control min-saliency → {csv_min_path}")
    print(f"  Saved control avg-saliency → {csv_avg_path}")
    return baseline_accuracy


# ------------------------------------------------------------------
# Run all architectures
# ------------------------------------------------------------------

ARCHITECTURES = [
    {"h1":    4, "h2":    2},
    {"h1":   10, "h2":    5},
    {"h1":   20, "h2":   10},
    {"h1":   50, "h2":   25},
    {"h1":  100, "h2":   50},
    {"h1":  300, "h2":  100},
    {"h1":  600, "h2":  200},
    {"h1": 1200, "h2":  400},
    {"h1": 2500, "h2":  800},
    {"h1": 5000, "h2": 1500},
]


def run_all():
    from src.infrastructure.nplh_run_context import NplhRunContext, METHOD_IMP
    run_ctx = NplhRunContext.create(
        run_name="lenet_variable_control",
        description={
            "model":           "Variable LeNet (784→h1→h2→10)",
            "dataset":         "MNIST",
            "method":          "control (no retraining)",
            "baseline_epochs": BASELINE_EPOCHS,
            "pruning_rate":    PRUNING_RATE,
            "target_sparsity": TARGET_SPARSITY,
        },
    )
    results = []
    for arch in ARCHITECTURES:
        h1, h2 = arch["h1"], arch["h2"]
        prunable = 784 * h1 + h1 * h2 + h2 * 10
        print(f"\n{'='*60}")
        print(f"Control: LeNet ({h1}, {h2})  |  {prunable:,} prunable weights")
        print(f"{'='*60}")
        baseline_acc = train_lenet_control(h1, h2, run_ctx)
        results.append((f"lenet_{h1}_{h2}", baseline_acc))
        print(f"  Finished. Baseline accuracy: {baseline_acc:.2f}%")

    print(f"\n{'='*60}")
    print("All control experiments complete.")
    print(f"{'='*60}")
    for name, acc in results:
        print(f"  {name:<22}  baseline: {acc:.2f}%")


if __name__ == "__main__":
    run_all()
