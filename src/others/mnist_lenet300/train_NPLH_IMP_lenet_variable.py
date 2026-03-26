import os
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
from src.infrastructure.layers import (
    ConfigsNetworkMask, calculate_pruning_epochs, get_layers_primitive,
)
from src.infrastructure.nplh_run_context import (
    NplhRunContext, COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
    SAL_MIN, SAL_AVG,
)
from src.infrastructure.pruning_policy import PruningPolicy, MagnitudePruningPolicy
from src.infrastructure.others import get_device, get_custom_model_sparsity_percent, prefix_path_with_root
from src.infrastructure.read_write import save_dict_to_csv
from src.infrastructure.training_utils import get_model_weights_params
from src.mnist_lenet300.model_class_variable import ModelLenetVariable

# ------------------------------------------------------------------
# Experiment hyper-parameters
# ------------------------------------------------------------------
BASELINE_EPOCHS = 50       # epochs before any pruning (dense training → baseline)
IMP_EPOCHS      = 300      # additional epochs for IMP after the baseline phase
TOTAL_EPOCHS    = BASELINE_EPOCHS + IMP_EPOCHS
PRUNING_RATE    = 0.05     # fraction of remaining weights pruned each IMP step
TARGET_SPARSITY = 0.999    # target fraction of weights to remove (99.9 %)


# ------------------------------------------------------------------
# Device-agnostic training helpers (no GradScaler – LeNet is tiny)
# ------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    dataset_context: DatasetSmallContext,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    while dataset_context.any_data_training_available():
        data, target = dataset_context.get_training_data_and_labels()
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()


def _test_epoch(
    model: nn.Module,
    dataset_context: DatasetSmallContext,
    epoch: int,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        while dataset_context.any_data_testing_available():
            data, target = dataset_context.get_testing_data_and_labels()
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total = dataset_context.get_data_testing_length()
    accuracy = 100.0 * correct / total
    print(f"  [epoch {epoch:3d}] accuracy: {correct}/{total} = {accuracy:.2f}%")
    return accuracy


# ------------------------------------------------------------------
# Weight-distribution helpers
# ------------------------------------------------------------------

def _collect_active_weights(model: nn.Module) -> np.ndarray:
    """Return raw weight values of all currently active (unpruned) weights."""
    parts = []
    for layer in get_layers_primitive(model):
        if hasattr(layer, "weights") and hasattr(layer, "mask_pruning"):
            w = layer.weights.data
            m = layer.mask_pruning.data
            parts.append(w[m >= 0].detach().cpu().float().numpy())
    return np.concatenate(parts) if parts else np.array([])


def _save_weight_distribution(
    active_before: np.ndarray,
    active_after: np.ndarray,
    threshold: float,
    step: int,
    epoch: int,
    arch_name: str,
    out_path: str,
) -> None:
    """
    Save side-by-side absolute-magnitude histograms (shared scale) + data CSV.

    Both panels share the same log-spaced x-axis and the same y-axis (density
    normalised by the total number of *before* weights so both panels are
    directly comparable).  The just-pruned weights are shown in red on the
    after panel.  A companion _data.csv is written next to the image.
    """
    import csv as _csv

    abs_before = np.abs(active_before)
    abs_after  = np.abs(active_after)
    abs_pruned = abs_before[abs_before <= threshold]

    # --- Log-spaced bins over the full magnitude range (from before) ---
    num_bins = min(200, max(50, len(abs_before) // 200))
    mag_min  = max(float(abs_before.min()), 1e-10)
    mag_max  = float(abs_before.max())
    bins     = np.logspace(np.log10(mag_min), np.log10(mag_max), num_bins + 1)
    bin_widths = np.diff(bins)
    bin_mids   = np.sqrt(bins[:-1] * bins[1:])   # geometric midpoints

    # --- Counts ---
    cnt_before, _ = np.histogram(abs_before, bins=bins)
    cnt_after,  _ = np.histogram(abs_after,  bins=bins)
    cnt_pruned, _ = np.histogram(abs_pruned, bins=bins)

    # Normalise everything by (total_before * bin_width) so both panels
    # are on the same density scale and together integrate to 1.
    n_before = max(len(abs_before), 1)
    dens_before = cnt_before / (n_before * bin_widths)
    dens_after  = cnt_after  / (n_before * bin_widths)
    dens_pruned = cnt_pruned / (n_before * bin_widths)

    # Shared y-limit
    y_max = max(
        dens_before.max() if len(dens_before) else 0,
        dens_after.max()  if len(dens_after)  else 0,
        dens_pruned.max() if len(dens_pruned) else 0,
    ) * 1.15

    # --- Plot ---
    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(14, 5))

    def _bar(ax, densities, color, alpha, label):
        ax.bar(bin_mids, densities, width=bin_widths,
               color=color, alpha=alpha, align="center", label=label)

    # Before panel
    _bar(ax_b, dens_before, "steelblue", 0.80, f"Active ({len(abs_before):,})")
    ax_b.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                 label=f"threshold ({threshold:.2e})")
    ax_b.set_xscale("log")
    ax_b.set_xlim(bins[0], bins[-1])
    ax_b.set_ylim(0, y_max)
    ax_b.set_title(f"Before pruning  (step {step}, epoch {epoch})\n"
                   f"{len(abs_before):,} active weights")
    ax_b.set_xlabel("|Weight| magnitude")
    ax_b.set_ylabel("Density  (normalised to before-set)")
    ax_b.legend(fontsize=8)

    # After panel
    _bar(ax_a, dens_after,  "steelblue", 0.80, f"Remaining  ({len(abs_after):,})")
    if len(abs_pruned) > 0:
        _bar(ax_a, dens_pruned, "red", 0.60, f"Just pruned  ({len(abs_pruned):,})")
    ax_a.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                 label=f"threshold ({threshold:.2e})")
    ax_a.set_xscale("log")
    ax_a.set_xlim(bins[0], bins[-1])
    ax_a.set_ylim(0, y_max)
    ax_a.set_title(f"After pruning  (step {step}, epoch {epoch})\n"
                   f"{len(abs_after):,} remaining / {len(abs_pruned):,} cut")
    ax_a.set_xlabel("|Weight| magnitude")
    ax_a.set_ylabel("Density  (normalised to before-set)")
    ax_a.legend(fontsize=8)

    fig.suptitle(f"{arch_name}  –  pruning step {step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)

    # --- Companion data CSV ---
    data_path = out_path.replace(".png", "_data.csv")
    with open(data_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "bin_lower", "bin_upper", "bin_mid",
            "count_active_before", "count_active_after", "count_just_pruned",
            "density_active_before", "density_active_after", "density_just_pruned",
        ])
        for i in range(len(bins) - 1):
            writer.writerow([
                f"{bins[i]:.6e}",      f"{bins[i+1]:.6e}",  f"{bin_mids[i]:.6e}",
                int(cnt_before[i]),    int(cnt_after[i]),    int(cnt_pruned[i]),
                f"{dens_before[i]:.6e}", f"{dens_after[i]:.6e}", f"{dens_pruned[i]:.6e}",
            ])


# ------------------------------------------------------------------
# Single-architecture IMP experiment
# ------------------------------------------------------------------

def train_lenet_variable_imp(
    hidden1: int,
    hidden2: int,
    run_ctx: NplhRunContext,
    pruning_policy: PruningPolicy | None = None,
    baseline_epochs: int = BASELINE_EPOCHS,
    imp_epochs: int = IMP_EPOCHS,
    pruning_rate: float = PRUNING_RATE,
    target_sparsity: float = TARGET_SPARSITY,
    save_weight_distributions: bool = False,
) -> float:
    """
    Train a variable-size LeNet on MNIST from scratch, then apply IMP.

    Phase 1 (epochs 1 .. baseline_epochs):
        Dense training – no pruning.  Accuracy at baseline_epochs is the
        "baseline accuracy" for this architecture.

    Phase 2 (epochs baseline_epochs+1 .. baseline_epochs+imp_epochs):
        IMP: prune pruning_rate of remaining weights at evenly-spaced
        epochs, fine-tune between steps.  Records NPLH data (step,
        saliency, % remaining, accuracy) at every pruning event.

    Parameters
    ----------
    hidden1, hidden2 : int
        Layer widths for the variable LeNet.
    run_ctx : NplhRunContext
        Shared run context that owns the output folder and CSV paths.
    pruning_policy : PruningPolicy, optional
        Which pruning criterion to apply.  Defaults to MagnitudePruningPolicy
        (standard IMP).  Data-dependent policies (Taylor, GraSP) will use
        one batch from the training set at each pruning step.
    baseline_epochs : int
        Epochs of dense training before any pruning.
    imp_epochs : int
        Additional epochs for the IMP phase.
    pruning_rate : float
        Fraction of remaining active weights pruned at each step.
    target_sparsity : float
        Stop pruning once this fraction of weights has been removed.
    save_weight_distributions : bool
        When True, saves per-step weight-magnitude histograms (PNG + CSV)
        to a ``weight_distributions/{arch_name}/`` subfolder.  Disabled by
        default because it generates many files and slows training slightly.

    Returns
    -------
    baseline_accuracy : float
    """
    if pruning_policy is None:
        pruning_policy = MagnitudePruningPolicy()

    total_epochs    = baseline_epochs + imp_epochs
    prunable_params = 784 * hidden1 + hidden1 * hidden2 + hidden2 * 10
    arch_name = f"lenet_{hidden1}_{hidden2}"
    dataset   = "mnist"
    print(f"\n[LeNet ({hidden1}, {hidden2})]  prunable weights: {prunable_params:,}")

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

    # Pruning schedule: compute relative epochs inside the IMP phase, then
    # shift by baseline_epochs to get global epoch numbers.
    pruning_epochs_relative = calculate_pruning_epochs(
        target_sparsity=target_sparsity,
        pruning_rate=pruning_rate,
        total_epochs=imp_epochs,
        start_epoch=1,
    )
    pruning_epochs = set(e + baseline_epochs for e in pruning_epochs_relative)
    print(f"  Pruning at {len(pruning_epochs)} epochs (global): {sorted(pruning_epochs)[:8]} ...")

    # Canonical CSV paths via run context — method tag comes from the policy
    method_tag   = pruning_policy.method_tag
    csv_min_path = run_ctx.csv_path(arch_name, dataset, SAL_MIN, method_tag)
    csv_avg_path = run_ctx.csv_path(arch_name, dataset, SAL_AVG, method_tag)

    # Optional weight-distribution output directory
    dist_dir = None
    if save_weight_distributions:
        dist_dir = os.path.join(run_ctx.folder_path, "weight_distributions", arch_name)
        os.makedirs(dist_dir, exist_ok=True)

    baseline_accuracy = 0.0

    steps_min  = []; sal_min  = []; rem_min  = []; acc_min  = []
    steps_avg  = []; sal_avg  = []; rem_avg  = []; acc_avg  = []
    pruning_step = 0

    for epoch in range(1, total_epochs + 1):
        dataset_context.init_data_split()
        _train_epoch(model, dataset_context, optimizer)
        acc = _test_epoch(model, dataset_context, epoch)

        if epoch == baseline_epochs:
            baseline_accuracy = acc
            print(f"  >>> Baseline accuracy: {baseline_accuracy:.2f}%")

        if epoch in pruning_epochs:
            try:
                pruning_step += 1

                active_before = _collect_active_weights(model) if save_weight_distributions else None

                def _get_batch():
                    dataset_context.init_data_split()
                    return dataset_context.get_training_data_and_labels()

                result    = pruning_policy.prune_step(model, pruning_rate, get_batch=_get_batch)
                threshold = result.threshold
                avg_sal_val = result.avg_saliency
                remaining = get_custom_model_sparsity_percent(model)

                if save_weight_distributions:
                    active_after = _collect_active_weights(model)
                    hist_path = os.path.join(
                        dist_dir, f"step_{pruning_step:03d}_epoch{epoch}.png"
                    )
                    _save_weight_distribution(
                        active_before, active_after, threshold,
                        pruning_step, epoch, arch_name, hist_path,
                    )

                # Accumulate min-saliency
                steps_min.append(pruning_step); sal_min.append(threshold)
                rem_min.append(remaining);      acc_min.append(acc)

                # Accumulate avg-saliency
                steps_avg.append(pruning_step); sal_avg.append(avg_sal_val)
                rem_avg.append(remaining);      acc_avg.append(acc)

                print(
                    f"  [step {pruning_step}] epoch {epoch}: "
                    f"threshold={threshold:.6e}, avg_sal={avg_sal_val:.6e}, "
                    f"remaining={remaining:.2f}%"
                )

                # Save both CSVs incrementally using standardised column names
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
                print(f"  Pruning stopped early at epoch {epoch}: {exc}")
                break

    print(f"  Saved min-saliency CSV  → {csv_min_path}")
    print(f"  Saved avg-saliency CSV  → {csv_avg_path}")
    return baseline_accuracy
