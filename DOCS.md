# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase investigates two related ideas:

1. **Hyperflux** – continuous/differentiable pruning via learnable *flow parameters* (`mask_pruning`) and a pressure scheduler that drives sparsity toward a target over training.
2. **NPLH (Neural Pruning Law Hypothesis)** – an empirical observation that saliency follows a power-law relationship with network density. NPLH is not limited to IMP; it applies to any pruning method (IMP, Hyperflux, etc.) — the hypothesis is about the saliency/density relationship regardless of how sparsity is achieved. The saliency metric is currently being explored in two forms: **minimum saliency** (the pruning threshold, i.e. the magnitude of the largest-magnitude weight removed at each step) and **average saliency** (mean absolute magnitude of all active weights before pruning). Both are recorded and both hypotheses are under active investigation.

## Running Experiments

All scripts must be run as modules from the project root (the directory containing `.root`):

```bash
# Hyperflux experiments (entry point CLI)
python entry.py --experiment <name> [--sparsity <pct>]
# Available: lenet300_bottleneck, lenet300_convergence,
#            resnet50_imagenet_96.5, resnet50_imagenet_95, resnet50_imagenet_90,
#            resnet50_cifar100, resnet50_cifar10, vgg19_cifar10, vgg19_cifar100

# Examples:
python entry.py --experiment resnet50_cifar100 --sparsity 98
python entry.py --experiment vgg19_cifar10 --sparsity 95

# NPLH IMP experiments (variable LeNet across architectures)
python -m src.mnist_lenet300.run_lenet_variable_experiment

# NPLH IMP experiments for larger models
python NPLH_experiments.py
python imagenet_NPLH.py   # requires IMAGENET_PATH env var
```

There are no automated tests. Install dependencies with:
```bash
pip install -r requirements.txt
# PyTorch must be installed separately: https://pytorch.org/get-started/locally/
```

## Architecture

### Infrastructure (`src/infrastructure/`)

The core layer system uses custom `nn.Parameter`-based layers instead of standard PyTorch layers:

- **`layers.py`** – `LayerLinearMaskImportance` and `LayerConv2MaskImportance` are the primitive building blocks. Each has three parameters: `weights`, `bias`, and `mask_pruning`. The mask encodes pruning state: `mask_pruning >= 0` → weight is active; `mask_pruning == -1.0` → pruned. `MaskPruningFunctionConstant` (a custom autograd function) converts mask values to binary (0/1) in forward, passes gradients through at a fixed scale (`GRADIENT_IDENTITY_SCALER`). This is applied in forward even when `mask_pruning_enabled=False`, so IMP works correctly without learnable masks.

- **`layers.py`** also contains the two key IMP utilities:
  - `prune_model_globally(model, rate)` – prunes `rate` fraction of remaining active weights by global magnitude; sets their mask to -1.0; returns the threshold value (= minimum saliency for NPLH).
  - `calculate_pruning_epochs(target_sparsity, pruning_rate, total_epochs, start_epoch)` – returns a list of epoch indices at which to prune.

- **`others.py`** – `get_device()`, `prefix_path_with_root()`, `get_custom_model_sparsity_percent(model)`.

- **`read_write.py`** – `save_dict_to_csv(dict_of_lists, filename)` saves CSV incrementally.

- **`training_common.py`** – `get_model_weights_params(model)` returns only `weights`/`bias` params (excludes `mask_pruning`). Use this to build optimizers for IMP.

- **`constants.py`** – global paths (`BASELINE_MODELS_PATH`, `PRUNED_MODELS_PATH`, `DATA_PATH`), attribute name constants (`WEIGHTS_ATTR='weights'`, `WEIGHTS_PRUNING_ATTR='mask_pruning'`, `BIAS_ATTR='bias'`), and LR/flow-param config functions (`config_sgd_setup()`, `config_adam_setup()`).

- **`configs_general.py`** – `WANDB_REGISTER` flag gates all wandb calls.

### Hyperflux Training Pipeline

For Hyperflux (differentiable pruning) experiments, each experiment package defines:
- **Model class** extending `LayerComposite`, registering `LayerPrimitive` layers in `self.registered_layers`. Call `get_layers_primitive()` to traverse.
- **`TrainingContextPrunedTrain`** – holds two optimizers: one for weights, one for flow params (`mask_pruning`). `set_gamma(gamma)` controls pressure magnitude.
- **`StagesContextPrunedTrain`** – orchestrates LR schedulers and `PressureSchedulerPolicy1` across two phases: pruning (epochs 1..`pruning_end`) and regrowth (epochs `pruning_end+1`..`regrowing_end`).
- **`PressureSchedulerPolicy1`** – adaptive pressure: increases `gamma` when network has too many parameters vs. trajectory, decreases when ahead. The trajectory is computed via `TrajectoryCalculator` (sigmoid-based cumulative pruning curve, binary-searched to hit the target).
- The flow-params loss term is `gamma * flow_params_loss` added to the classification loss during the pruning phase.

### NPLH Experiment Infrastructure

#### `NplhRunContext` (`src/infrastructure/nplh_run_context.py`)
Single source of truth for managing NPLH experiment runs. Always use this for output paths.
- `NplhRunContext.create(run_name, description)` → creates a timestamped folder under `neural_pruning_law/final_data/` and writes a `run_description.txt`.
- `ctx.csv_path(model, dataset, saliency_type, method)` → canonical CSV path, e.g. `resnet50_cifar10_min_IMP_magnitude.csv`.
- `ctx.plot_path(name)` → PDF path inside the run folder.
- Column name constants: `COL_STEP`, `COL_REMAINING`, `COL_SALIENCY`, `COL_ACCURACY`.
- Saliency type tags: `SAL_MIN` (`"min"`), `SAL_AVG` (`"avg"`).
- Method tags (use constants, not raw strings):
  - `METHOD_IMP_MAGNITUDE` (`"IMP_magnitude"`) — magnitude IMP; the only method where "IMP" prefix applies.
  - `METHOD_IMP_STATIC` (`"static"`) — magnitude pruning with no retraining (control/baseline).
  - `METHOD_IMP_TAYLOR` (`"taylor"`) — Taylor criterion.
  - `METHOD_GRADIENT` (`"gradient"`) — raw gradient magnitude.
  - `METHOD_RANDOM_REGROWTH` (`"random_regrowth"`) — random prune + gradient-guided regrowth.

#### `PruningPolicy` (`src/infrastructure/pruning_policy.py`)
Decoupled, network-agnostic pruning policies. All policies work on any model with `get_layers_primitive()`.
- All policies return `PruningStepResult(threshold, avg_saliency)` for NPLH recording.
- `policy.prune_step(model, pruning_rate, get_batch=None)` → `PruningStepResult`.

**Available policies:**

| Class | `method_tag` | Score / mechanism | Data needed |
|---|---|---|---|
| `MagnitudePruningPolicy` | `"IMP_magnitude"` | `\|w\|` — classic IMP | None |
| `TaylorPruningPolicy(criterion)` | `"taylor"` | `\|w · ∂L/∂w\|` — first-order loss-change estimate | One batch |
| `GradientPruningPolicy(criterion)` | `"gradient"` | `\|∂L/∂w\|` — raw gradient magnitude, weight-scale-independent | One batch |
| `RandomRegrowthPruningPolicy(criterion, oversample_factor=2.0)` | `"random_regrowth"` | Randomly prune `k×n_net` weights, zero them, compute `∂L/∂w\|_{w=0}`, regrow top `(k-1)×n_net` where `−sign(w_orig)·grad > 0` (GD update aligns with original sign); net pruned = `n_net`. | One batch |

**Naming convention:** "IMP" (Iterative Magnitude Pruning) appears **only** in `IMP_magnitude`. All other methods have their own names. The static/control experiment (no retraining) uses tag `"static"` — it describes the training schedule, not a scoring criterion.

**Key gradient mechanics for `RandomRegrowthPruningPolicy`:** candidates have their weight values zeroed while their mask stays `≥ 0`. Since `MaskPruningFunctionConstant` returns `1` for active masks, the forward sees `w·1 = 0` and `∂(w·1)/∂w = 1`, so the full upstream gradient flows to `w.grad` — giving the exact `∂L/∂w|_{w=0}` without any special instrumentation.

#### General NPLH experiment pattern
1. Create `NplhRunContext` (timestamped folder).
2. Train from scratch device-agnostically (no GradScaler/autocast — these are CUDA-only).
3. Save baseline with a fixed name to `networks_baseline/`.
4. For each pruning experiment, load a fresh copy of the baseline model.
5. Run pruning loop, record `(step, remaining%, saliency, accuracy)` after each pruning event.
6. Save two CSVs per experiment (min saliency + avg saliency) incrementally via `save_dict_to_csv`.
7. For static/control experiments: prune without any retraining, loop until target sparsity.
8. For IMP experiments: use `calculate_pruning_epochs` to get evenly-spaced pruning events, fine-tune between steps.

**Device handling:** `get_device()` returns CUDA → MPS (MacBook GPU) → CPU in priority order. Always use it; never hardcode `"cuda"`. The legacy `train_mixed_baseline` in `train_model_scratch_commons.py` uses `GradScaler("cuda")` and is **not** device-agnostic — write custom training loops for new NPLH experiments.

### NPLH / IMP Pattern

For IMP-based NPLH experiments (no Hyperflux):
- `mask_pruning_enabled=False` in `ConfigsNetworkMasksImportance` (masks are fixed, not learned).
- Masks init to 0.0 (all active) by default.
- Load a pre-trained baseline, then run IMP: prune → fine-tune → repeat.
- Record both `(Step, Saliency, RemainingParams, Accuracy)` (min saliency) and avg saliency at each pruning step; save via `save_dict_to_csv` using standardised column names from `nplh_run_context`.
- CSV output goes to `neural_pruning_law/final_data/<timestamped_run_folder>/`.

### ResNet50 CIFAR-10 NPLH Experiment

**Files in `src/resnet50_cifar10/`:**
- `train_resnet50_cifar10_nplh.py` – device-agnostic baseline training + all experiment functions:
  - `train_resnet50_cifar10_baseline()` – trains from scratch (160 epochs, SGD + MultiStepLR), saves as `"resnet50_cifar10_nplh_baseline"` in `networks_baseline/`.
  - `run_static_pruning(baseline_name, run_ctx)` – static magnitude pruning, no retraining, 5% per step → 0.2% remaining.
  - `run_imp_magnitude(baseline_name, run_ctx)` – IMP magnitude, 5% per step, 200 epochs, cosine LR.
  - `run_imp_taylor(baseline_name, run_ctx)` – Taylor criterion, same schedule.
  - `run_gradient_pruning(baseline_name, run_ctx)` – gradient magnitude criterion, same schedule.
  - `run_random_regrowth(baseline_name, run_ctx)` – random prune + gradient-guided regrowth, same schedule, `oversample_factor=2.0`.
- `run_resnet50_cifar10_nplh_experiment.py` – trains baseline from scratch, runs static + IMP magnitude + Taylor, saves 6 CSVs to one timestamped folder.
- `run_resnet50_cifar10_new_policies.py` – uses the already-saved baseline (no retraining), runs gradient + random_regrowth only, saves 4 CSVs.

**Run from project root:**
```bash
# Full experiment (trains baseline + static + IMP magnitude + Taylor):
python -m src.resnet50_cifar10.run_resnet50_cifar10_nplh_experiment

# New policies only (requires existing baseline in networks_baseline/):
python -m src.resnet50_cifar10.run_resnet50_cifar10_new_policies
```

**Plotting (`neural_pruning_law/plot_resnet50_cifar10.py`):**
Auto-picks the latest `resnet50_cifar10` run folder and generates PDFs next to the CSVs:
- `per_method_{magnitude,taylor,static,gradient,random_regrowth}.pdf` — min + avg saliency for one method
- `joint_min_saliency.pdf`, `joint_avg_saliency.pdf` — all methods overlaid
- `overlap_static_magnitude.pdf` — static vs magnitude, both saliency types

```bash
python -m neural_pruning_law.plot_resnet50_cifar10
```

### Experiment Package Conventions

Each experiment lives in its own package under `src/` (e.g., `src/resnet50_cifar10/`). The pattern:
- `model_class.py` / `*_class.py` – model definition with `registered_layers`.
- `*_attributes.py` – model-specific configs (layer sizes, etc.).
- `train_scratch_*.py` – trains a dense baseline from scratch.
- `train_pruned_*.py` – runs Hyperflux pruning (takes `TrainingConfigsWithResume`).
- `train_NPLH_IMP_*.py` – runs IMP for NPLH data collection.
- `train_*_nplh.py` – device-agnostic NPLH multi-policy experiments (newer pattern).
- `run_*_nplh_experiment.py` – orchestrates train + multiple pruning experiments.
- `run_existing_*.py` – evaluates a saved model.

Networks intentionally are **not** shared across experiment packages to avoid cascading bugs from dataset-specific architectural differences (e.g., ResNet first conv kernel size differs between CIFAR and ImageNet).

### Model Save/Load

Baseline models are saved under `networks_baseline/`, pruned under `networks_pruned/`. The model class typically exposes a `.save(name, path)` / `.load(name, path)` interface. `prefix_path_with_root(path)` prepends the project root (detected via `.root` marker file).

### Device Handling

`get_device()` returns CUDA if available, else CPU. The legacy `train_mixed_baseline` uses `GradScaler('cuda')` and is **not** device-agnostic. Newer experiment files (e.g., variable LeNet) are fully device-agnostic.

### WandB

All wandb calls are gated on `WANDB_REGISTER = True` in `configs_general.py`. Custom training loops in NPLH experiments bypass wandb entirely.
