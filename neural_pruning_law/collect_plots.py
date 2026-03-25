"""
collect_plots.py
================
Renames all experiment PDFs to self-describing names and copies them
into neural_pruning_law/plots/.

Run from project root:
    python -m neural_pruning_law.collect_plots
"""

import os
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINAL_DATA = os.path.join(_SCRIPT_DIR, "final_data")
_OUT_DIR    = os.path.join(_SCRIPT_DIR, "plots")
os.makedirs(_OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Rename map: (run_folder_substring, old_filename) → new_filename
# ---------------------------------------------------------------------------
RENAMES = {
    # ── LeNet-300 MNIST – all policies (magnitude / taylor / static) ─────────
    ("lenet300_all_policies", "lenet300_min_saliency_joint.pdf"):
        "lenet300_mnist__all_policies__min_saliency__joint.pdf",
    ("lenet300_all_policies", "lenet300_avg_saliency_joint.pdf"):
        "lenet300_mnist__all_policies__avg_saliency__joint.pdf",
    ("lenet300_all_policies", "lenet300_min_saliency_panels.pdf"):
        "lenet300_mnist__all_policies__min_saliency__per_method.pdf",
    ("lenet300_all_policies", "lenet300_avg_saliency_panels.pdf"):
        "lenet300_mnist__all_policies__avg_saliency__per_method.pdf",

    # ── ResNet50 CIFAR-10 – all policies (magnitude / taylor / static) ───────
    ("resnet50_cifar10_all_policies", "per_method_magnitude.pdf"):
        "resnet50_cifar10__IMP_magnitude__min_and_avg_saliency.pdf",
    ("resnet50_cifar10_all_policies", "per_method_taylor.pdf"):
        "resnet50_cifar10__taylor__min_and_avg_saliency.pdf",
    ("resnet50_cifar10_all_policies", "per_method_static.pdf"):
        "resnet50_cifar10__static_no_retrain__min_and_avg_saliency.pdf",
    ("resnet50_cifar10_all_policies", "joint_min_saliency.pdf"):
        "resnet50_cifar10__all_policies__min_saliency__joint.pdf",
    ("resnet50_cifar10_all_policies", "joint_avg_saliency.pdf"):
        "resnet50_cifar10__all_policies__avg_saliency__joint.pdf",
    ("resnet50_cifar10_all_policies", "overlap_static_magnitude.pdf"):
        "resnet50_cifar10__static_vs_IMP_magnitude__saliency_overlap.pdf",

    # ── ResNet50 CIFAR-10 – new policies (gradient / random_regrowth) ────────
    ("resnet50_cifar10_new_policies", "per_method_gradient.pdf"):
        "resnet50_cifar10__gradient_magnitude__min_and_avg_saliency.pdf",
    ("resnet50_cifar10_new_policies", "per_method_random_regrowth.pdf"):
        "resnet50_cifar10__random_regrowth__min_and_avg_saliency.pdf",
    ("resnet50_cifar10_new_policies", "joint_min_saliency.pdf"):
        "resnet50_cifar10__gradient_vs_random_regrowth__min_saliency__joint.pdf",
    ("resnet50_cifar10_new_policies", "joint_avg_saliency.pdf"):
        "resnet50_cifar10__gradient_vs_random_regrowth__avg_saliency__joint.pdf",
    ("resnet50_cifar10_new_policies", "accuracy_comparison.pdf"):
        "resnet50_cifar10__gradient_vs_random_regrowth__accuracy_vs_density.pdf",

    # ── ResNet50 CIFAR-10 – collapse (IMP + static to collapse density) ──────
    ("resnet50_cifar10_collapse", "per_method_collapse_imp.pdf"):
        "resnet50_cifar10__collapse_IMP__saliency_min_and_avg.pdf",
    ("resnet50_cifar10_collapse", "per_method_collapse_static.pdf"):
        "resnet50_cifar10__collapse_static_no_retrain__saliency_min_and_avg.pdf",
    ("resnet50_cifar10_collapse", "joint_min_saliency.pdf"):
        "resnet50_cifar10__collapse_IMP_vs_static__min_saliency__joint.pdf",
    ("resnet50_cifar10_collapse", "joint_avg_saliency.pdf"):
        "resnet50_cifar10__collapse_IMP_vs_static__avg_saliency__joint.pdf",
    ("resnet50_cifar10_collapse", "accuracy_comparison.pdf"):
        "resnet50_cifar10__collapse_IMP_vs_static__accuracy_vs_density.pdf",
    ("resnet50_cifar10_collapse", "saliency_vs_accuracy_imp.pdf"):
        "resnet50_cifar10__collapse_IMP__saliency_and_accuracy_twin_axis.pdf",

    # ── ResNet50 CIFAR-10 – intrinsic saliency (random prune ± training) ─────
    ("resnet50_cifar10_intrinsic_saliency", "per_method_random_prune_trained.pdf"):
        "resnet50_cifar10__intrinsic__random_pruning_with_training__saliency_and_accuracy.pdf",
    ("resnet50_cifar10_intrinsic_saliency", "per_method_random_prune_static.pdf"):
        "resnet50_cifar10__intrinsic__random_pruning_static_control__saliency_and_accuracy.pdf",
    ("resnet50_cifar10_intrinsic_saliency", "joint_avg_saliency.pdf"):
        "resnet50_cifar10__intrinsic__trained_vs_static__avg_saliency__joint.pdf",
    ("resnet50_cifar10_intrinsic_saliency", "accuracy_comparison.pdf"):
        "resnet50_cifar10__intrinsic__trained_vs_static__accuracy_vs_density.pdf",
}

# ---------------------------------------------------------------------------
# Find each run folder, rename in-place, copy to _OUT_DIR
# ---------------------------------------------------------------------------

run_dirs = {
    d: os.path.join(_FINAL_DATA, d)
    for d in os.listdir(_FINAL_DATA)
    if os.path.isdir(os.path.join(_FINAL_DATA, d))
}

for (run_substr, old_name), new_name in RENAMES.items():
    # find the matching run folder that contains the old or new filename
    matches = sorted(d for d in run_dirs if run_substr in d)
    if not matches:
        print(f"  [skip] no run folder matching '{run_substr}'")
        continue
    # prefer the folder that actually has the file (old or new name)
    run_dir = None
    for candidate in reversed(matches):
        d = run_dirs[candidate]
        if os.path.exists(os.path.join(d, old_name)) or os.path.exists(os.path.join(d, new_name)):
            run_dir = d
            break
    if run_dir is None:
        print(f"  [missing] no folder with '{old_name}' or '{new_name}' for '{run_substr}'")
        continue

    old_path = os.path.join(run_dir, old_name)
    new_path = os.path.join(run_dir, new_name)

    if not os.path.exists(old_path):
        # already renamed in a previous run
        if os.path.exists(new_path):
            shutil.copy2(new_path, os.path.join(_OUT_DIR, new_name))
            print(f"  [already renamed] copied  {new_name}")
        else:
            print(f"  [missing] {old_path}")
        continue

    os.rename(old_path, new_path)
    shutil.copy2(new_path, os.path.join(_OUT_DIR, new_name))
    print(f"  renamed + copied  {new_name}")

print(f"\nDone. All PDFs collected in:\n  {_OUT_DIR}")
print(f"Total files: {len(os.listdir(_OUT_DIR))}")
