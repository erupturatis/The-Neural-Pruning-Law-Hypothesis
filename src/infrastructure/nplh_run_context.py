"""
nplh_run_context.py
===================
Single source of truth for NPLH experiment run management.

Provides
--------
- CSV column name constants (COL_*)
- Saliency-type and method tags for filenames (SAL_*, METHOD_*)
- NplhRunContext: timestamped run folder + txt description + canonical paths

Usage
-----
    from src.infrastructure.nplh_run_context import (
        NplhRunContext, COL_STEP, COL_REMAINING, COL_SALIENCY, COL_ACCURACY,
        SAL_MIN, SAL_AVG,
        METHOD_IMP_MAGNITUDE, METHOD_IMP_STATIC,
        METHOD_IMP_TAYLOR, METHOD_GRADIENT, METHOD_RANDOM_REGROWTH,
    )

    ctx = NplhRunContext.create(
        run_name="resnet50_imagenet_static",
        description={
            "model":         "ResNet50",
            "dataset":       "ImageNet1k",
            "method":        METHOD_IMP_STATIC,
            "pruning_rate":  0.10,
            "target_sparsity": 0.999,
            "notes":         "No retraining, pure magnitude cuts.",
        },
    )

    # Canonical CSV path:  {folder}/resnet50_imagenet_min_static.csv
    path = ctx.csv_path("resnet50", "imagenet", SAL_MIN, METHOD_IMP_STATIC)

    # Canonical plot path: {folder}/joint.pdf
    out  = ctx.plot_path("joint")
"""

import os
from datetime import datetime
from src.infrastructure.others import prefix_path_with_root

# ---------------------------------------------------------------------------
# CSV column name constants  — single source of truth
# ---------------------------------------------------------------------------
COL_STEP      = "Step"           # pruning step number (or global epoch)
COL_REMAINING = "RemainingParams"  # % of weights still active
COL_SALIENCY  = "Saliency"       # saliency value (min or avg magnitude)
COL_ACCURACY  = "Accuracy"       # test accuracy in %  (optional column)

# ---------------------------------------------------------------------------
# Filename tags
# ---------------------------------------------------------------------------
SAL_MIN = "min"   # minimum saliency (pruning threshold)
SAL_AVG = "avg"   # average absolute magnitude of active weights

METHOD_IMP           = "IMP_magnitude"   # legacy alias; prefer METHOD_IMP_MAGNITUDE
METHOD_IMP_MAGNITUDE = "IMP_magnitude"   # magnitude IMP — this is the only true IMP method
METHOD_IMP_STATIC    = "static"          # magnitude pruning with no retraining (control)
METHOD_IMP_TAYLOR    = "taylor"          # first-order Taylor criterion (|w · ∂L/∂w|)
METHOD_GRADIENT        = "gradient"         # raw gradient magnitude (|∂L/∂w|)
METHOD_GRADIENT_STATIC = "gradient_static"  # gradient scoring, no retraining
METHOD_TAYLOR_STATIC   = "taylor_static"    # Taylor scoring, no retraining
METHOD_RANDOM_REGROWTH = "random_regrowth"  # random prune + gradient-guided regrowth

# Random pruning experiments (criterion = random; saliency measured separately)
METHOD_RANDOM_STATIC_MAGNITUDE = "random_static_magnitude"  # no retraining, magnitude saliency
METHOD_RANDOM_STATIC_GRADIENT  = "random_static_gradient"   # no retraining, gradient saliency
METHOD_RANDOM_STATIC_TAYLOR    = "random_static_taylor"     # no retraining, Taylor saliency
METHOD_RANDOM_RETRAIN_MAGNITUDE = "random_retrain_magnitude"  # with retraining, magnitude saliency
METHOD_RANDOM_RETRAIN_GRADIENT  = "random_retrain_gradient"   # with retraining, gradient saliency
METHOD_RANDOM_RETRAIN_TAYLOR    = "random_retrain_taylor"     # with retraining, Taylor saliency

# ---------------------------------------------------------------------------
# Internal base directory (relative to project root)
# ---------------------------------------------------------------------------
_BASE_DATA_DIR = "neural_pruning_law/final_data"


# ---------------------------------------------------------------------------
# NplhRunContext
# ---------------------------------------------------------------------------

class NplhRunContext:
    """
    Manages one experiment run: timestamped folder, description file,
    and canonical paths for CSVs and plots.

    Always create via ``NplhRunContext.create()``.
    """

    def __init__(self, run_id: str, run_name: str, folder_path: str):
        self.run_id      = run_id
        self.run_name    = run_name
        self.folder_path = folder_path

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, run_name: str, description: dict) -> "NplhRunContext":
        """
        Create a new run folder with a timestamped ID and write a
        human-readable ``run_description.txt`` inside it.

        Parameters
        ----------
        run_name:
            Short descriptive slug, e.g. ``"lenet_variable_mnist"`` or
            ``"resnet50_imagenet_static"``.  Used in the folder name.
        description:
            Arbitrary key-value pairs written verbatim to the txt file.
            Include at minimum: model, dataset, method, pruning_rate.
        """
        run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{run_id}_{run_name}"
        folder_path = prefix_path_with_root(f"{_BASE_DATA_DIR}/{folder_name}")
        os.makedirs(folder_path, exist_ok=True)

        ctx = cls(run_id=run_id, run_name=run_name, folder_path=folder_path)
        ctx._save_description(description)
        print(f"[NplhRunContext] Run folder: {folder_path}")
        return ctx

    # ------------------------------------------------------------------
    # Description file
    # ------------------------------------------------------------------

    def _save_description(self, description: dict) -> None:
        lines = [
            f"Run ID   : {self.run_id}",
            f"Run name : {self.run_name}",
            f"Created  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Experimental setup",
            "==================",
        ]
        for k, v in description.items():
            lines.append(f"  {k:<26}: {v}")
        lines += [
            "",
            "CSV format",
            "==========",
            "  Filename pattern : <model>_<dataset>_<saliency_type>_<method>.csv",
            "  Columns          :",
            f"    {COL_STEP:<16} pruning step number",
            f"    {COL_REMAINING:<16} % of weights still active after this step",
            f"    {COL_SALIENCY:<16} saliency value at this step",
            f"    {COL_ACCURACY:<16} test accuracy in % (omitted when not available)",
        ]
        txt_path = os.path.join(self.folder_path, "run_description.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[NplhRunContext] Description: {txt_path}")

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def csv_path(
        self,
        model:         str,
        dataset:       str,
        saliency_type: str,
        method:        str,
    ) -> str:
        """
        Return the absolute path for a standardised CSV file.

        Filename: ``{model}_{dataset}_{saliency_type}_{method}.csv``

        Example
        -------
        ctx.csv_path("resnet50", "imagenet", SAL_MIN, METHOD_IMP_STATIC)
        → …/20260311_142530_resnet50_imagenet_static/resnet50_imagenet_min_static.csv
        """
        filename = f"{model}_{dataset}_{saliency_type}_{method}.csv"
        return os.path.join(self.folder_path, filename)

    def plot_path(self, plot_name: str) -> str:
        """Return the absolute path for a PDF plot inside the run folder."""
        return os.path.join(self.folder_path, f"{plot_name}.pdf")
