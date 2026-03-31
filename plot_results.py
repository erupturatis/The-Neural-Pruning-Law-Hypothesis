from src.plots.nplh_plots import plot_saliency_loglog, SeriesSpec

# CSVs from run 20260328_1703_1guy05isw9
GRADIENT_CSV = "nplh_data/20260328_1703_1guy05isw9/lenet_mnist_alpha1.0_GradientSaliencyMeasurementPolicy_20260328_170353_vcuc.csv"
MAGNITUDE_CSV = "nplh_data/20260328_1703_1guy05isw9/lenet_mnist_alpha1.0_MagnitudeSaliencyMeasurementPolicy_20260328_170353_uh87.csv"

# Plot 1: Magnitude — all weights vs contributing on the same axes
plot_saliency_loglog(
    series=[
        SeriesSpec(MAGNITUDE_CSV, label="all weights (nominal density)",       saliency_col="avg_saliency",              x_col="density"),
        SeriesSpec(MAGNITUDE_CSV, label="contributing only (excl. dead-grad)", saliency_col="avg_saliency_contributing", x_col="contributing"),
    ],
    out_path="plots/magnitude_all_vs_contributing.png",
    title="NPLH: Magnitude — all vs contributing",
    x_label="Remaining weights (%)",
)

# Plot 2: Gradient — all weights vs contributing on the same axes
plot_saliency_loglog(
    series=[
        SeriesSpec(GRADIENT_CSV, label="all weights (nominal density)",       saliency_col="avg_saliency",              x_col="density"),
        SeriesSpec(GRADIENT_CSV, label="contributing only (excl. dead-grad)", saliency_col="avg_saliency_contributing", x_col="contributing"),
    ],
    out_path="plots/gradient_all_vs_contributing.png",
    title="NPLH: Gradient — all vs contributing",
    x_label="Remaining weights (%)",
)
