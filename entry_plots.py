from src.plots.nplh_plots import plot_saliency_loglog, SeriesSpec


def example_saliency_overlay():
    """
    Example: overlay several saliency CSVs on one log-log plot.

    Edit the SeriesSpec list below to point at your CSV files.
    - csv_path:    path to the CSV produced by NplhSeries.save()
    - label:       legend label (optional — defaults to the filename)
    - saliency_col: 'avg_saliency' or 'min_saliency'
    """
    folder_path = "nplh_data/20260326_1151_mvgzsmf4zq"  # edit this to your folder path containing the CSVs
    series = [
        SeriesSpec(
            csv_path=f"nplh_data/20260326_1236_lcdtsdms8j/lenet_mnist_alpha0.5_TaylorSaliencyMeasurementPolicy_20260326_123625_392q.csv",
            label="Magnitude (alpha=1.0)",
            saliency_col="min_saliency",
        ),
        # SeriesSpec(
        #     csv_path="nplh_data/20260326_1151_mvgzsmf4zq/lenet_mnist_alpha0.5_TaylorSaliencyMeasurementPolicy_20260326_115140_fapk.csv",
        #     label="Taylor (alpha=1.0)",
        #     saliency_col="min_saliency",
        # ),
        # SeriesSpec(
        #     csv_path="nplh_data/20260326_1151_mvgzsmf4zq/lenet_mnist_alpha0.5_GradientSaliencyMeasurementPolicy_20260326_115140_wibx.csv",
        #     label="Gradient (alpha=1.0)",
        #     saliency_col="avg_saliency",
        # ),
    ]

    plot_saliency_loglog(
        series=series,
        title="NPLH: Saliency vs Density — Random Pruning (LeNet, MNIST)",
        out_path="plots/saliency_overlay.png",   # set to None to show interactively
    )


if __name__ == "__main__":
    example_saliency_overlay()
