import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # For custom legend
from src.infrastructure.others import prefix_path_with_root

AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 18
LEGEND_FONT_SIZE = 14
MARKER_SIZE = 7

def load_data(filename, max_epoch=None):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        values = json.load(file)

    x_data = np.arange(1, len(values) + 1, dtype=float)
    y_data = np.array(values, dtype=float)

    if max_epoch is not None:
        x_data = x_data[:max_epoch]
        y_data = y_data[:max_epoch]

    return x_data, y_data

def plot_all(files):
    """Plot all runs with different colors representing different optimizers and networks,
    and add a horizontal line at each run's minimum y-value."""
    plt.figure(figsize=(12, 8))

    # Define color palette (extend as needed)
    color_palette = [
        'blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray'
    ]

    # Mapping from (network, optimizer, settings) to color
    run_color_mapping = {}

    # Maximum number of epochs to plot
    max_ep = 20000

    # Iterate over each file and assign a unique color
    for i, filename in enumerate(files):
        # Extract metadata from filename
        # Example filename: "experiments_outputs/mnist_lenet300_adam_1_LR_0.005.json"
        parts = filename.split('/')[-1].replace('.json', '').split('_')

        # Identify network
        if 'mnist' in parts:
            network = 'LeNet300'
        elif 'cifar10' in parts:
            network = 'ResNet18'
        else:
            network = ''

        # Identify optimizer
        if 'sgd' in parts:
            optimizer = 'SGD'
        elif 'adam' in parts:
            optimizer = 'Adam'
        else:
            optimizer = ''

        # Identify learning rate from the filename
        lr_value = None
        for j, part in enumerate(parts):
            if part == 'LR' and j < len(parts) - 1:
                lr_value = parts[j + 1]
                break

        # Construct settings string with learning rate
        settings = []
        if lr_value is not None:
            settings.append(f'LR={lr_value}')
        settings_str = ', '.join(settings)

        # Construct the run key for the legend
        run_key = f"{network}"
        if settings_str:
            run_key += f", {settings_str}"

        color = color_palette[i % len(color_palette)]
        run_color_mapping[run_key] = color

        x, y = load_data(filename, max_ep)

        def log_sample(x, y, num_points=50):
            idxs = np.unique(
                np.round(np.logspace(0, np.log10(len(x)), num_points))
            ).astype(int)
            idxs = np.clip(idxs, 1, len(x)) - 1

            return x[idxs], y[idxs]

        x_sample, y_sample = log_sample(x, y, num_points=50)

        # Plot the sampled data
        plt.plot(
            x_sample, y_sample,
            color=color,
            marker='o',
            linestyle='--',
            linewidth=2,
            markersize=MARKER_SIZE,
            label=run_key
        )

        y_min = y.min()

        plt.axhline(
            y=y_min,
            color=color,
            linestyle=':',
            linewidth=1.5,
            alpha=0.7,
            label=f"{run_key} Min"
        )

    # Set both axes to logarithmic scale
    # plt.xscale('log')
    plt.yscale('log')

    # Set custom ticks for Iterations (x-axis)
    epoch_ticks = [1, 500, 1000, 2000, 4000]
    plt.xticks(epoch_ticks, [str(tick) for tick in epoch_ticks])

    # Set custom ticks for Sparsity (y-axis)
    sparsity_ticks = [100, 10, 1, 0.1]
    sparsity_labels = ['100', '10', '1', '0.1']
    plt.yticks(sparsity_ticks, sparsity_labels)

    # Set axis labels with increased font size
    plt.xlabel('Iteration', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Sparsity (%)', fontsize=AXIS_LABEL_SIZE)

    # Retrieve existing handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    run_handles = []
    run_labels = []

    for handle, label in zip(handles, labels):
        run_handles.append(handle)
        run_labels.append(label)

    # Sort the runs for consistent legend ordering
    sorted_runs = sorted(zip(run_labels, run_handles), key=lambda x: x[0])
    run_labels_sorted, run_handles_sorted = zip(*sorted_runs) if sorted_runs else ([], [])

    # Define legend elements (excluding minimum lines)
    legend_elements = [
        Line2D([0], [0], color=handle.get_color(), linestyle='-', linewidth=3, label=label)
        for label, handle in zip(run_labels_sorted, run_handles_sorted)
        if not label.endswith("Min")
    ]

    plt.legend(handles=legend_elements, fontsize=LEGEND_FONT_SIZE, loc='upper right')

    # Improve layout and display grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Save the figure as SVG and PDF
    plt.savefig('plot.svg', format='svg')
    plt.savefig('plot.pdf', format='pdf')

    # Display the plot
    plt.show()

# Define file paths
file_tuples = [
    "experiments_outputs/mnist_lenet300_1_LR_5e-03",
    "experiments_outputs/mnist_lenet300_1_LR_5e-04",
    "experiments_outputs/mnist_lenet300_1_LR_5e-05",
]

plt.rcParams['axes.labelsize'] = AXIS_LABEL_SIZE      # Increased to 20
plt.rcParams['xtick.labelsize'] = TICK_LABEL_SIZE     # Increased to 18
plt.rcParams['ytick.labelsize'] = TICK_LABEL_SIZE     # Increased to 18
plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE   # Increased to 14
plt.rcParams['font.size'] = 18                        # Base font size

# Prefix paths as needed
file_tuples = [
    prefix_path_with_root(fl) for fl in file_tuples
]

# Plot the data
plot_all(file_tuples)
