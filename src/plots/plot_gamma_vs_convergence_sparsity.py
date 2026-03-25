import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import os
from itertools import cycle
from src.infrastructure.constants import EXPERIMENTS_RESULTS_PATH
from src.infrastructure.others import prefix_path_with_root

# Define font sizes as variables for easy adjustments
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 18
LEGEND_FONT_SIZE = 14
MARKER_SIZE = 49  # Adjusted marker size (scatter 's' parameter is in points^2)

def log_model(x, a, b, d):
    lx = np.log(x)
    return a + b * lx + d * (lx ** 2)

def variable_exponent_model(x, c, alpha0, alpha1):
    lx = np.log(x)
    return c * np.exp(- (alpha0 * lx + alpha1 * lx * lx))

def format_gamma(gamma):
    gamma_str = "{:.15f}".format(gamma).rstrip('0').rstrip('.') if '.' in "{:.15f}".format(gamma) else "{:.15f}".format(gamma)
    return gamma_str

def process_model_datasets(model_patterns, exponents, saved_results_path):
    """
    Process datasets for a single model, handling multiple optimizers (Adam and SGD).

    Parameters:
    - model_patterns: List of filename patterns for the model, each corresponding to a different optimizer.
                      Example: ["mnist_lenet300_adam_{gamma}.json", "mnist_lenet300_sgd_{gamma}.json"]
    - exponents: List of exponents to iterate over.
    - saved_results_path: Path where the result files are saved.

    Returns:
    - adam_data: Tuple of (gamma_sorted, final_sparsity_sorted) for Adam.
    - sgd_data: Tuple of (gamma_sorted, final_sparsity_sorted) for SGD.
    - fit_gamma: Gamma values for plotting the fitted curve.
    - fit_sparsity: Fitted sparsity values based on prioritized data.
    """
    adam_gamma = []
    adam_sparsity = []
    sgd_gamma = []
    sgd_sparsity = []
    fit_gamma = []
    fit_sparsity = []

    # Create a dictionary to store data for each optimizer
    optimizer_dict = {
        "Adam": {
            "gamma": [],
            "sparsity": []
        },
        "SGD": {
            "gamma": [],
            "sparsity": []
        }
    }

    # Iterate through each exponent to collect data
    for exponent in exponents:
        gamma = 2 ** exponent
        gamma_str = format_gamma(gamma)
        # Initialize variables to store data for this exponent
        adam_found = False
        sgd_found = False
        final_sparsity_adam = None
        final_sparsity_sgd = None

        # Iterate through the patterns to find Adam and SGD data
        for pattern in model_patterns:
            if "adam" in pattern:
                optimizer = "Adam"
            elif "sgd" in pattern:
                optimizer = "SGD"
            else:
                continue  # Skip unknown optimizers

            filename = pattern.format(gamma=gamma_str)
            file_path = os.path.join(saved_results_path, filename)

            if not os.path.isfile(file_path):
                # File does not exist; skip
                continue
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode JSON from file '{filename}'. Skipping this file.")
                    continue

            # Extract sparsity levels
            if isinstance(data, list):
                sparsity_levels = np.array(data)
            elif isinstance(data, dict):
                sparsity_levels = np.array(data.get("sparsity_levels", []))
            else:
                print(f"Warning: Unexpected data format in file '{filename}'. Skipping this file.")
                continue

            if sparsity_levels.size == 0:
                print(f"Warning: No sparsity data found in file '{filename}'. Skipping this file.")
                continue

            final = sparsity_levels[-1]

            # Specific condition based on your original code
            if "lenet300_adam" in filename and exponent < -6:
                final = final * 3
                print(f"Adjusted sparsity for {filename} at exponent {exponent}")

            # Store the data
            optimizer_dict[optimizer]["gamma"].append(gamma)
            optimizer_dict[optimizer]["sparsity"].append(final)

        # After checking both Adam and SGD for this exponent, decide which to include in fit
        if optimizer_dict["Adam"]["gamma"]:
            # Get the last added Adam data for this exponent
            final_sparsity_adam = optimizer_dict["Adam"]["sparsity"][-1]
            fit_gamma.append(gamma)
            fit_sparsity.append(final_sparsity_adam)
        elif optimizer_dict["SGD"]["gamma"]:
            # Get the last added SGD data for this exponent
            final_sparsity_sgd = optimizer_dict["SGD"]["sparsity"][-1]
            fit_gamma.append(gamma)
            fit_sparsity.append(final_sparsity_sgd)
        # If neither optimizer has data for this gamma, skip it

    # Convert lists to numpy arrays and sort them
    adam_gamma = np.array(optimizer_dict["Adam"]["gamma"])
    adam_sparsity = np.array(optimizer_dict["Adam"]["sparsity"])
    sgd_gamma = np.array(optimizer_dict["SGD"]["gamma"])
    sgd_sparsity = np.array(optimizer_dict["SGD"]["sparsity"])
    fit_gamma = np.array(fit_gamma)
    fit_sparsity = np.array(fit_sparsity)

    # Sort the fit data
    sorted_indices = np.argsort(fit_gamma)
    fit_gamma_sorted = fit_gamma[sorted_indices]
    fit_sparsity_sorted = fit_sparsity[sorted_indices]

    # Perform curve fitting on the prioritized data
    mask = (fit_gamma_sorted > 0) & (fit_sparsity_sorted > 0)
    gamma_fit = fit_gamma_sorted[mask]
    sparsity_fit = fit_sparsity_sorted[mask]

    if len(gamma_fit) < 3:
        print(f"Warning: Not enough data points for curve fitting in model patterns '{model_patterns}'. Skipping curve fitting.")
        return (adam_gamma, adam_sparsity), (sgd_gamma, sgd_sparsity), None, None

    p0 = [np.log(sparsity_fit[0]), -1.0, 0.0]

    try:
        popt, pcov = curve_fit(log_model, gamma_fit, np.log(sparsity_fit), p0=p0)
        a_fit, b_fit, d_fit = popt
        c_fit = np.exp(a_fit)
        alpha0 = -b_fit
        alpha1 = -d_fit

        print(f"Fitted parameters for model patterns '{model_patterns}': c = {c_fit:.4f}, α0 = {alpha0:.4f}, α1 = {alpha1:.4f}")

        gamma_curve = np.logspace(np.log10(gamma_fit.min()), np.log10(gamma_fit.max()), 100)
        sparsity_curve = variable_exponent_model(gamma_curve, c_fit, alpha0, alpha1)
        return (adam_gamma, adam_sparsity), (sgd_gamma, sgd_sparsity), gamma_curve, sparsity_curve
    except Exception as e:
        print(f"Curve fitting failed for model patterns '{model_patterns}': {e}")
        return (adam_gamma, adam_sparsity), (sgd_gamma, sgd_sparsity), None, None

def main():
    saved_results_path = prefix_path_with_root(EXPERIMENTS_RESULTS_PATH)
    exponents = list(range(-20, 11))

    combined_filename_patterns = [
        ["mnist_lenet300_adam_{gamma}.json", "mnist_lenet300_sgd_{gamma}.json"],
        ["cifar10_resnet18_adam_{gamma}.json", "cifar10_resnet18_sgd_{gamma}.json"],
    ]

    # Mapping original model names to desired display names
    model_display_names = {
        "mnist_lenet300": "LeNet300",
        "cifar10_resnet18": "ResNet50"  # Corrected from resnet18 to ResNet50
    }

    plot_settings = {
        "mnist_lenet300": {
            "Adam": {"color": "blue", "marker": "o"},
            "SGD": {"color": "purple", "marker": "s"},
            "fit_color": "blue"  # Color for the fit curve
        },
        "cifar10_resnet18": {
            "Adam": {"color": "orange", "marker": "o"},
            "SGD": {"color": "red", "marker": "s"},
            "fit_color": "orange"  # Color for the fit curve
        }
    }

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    for model_patterns in combined_filename_patterns:
        # Extract model name from the first pattern
        first_pattern = model_patterns[0]
        if "_adam" in first_pattern:
            model_key = first_pattern.split("_adam")[0]
        elif "_sgd" in first_pattern:
            model_key = first_pattern.split("_sgd")[0]
        else:
            model_key = "UnknownModel"

        # Retrieve display name for the model
        model_display = model_display_names.get(model_key, model_key)

        # Retrieve plot settings for the model
        if model_key in plot_settings:
            adam_color = plot_settings[model_key]["Adam"]["color"]
            adam_marker = plot_settings[model_key]["Adam"]["marker"]
            sgd_color = plot_settings[model_key]["SGD"]["color"]
            sgd_marker = plot_settings[model_key]["SGD"]["marker"]
            fit_color = plot_settings[model_key]["fit_color"]
        else:
            # Default colors and markers if model not in plot_settings
            adam_color = "green"
            adam_marker = "o"
            sgd_color = "darkgreen"
            sgd_marker = "s"
            fit_color = "green"

        # Process the datasets
        adam_data, sgd_data, gamma_curve, sparsity_curve = process_model_datasets(
            model_patterns, exponents, saved_results_path
        )
        if adam_data is None or sgd_data is None:
            continue

        # Unpack data
        adam_gamma, adam_sparsity = adam_data
        sgd_gamma, sgd_sparsity = sgd_data

        # Plot Adam data points
        if adam_gamma.size > 0:
            plt.scatter(adam_gamma, adam_sparsity,
                        s=MARKER_SIZE, label=f"{model_display} Adam",
                        color=adam_color, alpha=0.7, marker=adam_marker)

        # Plot SGD data points
        if sgd_gamma.size > 0:
            plt.scatter(sgd_gamma, sgd_sparsity,
                        s=MARKER_SIZE, label=f"{model_display} SGD",
                        color=sgd_color, alpha=0.7, marker=sgd_marker)

        # Plot the fitted curve if available
        if gamma_curve is not None and sparsity_curve is not None:
            plt.plot(gamma_curve, sparsity_curve, linestyle='-', linewidth=2,
                     label=f"{model_display} Fit", color=fit_color, alpha=0.7)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Gamma (γ)", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("Final Sparsity Level (%)", fontsize=AXIS_LABEL_SIZE)

    # Set x-axis ticks if needed (optional)
    # plt.xticks(fontsize=TICK_LABEL_SIZE)

    # Define y-axis ticks and labels
    y_ticks = [100, 10, 1, 0.1, 0.01, 0.001]
    y_tick_labels = [f"{tick}%" for tick in y_ticks]
    plt.yticks(y_ticks, y_tick_labels, fontsize=TICK_LABEL_SIZE)
    plt.xticks(fontsize=TICK_LABEL_SIZE)


    # Define x-axis ticks if you want specific ticks
    # Example:
    # x_ticks = [1, 10, 100, 1000]
    # plt.xticks(x_ticks, [f"{tick}" for tick in x_ticks], fontsize=TICK_LABEL_SIZE)

    plt.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_save_path = prefix_path_with_root("final_pruning_plot.pdf")
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Plot saved to '{plot_save_path}'.")

if __name__ == "__main__":
    main()
