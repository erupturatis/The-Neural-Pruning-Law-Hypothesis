import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import os
import argparse
from matplotlib.ticker import FuncFormatter

# --- 1. Model Definitions for Curve Fitting ---

def variable_exponent_model(x, c, alpha0, alpha1):
    """
    The model describing the relationship between saliency (γ) and sparsity (S).
    S(γ) = c * exp(-(α₀*log(γ) + α₁*(log(γ))²))
    """
    lx = np.log(x)
    return c * np.exp(- (alpha0 * lx + alpha1 * lx * lx))

def centered_log_model(log_x_centered, a_prime, b_prime, d_prime):
    """
    A numerically stable model for fitting.
    It fits a polynomial to log(x) values that have been centered around their mean.
    This function is used ONLY for the scipy.optimize.curve_fit call.
    """
    return a_prime + b_prime * log_x_centered + d_prime * (log_x_centered ** 2)

# --- 2. Data Loading Function ---
def read_pruning_data(csv_filepath, saliency_col, remaining_col):
    """
    Reads a CSV file and extracts saliency and remaining weight percentage columns.
    """
    saliency_scores = []
    remaining_percentages = []

    if not os.path.isfile(csv_filepath):
        print(f"Error: The file '{csv_filepath}' was not found.")
        return None, None

    with open(csv_filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if saliency_col not in reader.fieldnames or remaining_col not in reader.fieldnames:
            print(f"Error: One or both specified columns were not found in the CSV.")
            print(f"  - Expected Saliency Column: '{saliency_col}'")
            print(f"  - Expected Remaining Weights Column: '{remaining_col}'")
            print(f"  - Available columns: {', '.join(reader.fieldnames)}")
            return None, None
        for row in reader:
            try:
                saliency_scores.append(float(row[saliency_col]))
                remaining_percentages.append(float(row[remaining_col]))
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process row in CSV: {row}. Error: {e}")
                continue
    return saliency_scores, remaining_percentages

# --- 3. Main Execution Block ---
def main():
    """
    Main function to parse arguments, load data, perform curve fitting,
    and generate the plot.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Plot sparsity vs. saliency score and fit a curve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--file', '-f', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--saliency_col', '-sa', type=str, default='Saliency', help="Name of the column for the saliency score.")
    parser.add_argument('--remaining_col', '-sp', type=str, default='Remaining', help="Name of the column for the remaining weights percentage.")
    args = parser.parse_args()

    # --- Plot Styling Configuration ---
    AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_FONT_SIZE, TITLE_FONT_SIZE, MARKER_SIZE = 16, 14, 12, 18, 60

    # --- Data Loading ---
    saliency_scores, remaining_percentages = read_pruning_data(args.file, args.saliency_col, args.remaining_col)
    if saliency_scores is None:
        return

    saliency_scores = np.array(saliency_scores, dtype=float)
    remaining_percentages = np.array(remaining_percentages, dtype=float)

    # --- CORE CHANGE: Convert remaining percentage to sparsity ---
    sparsity = remaining_percentages

    # --- Curve Fitting (Numerically Stable Version) ---
    # Filter data: saliency and sparsity must be positive for log scale fitting
    mask = (saliency_scores > 0) & (sparsity > 0)
    if np.sum(mask) < 3:
        print("Warning: Not enough valid data points (need at least 3) for curve fitting.")
        return

    saliency_fit = saliency_scores[mask]
    sparsity_fit = sparsity[mask]
    
    # Transform y-data to log space for fitting
    log_sparsity = np.log(sparsity_fit)

    threshold_curve, sparsity_curve = None, None
    try:
        # 1. Center the independent variable (log of saliency) around its mean for stability
        log_saliency = np.log(saliency_fit)
        log_x_center = np.mean(log_saliency)
        centered_log_saliency = log_saliency - log_x_center

        # 2. Provide a robust initial guess for the centered model
        p0_centered = [np.mean(log_sparsity), -1.0, 0.0]

        # 3. Fit using the STABLE centered_log_model
        popt_prime, _ = curve_fit(centered_log_model, centered_log_saliency, log_sparsity, p0=p0_centered)
        a_prime, b_prime, d_prime = popt_prime

        # 4. Convert the stable parameters (a', b', d') back to the original model's parameters (a, b, d)
        d_fit = d_prime
        b_fit = b_prime - 2 * d_prime * log_x_center
        a_fit = a_prime - b_prime * log_x_center + d_prime * (log_x_center ** 2)

        # 5. Convert to your final presentation model's parameters (c, α₀, α₁)
        c_fit = np.exp(a_fit)
        alpha0_fit = -b_fit
        alpha1_fit = -d_fit

        # c_fit = 20.0786
        # alpha0_fit = 0.99
        # alpha1_fit = 0.1057


        print("--- Curve Fit Results ---")
        print(f"Fitted parameters: c = {c_fit:.4f}, α₀ = {alpha0_fit:.4f}, α₁ = {alpha1_fit:.4f}")

        # Generate the curve using the final model
        threshold_curve = np.logspace(np.log10(saliency_fit.min()), np.log10(saliency_fit.max()), 200)
        sparsity_curve = variable_exponent_model(threshold_curve, c_fit, alpha0_fit, alpha1_fit)

    except Exception as e:
        print(f"Curve fitting failed: {e}")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    # Plot the calculated sparsity
    plt.scatter(saliency_scores, sparsity, s=MARKER_SIZE, label="Data Points", color='blue', alpha=0.7, zorder=5)
    if threshold_curve is not None:
        plt.plot(threshold_curve, sparsity_curve, linestyle='-', linewidth=2.5, label="Fitted Curve", color='red', alpha=0.8)

    plt.xscale('log')
    plt.yscale('log')
    # Y-axis limits are still appropriate for percentage
    # plt.ylim(0.1, 100)
    
    xlabel = args.saliency_col.replace('_', ' ').title()
    # Update y-axis label and title to reflect sparsity
    ylabel = "Sparsity (%)"
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    plt.title(f"Sparsity vs. {xlabel} for ResNet-50 on CIFAR-10", fontsize=TITLE_FONT_SIZE, pad=20)
    
    # Formatter for percentage on the y-axis remains correct
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}%'))
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()

    plot_filename = "sparsity_vs_saliency_fit.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    print(f"\nPlot successfully saved to {os.path.abspath(plot_filename)}")
    plt.show()

if __name__ == "__main__":
    main()