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
            print(f"Error: One or both specified columns were not found in the CSV '{csv_filepath}'.")
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
    Main function to parse arguments, load data from two files, perform curve
    fitting for each, and generate a comparative plot.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Plot sparsity vs. saliency from two files and fit a curve to each.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # MODIFIED: Changed to accept two file paths
    parser.add_argument('--files', '-f', type=str, nargs=2, required=True,
                        metavar=('FILE1', 'FILE2'), help="Paths to the two input CSV files.")
    parser.add_argument('--saliency_col', '-sa', type=str, default='magnitude_threshold', help="Name of the column for the saliency score.")
    parser.add_argument('--remaining_col', '-sp', type=str, default='remaining_weights_percentage', help="Name of the column for the remaining weights percentage.")
    args = parser.parse_args()

    # --- Plot Styling Configuration ---
    AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_FONT_SIZE, TITLE_FONT_SIZE, MARKER_SIZE = 16, 14, 12, 18, 60
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define color scheme for the two datasets
    data_colors = ['#1f77b4', '#2ca02c']  # Muted blue and green
    fit_colors = ['#ff7f0e', '#d62728']   # Muted orange and red

    # --- MODIFIED: Loop through each provided file ---
    for i, filepath in enumerate(args.files):
        print(f"\n--- Processing File: {os.path.basename(filepath)} ---")
        
        # --- Data Loading ---
        saliency_scores, remaining_percentages = read_pruning_data(filepath, args.saliency_col, args.remaining_col)
        if saliency_scores is None:
            continue

        saliency_scores = np.array(saliency_scores, dtype=float)
        remaining_percentages = np.array(remaining_percentages, dtype=float)

        # --- Convert remaining percentage to sparsity ---
        sparsity = remaining_percentages
        
        # --- Plot the raw data points ---
        file_label = os.path.basename(filepath)
        ax.scatter(saliency_scores, sparsity, s=MARKER_SIZE, label=f"Data: {file_label}", color=data_colors[i], alpha=0.7, zorder=5)

        # --- Curve Fitting (Numerically Stable Version) ---
        # Filter data: saliency and sparsity must be positive for log scale fitting
        mask = (saliency_scores > 0) & (sparsity > 0)
        if np.sum(mask) < 3:
            print("Warning: Not enough valid data points (need at least 3) for curve fitting. Skipping fit.")
            continue

        saliency_fit = saliency_scores[mask]
        sparsity_fit = sparsity[mask]
        
        log_sparsity = np.log(sparsity_fit)

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

            print("--- Curve Fit Results ---")
            print(f"Fitted parameters: c = {c_fit:.4f}, α₀ = {alpha0_fit:.4f}, α₁ = {alpha1_fit:.4f}")

            # Generate and plot the curve using the final model
            threshold_curve = np.logspace(np.log10(saliency_fit.min()), np.log10(saliency_fit.max()), 200)
            sparsity_curve = variable_exponent_model(threshold_curve, c_fit, alpha0_fit, alpha1_fit)
            ax.plot(threshold_curve, sparsity_curve, linestyle='-', linewidth=2.5, label=f"Fit: {file_label}", color=fit_colors[i], alpha=0.8)

        except Exception as e:
            print(f"Curve fitting failed for {filepath}: {e}")

    # --- Final Plotting Configuration ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    xlabel = args.saliency_col.replace('_', ' ').title()
    ylabel = "Sparsity (%)"
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    # MODIFIED: Made title more general
    ax.set_title(f"Sparsity vs. {xlabel} Comparison", fontsize=TITLE_FONT_SIZE, pad=20)
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}%'))
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()

    # MODIFIED: Changed output filename
    plot_filename = "sparsity_vs_saliency_comparison.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    print(f"\nPlot successfully saved to {os.path.abspath(plot_filename)}")
    plt.show()

if __name__ == "__main__":
    main()
