AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 18
LEGEND_FONT_SIZE = 14
MARKER_SIZE = 7  # This will no longer be used but kept for consistency

import pandas as pd
import matplotlib.pyplot as plt
import os
from src.infrastructure.others import prefix_path_with_root

def find_sparsity_column(columns):
    sparsity_cols = [col for col in columns
                     if field in col.lower()
                     and 'max' not in col.lower()
                     and 'min' not in col.lower()]
    if not sparsity_cols:
        raise ValueError(f"No sparsity column found containing '{field}' without 'MAX' or 'MIN'.")
    return sparsity_cols[0]

def plot_accuracies(csv_paths, labels):
    plt.figure(figsize=(12, 8))

    for name, label in zip(csv_paths, labels):
        path = os.path.join(name + '.csv')
        path = prefix_path_with_root(path)
        print(path)

        if not os.path.isfile(path):
            print(f"File not found: {path}. Skipping.")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}. Skipping.")
            continue

        if 'epoch' not in df.columns:
            print(f"'epoch' column not found in {path}. Skipping.")
            continue

        try:
            accuracy_col = find_sparsity_column(df.columns)
        except ValueError as ve:
            print(f"{ve} in file {path}. Skipping.")
            continue
        df_sorted = df.sort_values(by='epoch')

        plt.plot(df_sorted['epoch'], df_sorted[accuracy_col], label=label, linewidth=2)
        print(f"Plotted {accuracy_col} from {path} as '{label}'")

    plt.yscale('log')
    plt.yticks([1, 10, 100], ['1', '10', '100'], fontsize=TICK_LABEL_SIZE)
    plt.xlabel('Epoch', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Sparsity (%)', fontsize=AXIS_LABEL_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    plt.grid(True)

    # Set tick label sizes
    plt.xticks(fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)

    plt.tight_layout()
    plt.savefig("figure_placeholder.pdf", format="pdf")

    plt.show()

field = "sparsity"

labels_decay = [
    'With decay, 1.35% sparsity',
    'Without decay, 1.43% sparsity',
]
paths_decay = [
    f'src/plots/high_to_low_{field}',
    f'src/plots/high_to_low_no_decay_{field}',
]

labels_lrs_flow = [
    'High constant LR',
    'Low constant LR',
    'High to low LR',
]
paths_lrs_flow = [
    f'src/plots/high_constant_{field}',
    f'src/plots/low_constant_{field}',
    f'src/plots/high_to_low_{field}',
]


labels_lrs_training = [
    'initial to low LR',
    'high to low LR',
    'low to low LR',
]

paths_lrs_training = [
    f'src/plots/initial_to_low_{field}',
    f'src/plots/high_to_low_{field}',
    f'src/plots/low_constant_{field}',
]


labels_flow_lrs_regrowth = [
    '10 times flow LR',
    '20 times flow LR',
    '50 times flow LR',
]
paths_flow_lrs_regrowth = [
    f'src/plots/flow10_{field}',
    f'src/plots/flow20_{field}',
    f'src/plots/flow50_{field}',
]

csv_paths = paths_flow_lrs_regrowth
custom_labels =  labels_flow_lrs_regrowth


if __name__ == "__main__":
    if len(csv_paths) != len(custom_labels):
        print("Error: The number of labels does not match the number of CSV files.")
    else:
        plot_accuracies(csv_paths, custom_labels)
