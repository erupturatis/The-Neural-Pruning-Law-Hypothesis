AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 18
LEGEND_FONT_SIZE = 14
MARKER_SIZE = 7

import pandas as pd
import matplotlib.pyplot as plt
import os
from src.infrastructure.others import prefix_path_with_root

def find_accuracy_column(columns):
    accuracy_cols = [col for col in columns
                     if field in col.lower() and 'max' not in col.lower() and 'min' not in col.lower()]

    if not accuracy_cols:
        raise ValueError("No accuracy column found that contains 'accuracy' without 'MAX' or 'MIN'.")
    return accuracy_cols[0]

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
            accuracy_col = find_accuracy_column(df.columns)
        except ValueError as ve:
            print(f"{ve} in file {path}. Skipping.")
            continue
        df_sorted = df.sort_values(by='epoch')

        plt.plot(df_sorted['epoch'], df_sorted[accuracy_col], label=label, linewidth=2)
        print(f"Plotted {accuracy_col} from {path} as '{label}'")

    plt.xlabel('Epoch', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Accuracy (%)', fontsize=AXIS_LABEL_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True)

    # Set tick label sizes
    plt.xticks(fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)

    plt.tight_layout()
    plt.savefig("figure_placeholder.pdf", format="pdf")

    plt.show()

field = "accuracy"


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

csv_paths = paths_lrs_training
custom_labels = labels_lrs_training


if __name__ == "__main__":
    if len(csv_paths) != len(custom_labels):
        print("Error: The number of labels does not match the number of CSV files.")
    else:
        plot_accuracies(csv_paths, custom_labels)
