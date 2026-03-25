import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def sparsity_per_layer(state_dict_path):
    stateDict = torch.load(state_dict_path)
    layer_total = []
    layer_remaining = []
    layer_names = []
    total = 0
    remaining = 0
    
    for key in stateDict.keys():
      
        if ("weight" not in key or "bias" in key or "bn" in key or 
            key in [
                "layer1.0.downsample.1.weight",
                "layer2.0.downsample.1.weight",
                "layer3.0.downsample.1.weight",
                "layer4.0.downsample.1.weight"
            ]  or "running_mean" in key or "running_var" in key or "num_batches_tracked" in key):
            continue
        print(key)
        this_layer_total = stateDict[key].numel()
        this_layer_remaining = int((stateDict[key] != 0).float().sum().item())
        layer_names.append(key)

        layer_remaining.append(this_layer_remaining)
        layer_total.append(this_layer_total) 

        total += this_layer_total
        remaining += this_layer_remaining

    print("Total parameters: ", total)
    print("Remaining parameters: ", remaining)
    print("Sparsity: {:.2f}%".format((total - remaining) / total * 100))

    return layer_total, layer_remaining, layer_names

def plot_histogram_sparsity(sparsity_percentage, layers):
    x = np.arange(len(layers))

    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 8,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    plt.figure(figsize=(12, 8))

    bars = plt.bar(x, sparsity_percentage, color='skyblue', label='Sparsity (%)')

    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Sparsity (%)", fontsize=12)
    plt.title("Logarithmic Sparsity per Layer", fontsize=14)

    plt.xticks(x, layers, rotation='vertical', ha='center', fontsize=8)

    plt.yscale("log")

    yticks = [0.01, 0.1, 1, 10, 100]
    plt.yticks(yticks)

    def percent_formatter(y, pos):
        return f'{y}%'

    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(percent_formatter))

    plt.ylim(min(yticks), max(yticks))

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, sparsity in zip(bars, sparsity_percentage):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{sparsity:.2f}%', 
                 ha='center', va='bottom', fontsize=6, rotation=90)

    plt.tight_layout()

    plt.legend()
    plt.show()

