#!/usr/bin/env python3
"""
Plot per-layer utility distribution comparison across methods.

This script creates histogram visualizations comparing the scaled utility 
distribution for each layer (hidden layers vs output layer) across different 
UPGD methods.

Usage:
    python plot_layer_utility_distribution.py [dataset]

Datasets: mini_imagenet, input_mnist, emnist, cifar10
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 12})

# Dataset display names
DATASET_DISPLAY_NAMES = {
    'mini_imagenet': 'Mini-ImageNet',
    'input_mnist': 'Input-Permuted MNIST',
    'emnist': 'Label-Permuted EMNIST',
    'cifar10': 'Label-Permuted CIFAR-10',
}

# Global variables set by set_dataset()
DATASET = None
DATA_FILE = None
PLOT_DIR = None
DISPLAY_NAME = None

# Paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent  # upgd_plots/

def set_dataset(dataset):
    """Set global dataset variables."""
    global DATASET, DATA_FILE, PLOT_DIR, DISPLAY_NAME
    DATASET = dataset
    DATA_FILE = PLOTS_DIR / 'data' / 'utility_histograms' / f'{DATASET}_utility_histograms.json'
    PLOT_DIR = PLOTS_DIR / 'figures' / DATASET
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DISPLAY_NAME = DATASET_DISPLAY_NAMES.get(dataset, dataset.replace('_', ' ').title())

# Colors for each method (consistent with other plots)
COLORS = {
    'S&P': '#7f7f7f',
    'UPGD (Full)': '#1f77b4',
    'UPGD (Output Only)': '#2ca02c',
    'UPGD (Hidden Only)': '#ff7f0e',
    'UPGD (Hidden+Output)': '#9467bd',
    'UPGD (Clamped 0.52)': '#d62728',
    'UPGD (Clamped 0.48-0.52)': '#8c564b',
    'UPGD (Clamped 0.44-0.56)': '#e377c2',
}

MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

# Bins for scaled utility (9 bins)
UTILITY_BIN_LABELS = [
    '[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.44)', '[0.44, 0.48)', '[0.48, 0.52)',
    '[0.52, 0.56)', '[0.56, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]'
]

# Short bin labels for cleaner plots
UTILITY_BIN_SHORT = [
    '0-0.2', '0.2-0.4', '0.4-0.44', '0.44-0.48', '0.48-0.52',
    '0.52-0.56', '0.56-0.6', '0.6-0.8', '0.8-1.0'
]

# Layer labels with descriptive names
LAYERS = ['linear_1', 'linear_2', 'linear_3']
LAYER_DISPLAY_NAMES = {
    'linear_1': 'Hidden Layer 1 (Input→Hidden)',
    'linear_2': 'Hidden Layer 2 (Hidden→Hidden)',
    'linear_3': 'Output Layer (Hidden→Output)',
}

# Methods that typically have per-layer data
METHODS_WITH_LAYERS = [
    'UPGD (Output Only)',
    'UPGD (Hidden Only)',
    'UPGD (Hidden+Output)',
    'UPGD (Clamped 0.52)',
]


def load_histogram_data():
    """Load histogram data from JSON file."""
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        return None
    
    with open(DATA_FILE) as f:
        return json.load(f)


def get_methods_with_layer_data(data):
    """Find all methods that have per-layer utility data."""
    methods_with_data = []
    for method, method_data in data.items():
        layers = method_data.get('utility', {}).get('layers', {})
        has_data = any(
            layers.get(layer) and len(layers.get(layer)) > 0 
            for layer in LAYERS
        )
        if has_data:
            methods_with_data.append(method)
    return methods_with_data


def plot_layer_comparison_single_layer(data, layer, methods, log_scale=False):
    """
    Plot utility distribution for a single layer, comparing all methods.
    
    Args:
        data: Full histogram data dict
        layer: Layer name (linear_1, linear_2, linear_3)
        methods: List of method names to include
        log_scale: Whether to use log scale on y-axis
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_positions = np.arange(len(UTILITY_BIN_LABELS))
    bar_width = 0.8 / len(methods)  # Width of each bar group
    
    for i, method in enumerate(methods):
        if method not in data:
            continue
        
        layer_data = data[method].get('utility', {}).get('layers', {}).get(layer, {})
        if not layer_data:
            continue
        
        values = []
        for bin_label in UTILITY_BIN_LABELS:
            val = layer_data.get(bin_label, 0)
            if log_scale and val == 0:
                values.append(np.nan)
            else:
                values.append(val)
        
        values = np.array(values)
        valid_mask = ~np.isnan(values)
        
        # Offset each method's bars/markers
        offset = (i - len(methods) / 2 + 0.5) * bar_width
        
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS.get(method, f'C{i}')
        
        # Use scatter plot with offset
        ax.scatter(x_positions[valid_mask] + offset, values[valid_mask],
                   s=100, label=method, color=color, marker=marker,
                   alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(UTILITY_BIN_SHORT, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel('Utility Range', fontsize=14)
    
    # Add reference line at 0.52 boundary (between bin 4 and 5)
    ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Set y-axis
    layer_name = LAYER_DISPLAY_NAMES.get(layer, layer)
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Percentage of Parameters (%, log scale)', fontsize=14)
        ax.set_title(f'{layer_name}\nUtility Distribution (Log Scale) - {DISPLAY_NAME}', fontsize=14)
        ax.text(4.6, ax.get_ylim()[1] * 0.5, '> 0.52', color='red', fontsize=11, va='center')
    else:
        ax.set_ylabel('Percentage of Parameters (%)', fontsize=14)
        ax.set_title(f'{layer_name}\nUtility Distribution - {DISPLAY_NAME}', fontsize=14)
        ax.text(4.6, ax.get_ylim()[1] * 0.9, '> 0.52', color='red', fontsize=11, va='center')
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, markerscale=1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    suffix = '_log' if log_scale else ''
    filename = f'layer_utility_distribution_{layer}{suffix}.png'
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / filename}")


def plot_all_layers_grid(data, methods, log_scale=False):
    """
    Create a 3-panel grid showing utility distribution for all layers.
    
    Args:
        data: Full histogram data dict
        methods: List of method names to include
        log_scale: Whether to use log scale on y-axis
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    
    x_positions = np.arange(len(UTILITY_BIN_LABELS))
    
    for ax_idx, (ax, layer) in enumerate(zip(axes, LAYERS)):
        for i, method in enumerate(methods):
            if method not in data:
                continue
            
            layer_data = data[method].get('utility', {}).get('layers', {}).get(layer, {})
            if not layer_data:
                continue
            
            values = []
            for bin_label in UTILITY_BIN_LABELS:
                val = layer_data.get(bin_label, 0)
                if log_scale and val == 0:
                    values.append(np.nan)
                else:
                    values.append(val)
            
            values = np.array(values)
            valid_mask = ~np.isnan(values)
            
            marker = MARKERS[i % len(MARKERS)]
            color = COLORS.get(method, f'C{i}')
            
            ax.scatter(x_positions[valid_mask], values[valid_mask],
                       s=80, label=method if ax_idx == 0 else "", color=color, marker=marker,
                       alpha=0.8, edgecolors='black', linewidth=1)
        
        # Set x-axis for this panel
        ax.set_xticks(x_positions)
        ax.set_xticklabels(UTILITY_BIN_SHORT, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Utility Range', fontsize=12)
        
        # Layer title
        layer_name = LAYER_DISPLAY_NAMES.get(layer, layer)
        ax.set_title(layer_name, fontsize=12, fontweight='bold')
        
        # Add reference line at 0.52 boundary
        ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    # Y-axis label on first panel only
    if log_scale:
        axes[0].set_ylabel('Percentage of Parameters (%, log scale)', fontsize=12)
    else:
        axes[0].set_ylabel('Percentage of Parameters (%)', fontsize=12)
    
    # Add > 0.52 label to last panel
    if log_scale:
        axes[2].text(4.6, axes[2].get_ylim()[1] * 0.5, '> 0.52', color='red', fontsize=11, va='center')
    else:
        axes[2].text(4.6, axes[2].get_ylim()[1] * 0.9, '> 0.52', color='red', fontsize=11, va='center')
    
    # Add legend to first panel
    axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9, markerscale=1.0)
    
    # Main title
    suffix_text = ' (Log Scale)' if log_scale else ''
    fig.suptitle(f'Per-Layer Utility Distribution{suffix_text} - {DISPLAY_NAME}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    suffix = '_log' if log_scale else ''
    filename = f'layer_utility_distribution_grid{suffix}.png'
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / filename}")


def plot_hidden_vs_output_comparison(data, methods, log_scale=False):
    """
    Create a 2-panel comparison: hidden layers (combined) vs output layer.
    
    Args:
        data: Full histogram data dict
        methods: List of method names to include
        log_scale: Whether to use log scale on y-axis
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    x_positions = np.arange(len(UTILITY_BIN_LABELS))
    
    # Panel 1: Hidden Layers (average of linear_1 and linear_2)
    ax = axes[0]
    ax.set_title('Hidden Layers (linear_1 + linear_2 averaged)', fontsize=14, fontweight='bold')
    
    for i, method in enumerate(methods):
        if method not in data:
            continue
        
        layers_data = data[method].get('utility', {}).get('layers', {})
        
        # Average linear_1 and linear_2
        hidden_values = []
        for bin_label in UTILITY_BIN_LABELS:
            l1_val = layers_data.get('linear_1', {}).get(bin_label, 0)
            l2_val = layers_data.get('linear_2', {}).get(bin_label, 0)
            
            # Only average if both have data, otherwise use what's available
            if l1_val or l2_val:
                avg = (l1_val + l2_val) / 2 if (l1_val and l2_val) else (l1_val or l2_val)
            else:
                avg = 0
            
            if log_scale and avg == 0:
                hidden_values.append(np.nan)
            else:
                hidden_values.append(avg)
        
        hidden_values = np.array(hidden_values)
        valid_mask = ~np.isnan(hidden_values)
        
        if not any(valid_mask):
            continue
        
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS.get(method, f'C{i}')
        
        ax.scatter(x_positions[valid_mask], hidden_values[valid_mask],
                   s=100, label=method, color=color, marker=marker,
                   alpha=0.8, edgecolors='black', linewidth=1.5)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(UTILITY_BIN_SHORT, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Utility Range', fontsize=12)
    ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, markerscale=1.2)
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Percentage of Parameters (%, log scale)', fontsize=12)
    else:
        ax.set_ylabel('Percentage of Parameters (%)', fontsize=12)
    
    # Panel 2: Output Layer (linear_3)
    ax = axes[1]
    ax.set_title('Output Layer (linear_3)', fontsize=14, fontweight='bold')
    
    for i, method in enumerate(methods):
        if method not in data:
            continue
        
        output_data = data[method].get('utility', {}).get('layers', {}).get('linear_3', {})
        if not output_data:
            continue
        
        values = []
        for bin_label in UTILITY_BIN_LABELS:
            val = output_data.get(bin_label, 0)
            if log_scale and val == 0:
                values.append(np.nan)
            else:
                values.append(val)
        
        values = np.array(values)
        valid_mask = ~np.isnan(values)
        
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS.get(method, f'C{i}')
        
        ax.scatter(x_positions[valid_mask], values[valid_mask],
                   s=100, color=color, marker=marker,
                   alpha=0.8, edgecolors='black', linewidth=1.5)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(UTILITY_BIN_SHORT, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Utility Range', fontsize=12)
    ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    
    if log_scale:
        ax.set_yscale('log')
        ax.text(4.6, ax.get_ylim()[1] * 0.5, '> 0.52', color='red', fontsize=11, va='center')
    else:
        ax.text(4.6, ax.get_ylim()[1] * 0.9, '> 0.52', color='red', fontsize=11, va='center')
    
    # Main title
    suffix_text = ' (Log Scale)' if log_scale else ''
    fig.suptitle(f'Hidden vs Output Layer Utility Distribution{suffix_text} - {DISPLAY_NAME}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    suffix = '_log' if log_scale else ''
    filename = f'hidden_vs_output_utility{suffix}.png'
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / filename}")


def print_layer_statistics(data, methods):
    """Print summary statistics for each layer and method."""
    print("\n" + "=" * 80)
    print("Per-Layer Utility Statistics Summary")
    print("=" * 80)
    
    for layer in LAYERS:
        print(f"\n{LAYER_DISPLAY_NAMES[layer]}")
        print("-" * 60)
        
        for method in methods:
            if method not in data:
                continue
            
            layer_data = data[method].get('utility', {}).get('layers', {}).get(layer, {})
            if not layer_data:
                continue
            
            # Find key statistics
            bin_48_52 = layer_data.get('[0.48, 0.52)', 0)
            above_52 = sum(layer_data.get(b, 0) for b in ['[0.52, 0.56)', '[0.56, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]'])
            below_48 = sum(layer_data.get(b, 0) for b in ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.44)', '[0.44, 0.48)'])
            
            print(f"  {method:30s} | [0.48-0.52): {bin_48_52:6.2f}% | >0.52: {above_52:6.3f}% | <0.48: {below_48:6.3f}%")


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = 'mini_imagenet'  # default
    
    set_dataset(dataset)
    
    print("=" * 60)
    print(f"Plotting Per-Layer Utility Distribution for {DISPLAY_NAME}")
    print("=" * 60)
    
    # Load data from JSON
    data = load_histogram_data()
    if data is None:
        print("No histogram data available")
        print(f"Run: python extract_utility_histograms_local.py {DATASET}")
        return
    
    # Find methods with per-layer data
    methods_with_data = get_methods_with_layer_data(data)
    print(f"\nMethods with per-layer data: {methods_with_data}")
    
    if not methods_with_data:
        print("No methods have per-layer utility data available.")
        print("Please ensure experiments logged per-layer utility histograms to WandB.")
        return
    
    # Print statistics
    print_layer_statistics(data, methods_with_data)
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    # 1. Grid plot (all 3 layers)
    print("\n1. All layers grid plot...")
    plot_all_layers_grid(data, methods_with_data, log_scale=False)
    plot_all_layers_grid(data, methods_with_data, log_scale=True)
    
    # 2. Hidden vs Output comparison
    print("\n2. Hidden vs Output comparison...")
    plot_hidden_vs_output_comparison(data, methods_with_data, log_scale=False)
    plot_hidden_vs_output_comparison(data, methods_with_data, log_scale=True)
    
    # 3. Individual layer plots
    print("\n3. Individual layer plots...")
    for layer in LAYERS:
        plot_layer_comparison_single_layer(data, layer, methods_with_data, log_scale=False)
        plot_layer_comparison_single_layer(data, layer, methods_with_data, log_scale=True)
    
    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
