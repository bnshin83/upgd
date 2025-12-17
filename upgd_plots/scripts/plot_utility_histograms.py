#!/usr/bin/env python3
"""
Plot utility histograms for continual learning experiments.

Usage:
    python plot_utility_histograms.py [dataset]

Datasets: mini_imagenet, input_mnist, emnist, cifar10

Creates scatter plots with 9 bins on x-axis showing utility distribution.
Data source: JSON file extracted from WandB summary files.
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


# Colors for each method
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

# Bins for raw utility (5 bins)
RAW_UTILITY_BIN_LABELS = ['< -0.001', '[-0.001, -0.0002)', '[-0.0002, 0.0002]', '(0.0002, 0.001]', '> 0.001']

# Methods to plot (in order)
METHODS = [
    'S&P', 'UPGD (Full)', 'UPGD (Output Only)', 'UPGD (Hidden Only)',
    'UPGD (Hidden+Output)', 'UPGD (Clamped 0.52)',
    'UPGD (Clamped 0.48-0.52)', 'UPGD (Clamped 0.44-0.56)'
]

LAYERS = ['linear_1', 'linear_2', 'linear_3']


def load_histogram_data():
    """Load histogram data from JSON file."""
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        return None

    with open(DATA_FILE) as f:
        return json.load(f)


def plot_utility_histogram(data, log_scale=False):
    """Plot utility histogram with 9 bins on x-axis."""

    fig, ax = plt.subplots(figsize=(14, 7))

    x_positions = np.arange(len(UTILITY_BIN_LABELS))

    # Plot each method
    for i, method in enumerate(METHODS):
        if method not in data or not data[method]['utility']['global']:
            continue

        values = []
        for bin_label in UTILITY_BIN_LABELS:
            val = data[method]['utility']['global'].get(bin_label, 0)
            # Use small value for log scale when zero
            if log_scale and val == 0:
                values.append(np.nan)
            else:
                values.append(val)

        values = np.array(values)
        valid_mask = ~np.isnan(values)

        marker = MARKERS[i % len(MARKERS)]
        color = COLORS.get(method, f'C{i}')

        ax.scatter(x_positions[valid_mask], values[valid_mask],
                   s=100, label=method, color=color, marker=marker,
                   alpha=0.8, edgecolors='black', linewidth=1.5)

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(UTILITY_BIN_LABELS, rotation=45, ha='right')
    ax.set_xlabel('Utility Range', fontsize=14)

    # Add reference line at 0.52 boundary (between bin 4 and 5)
    ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Set y-axis
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Percentage of Parameters (%, log scale)', fontsize=14)
        ax.set_title(f'Utility Distribution - Log Scale ({DISPLAY_NAME})', fontsize=16)
        ax.text(4.6, ax.get_ylim()[1] * 0.5, '> 0.52', color='red', fontsize=12, va='center')
    else:
        ax.set_ylabel('Percentage of Parameters (%)', fontsize=14)
        ax.set_title(f'Utility Distribution ({DISPLAY_NAME})', fontsize=16)
        ax.text(4.6, ax.get_ylim()[1] * 0.9, '> 0.52', color='red', fontsize=12, va='center')

    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, markerscale=1.2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    suffix = '_log' if log_scale else ''
    filename = f'utility_histogram{suffix}.png'
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / filename}")


def plot_raw_utility_histogram(data, log_scale=False):
    """Plot raw utility histogram with 5 bins on x-axis."""

    fig, ax = plt.subplots(figsize=(12, 7))

    x_positions = np.arange(len(RAW_UTILITY_BIN_LABELS))

    # Plot each method
    for i, method in enumerate(METHODS):
        if method not in data or not data[method]['raw_utility']['global']:
            continue

        values = []
        for bin_label in RAW_UTILITY_BIN_LABELS:
            val = data[method]['raw_utility']['global'].get(bin_label, 0)
            if log_scale and val == 0:
                values.append(np.nan)
            else:
                values.append(val)

        values = np.array(values)
        valid_mask = ~np.isnan(values)

        marker = MARKERS[i % len(MARKERS)]
        color = COLORS.get(method, f'C{i}')

        ax.scatter(x_positions[valid_mask], values[valid_mask],
                   s=100, label=method, color=color, marker=marker,
                   alpha=0.8, edgecolors='black', linewidth=1.5)

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(RAW_UTILITY_BIN_LABELS, rotation=0, ha='center')
    ax.set_xlabel('Raw Utility Range', fontsize=14)

    # Set y-axis
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Percentage of Parameters (%, log scale)', fontsize=14)
        ax.set_title(f'Raw Utility Distribution - Log Scale ({DISPLAY_NAME})', fontsize=16)
    else:
        ax.set_ylabel('Percentage of Parameters (%)', fontsize=14)
        ax.set_title(f'Raw Utility Distribution ({DISPLAY_NAME})', fontsize=16)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, markerscale=1.2, ncol=1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    suffix = '_log' if log_scale else ''
    filename = f'raw_utility_histogram{suffix}.png'
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / filename}")


def plot_layer_utility_histogram(data, method, log_scale=False):
    """Plot per-layer utility histogram for a specific method."""

    if method not in data:
        return

    layers_data = data[method].get('utility', {}).get('layers', {})
    if not any(layers_data.get(layer) for layer in LAYERS):
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    x_positions = np.arange(len(UTILITY_BIN_LABELS))

    for ax, layer in zip(axes, LAYERS):
        layer_data = layers_data.get(layer, {})
        if not layer_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(layer, fontsize=14)
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

        color = COLORS.get(method, '#1f77b4')
        ax.scatter(x_positions[valid_mask], values[valid_mask],
                   s=80, color=color, marker='o',
                   alpha=0.8, edgecolors='black', linewidth=1.5)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(UTILITY_BIN_LABELS, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Utility Range', fontsize=12)
        ax.set_title(layer, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Add reference line at 0.52 boundary
        ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        if log_scale:
            ax.set_yscale('log')

    axes[0].set_ylabel('Percentage of Parameters (%)', fontsize=12)

    # Add "> 0.52" text to the last subplot
    if log_scale:
        axes[2].text(4.6, axes[2].get_ylim()[1] * 0.5, '> 0.52', color='red', fontsize=11, va='center')
    else:
        axes[2].text(4.6, axes[2].get_ylim()[1] * 0.9, '> 0.52', color='red', fontsize=11, va='center')

    suffix = '_log' if log_scale else ''
    fig.suptitle(f'Per-Layer Utility Distribution - {method} ({DISPLAY_NAME})', fontsize=16)

    plt.tight_layout()

    # Create safe filename
    safe_method = method.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
    filename = f'layer_utility_{safe_method}{suffix}.png'
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / filename}")


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = 'mini_imagenet'  # default

    set_dataset(dataset)

    print("=" * 60)
    print(f"Plotting Utility Histograms for {DISPLAY_NAME}")
    print("=" * 60)

    # Load data from JSON (extracted from WandB)
    data = load_histogram_data()
    if data is None:
        print("No histogram data available")
        print(f"Run: python extract_utility_histograms_local.py {DATASET}")
        return

    print(f"Loaded histogram data for {len(data)} methods")

    # Plot utility histograms (9 bins)
    print("\nGenerating utility histogram plots...")
    plot_utility_histogram(data, log_scale=False)
    plot_utility_histogram(data, log_scale=True)

    # Plot raw utility histograms (5 bins)
    print("\nGenerating raw utility histogram plots...")
    plot_raw_utility_histogram(data, log_scale=False)
    plot_raw_utility_histogram(data, log_scale=True)

    # Plot per-layer utility histograms for methods that have layer data
    print("\nGenerating per-layer utility histogram plots...")
    for method in METHODS:
        if method in data:
            layers_data = data[method].get('utility', {}).get('layers', {})
            if any(layers_data.get(layer) for layer in LAYERS):
                plot_layer_utility_histogram(data, method, log_scale=False)
                plot_layer_utility_histogram(data, method, log_scale=True)

    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
