#!/usr/bin/env python3
"""
Generate utility histogram plots for all 4 datasets.

Creates plots comparing UPGD (Full) vs UPGD (Output Only) for:
- Mini-ImageNet
- Input-Permuted MNIST
- EMNIST
- CIFAR-10

Both linear and log scale versions.
"""

import sys
from pathlib import Path

# Add upgd_plots to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data import UtilityHistogramLoaderAgent
from agents.plot import UtilityHistogramPlotAgent
from config import default_config

# Override config to only generate PNG (not PDF)
plot_config = default_config.copy()
plot_config['plotting']['formats'] = ['png']  # Only PNG

# Initialize agents
loader = UtilityHistogramLoaderAgent(config=plot_config)
plotter = UtilityHistogramPlotAgent(config=plot_config)

# Datasets to process
datasets = ['mini_imagenet', 'input_mnist', 'emnist', 'cifar10']

print("=" * 70)
print("Generating Utility Histogram Plots for All Datasets")
print("=" * 70)
print()

for dataset in datasets:
    print(f"Processing {dataset}...")
    print("-" * 70)

    # Load data
    try:
        hist_data = loader.execute(
            dataset=dataset,
            methods=['UPGD (Full)', 'UPGD (Output Only)']
        )
    except FileNotFoundError as e:
        print(f"  ⚠ Skipping {dataset}: {e}")
        print()
        continue

    # Linear scale plot
    fig1, path1 = plotter.execute(
        histogram_data=hist_data,
        methods=['UPGD (Full)', 'UPGD (Output Only)'],
        plot_type='scatter',
        log_scale=False,
        per_layer=False,
        title=f'Utility Distribution - {dataset.replace("_", " ").title()}',
        subdir=dataset,
        filename='utility_histogram_comparison'
    )
    print(f"  ✓ Linear: {path1}")

    # Log scale plot
    fig2, path2 = plotter.execute(
        histogram_data=hist_data,
        methods=['UPGD (Full)', 'UPGD (Output Only)'],
        plot_type='scatter',
        log_scale=True,
        per_layer=False,
        title=f'Utility Distribution - {dataset.replace("_", " ").title()} [Log]',
        subdir=dataset,
        filename='utility_histogram_comparison_log'
    )
    print(f"  ✓ Log:    {path2}")
    print()

print("=" * 70)
print("All plots generated!")
print("=" * 70)
print()
print("Plots saved to:")
for dataset in datasets:
    print(f"  - figures/{dataset}/utility_histogram_comparison.png")
    print(f"  - figures/{dataset}/utility_histogram_comparison_log.png")
