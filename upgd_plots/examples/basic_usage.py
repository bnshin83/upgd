#!/usr/bin/env python3
"""
Basic example demonstrating UPGD agent usage.

This script shows how to:
1. Load experiment data with DataLoaderAgent
2. Perform statistical tests with StatisticalTestAgent
3. Create time series plots with TimeSeriesPlotAgent
"""

import sys
from pathlib import Path

# Add upgd_plots to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data import DataLoaderAgent
from agents.stats import StatisticalTestAgent
from agents.plot import TimeSeriesPlotAgent
from config import default_config

def main():
    print("=" * 70)
    print("UPGD Agent-Based Analysis Example")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("Step 1: Loading experiment data...")
    print("-" * 70)

    loader = DataLoaderAgent(name='loader', config=default_config)

    # Load data for Mini-ImageNet
    # Try with a small subset first: just seed 2, one metric
    data = loader.execute(
        dataset='mini_imagenet',
        methods=['UPGD (Full)', 'UPGD (Output Only)'],
        seeds=[2],
        metrics=['accuracy']
    )

    print(f"Loaded {len(data)} rows of data")
    print(f"Datasets: {data['dataset'].unique()}")
    print(f"Methods: {data['method'].unique()}")
    print(f"Seeds: {data['seed'].unique()}")
    print(f"Metrics: {data['metric'].unique()}")
    print(f"Steps: {data['step'].min()} to {data['step'].max()}")
    print()

    # ========================================================================
    # 2. Statistical Tests
    # ========================================================================
    print("Step 2: Performing statistical tests...")
    print("-" * 70)

    tester = StatisticalTestAgent(name='tester', config=default_config)

    # Compare UPGD variants
    # Note: For a single seed, statistical test may not be meaningful
    # This is just to demonstrate the API
    print("Note: Statistical tests require multiple seeds for meaningful results.")
    print("This example uses only seed 2 for demonstration purposes.")
    print()

    # ========================================================================
    # 3. Plot Time Series
    # ========================================================================
    print("Step 3: Creating time series plots...")
    print("-" * 70)

    plotter = TimeSeriesPlotAgent(config=default_config)

    # Subsample for faster plotting (plot every 1000th point)
    fig, path = plotter.execute(
        data=data,
        methods=['UPGD (Full)', 'UPGD (Output Only)'],
        metric='accuracy',
        x_axis='step',
        confidence_level=0.95,
        show_bands=True,
        subsample=1000,  # Plot every 1000th step
        title='Accuracy Over Time (Mini-ImageNet)',
        subdir='mini_imagenet',
        filename='example_accuracy_comparison'
    )

    print(f"Plot saved to: {path}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Load data for multiple seeds to enable statistical testing")
    print("2. Try different datasets: 'input_mnist', 'emnist', 'cifar10'")
    print("3. Compare more methods and metrics")
    print("4. Explore other agents in Phase 2 (AggregatorAgent, etc.)")
    print()

if __name__ == '__main__':
    main()
