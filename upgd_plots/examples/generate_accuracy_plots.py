#!/usr/bin/env python3
"""
Generate accuracy comparison plots for all datasets.
Compares all 8 methods: S&P baseline + 7 UPGD variants.
"""

import sys
sys.path.insert(0, '/scratch/gautschi/shin283/upgd/upgd_plots')

from agents.data.loader import DataLoaderAgent
from agents.data.aggregator import AggregatorAgent
from agents.plot.time_series import TimeSeriesPlotAgent
from config.default_config import DATASET_CONFIGS, default_config
import pandas as pd

def main():
    # Override config to only generate PNG
    plot_config = default_config.copy()
    plot_config['plotting']['formats'] = ['png']

    # Initialize agents
    loader = DataLoaderAgent(name="DataLoader", config=default_config)
    aggregator = AggregatorAgent(name="Aggregator", config=default_config)
    plotter = TimeSeriesPlotAgent(name="TimeSeriesPlotter", config=plot_config)

    datasets = ['mini_imagenet', 'input_mnist', 'emnist', 'cifar10']

    # All 8 methods to compare
    all_methods = [
        'S&P',
        'UPGD (Full)',
        'UPGD (Output Only)',
        'UPGD (Hidden Only)',
        'UPGD (Hidden+Output)',
        'UPGD (Clamped 0.52)',
        'UPGD (Clamped 0.48-0.52)',
        'UPGD (Clamped 0.44-0.56)'
    ]

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset}")
        print(f"{'='*60}")

        # Load accuracy data for all methods
        print(f"Loading accuracy data...")
        data = loader.execute(
            dataset=dataset,
            methods=all_methods,
            metrics=['accuracy']
        )

        print(f"Loaded {len(data)} rows")

        # Aggregate per task
        print(f"Aggregating per task...")
        aggregated = aggregator.aggregate_per_task(
            data=data,
            dataset=dataset
        )

        # Rename value_mean to value for TimeSeriesPlotAgent
        if 'value_mean' in aggregated.columns and 'value' not in aggregated.columns:
            aggregated['value'] = aggregated['value_mean']
            if 'value_std' in aggregated.columns:
                aggregated['value_ci_lower'] = aggregated['value_mean'] - 1.96 * aggregated['value_sem']
                aggregated['value_ci_upper'] = aggregated['value_mean'] + 1.96 * aggregated['value_sem']

        # Generate plot
        print(f"Generating plot...")
        dataset_name = DATASET_CONFIGS[dataset]['display_name']

        fig, path = plotter.execute(
            data=aggregated,
            methods=all_methods,
            metric='accuracy',
            x_axis='task',
            confidence_level=0.95,
            show_bands=True,
            title=f'Accuracy Comparison - {dataset_name}',
            xlabel='Task',
            ylabel='Accuracy',
            output_filename=f'{dataset}/accuracy_comparison'
        )

        print(f"✓ Plot saved to {path}")

    print(f"\n{'='*60}")
    print("All accuracy plots generated successfully!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
