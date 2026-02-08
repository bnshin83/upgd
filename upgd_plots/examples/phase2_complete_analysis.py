#!/usr/bin/env python3
"""
Phase 2 Example: Complete Statistical Analysis and Comparison

This script demonstrates the full workflow with Phase 2 agents:
1. Load data with DataLoaderAgent
2. Aggregate per-task and across seeds with AggregatorAgent
3. Perform statistical tests with StatisticalTestAgent
4. Generate comparison tables with ComparatorAgent
5. Create visualizations with TimeSeriesPlotAgent and ComparisonBarPlotAgent
"""

import sys
from pathlib import Path

# Add upgd_plots to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data import DataLoaderAgent, AggregatorAgent
from agents.stats import StatisticalTestAgent, ComparatorAgent
from agents.plot import TimeSeriesPlotAgent
# from agents.plot import ComparisonBarPlotAgent  # Uncomment when ready to use
from config import default_config

def main():
    print("=" * 70)
    print("UPGD Phase 2: Complete Statistical Analysis Workflow")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("Step 1: Loading experiment data...")
    print("-" * 70)

    loader = DataLoaderAgent(name='loader', config=default_config)

    # Load data for multiple methods and seeds
    data = loader.execute(
        dataset='mini_imagenet',
        methods=['UPGD (Full)', 'UPGD (Output Only)', 'UPGD (Hidden Only)'],
        seeds=[2],  # Add more seeds when available: [1, 2, 3]
        metrics=['accuracy', 'plasticity']
    )

    print(f"✓ Loaded {len(data)} rows of data")
    print(f"  Methods: {data['method'].unique()}")
    print(f"  Seeds: {data['seed'].unique()}")
    print(f"  Metrics: {data['metric'].unique()}")
    print()

    # ========================================================================
    # 2. Aggregate Data
    # ========================================================================
    print("Step 2: Aggregating data...")
    print("-" * 70)

    aggregator = AggregatorAgent(name='aggregator', config=default_config)

    # Convert per-step (1M points) to per-task (~400 points)
    task_data = aggregator.aggregate_per_task(
        data=data,
        dataset='mini_imagenet',
        steps_per_task=2500
    )

    print(f"✓ Aggregated from {len(data)} steps to {len(task_data)} task windows")
    print(f"  Steps per task: 2500")
    print(f"  Tasks: {task_data['task'].nunique()}")
    print()

    # If multiple seeds available, aggregate across seeds
    if len(data['seed'].unique()) > 1:
        seed_aggregated = aggregator.aggregate_across_seeds(
            data=task_data,
            groupby=['method', 'task', 'metric']
        )
        print(f"✓ Aggregated across {seed_aggregated['n_seeds'].iloc[0]} seeds")
        analysis_data = seed_aggregated
    else:
        print("  Note: Only 1 seed available, skipping cross-seed aggregation")
        analysis_data = task_data

    print()

    # ========================================================================
    # 3. Statistical Comparison
    # ========================================================================
    print("Step 3: Statistical testing...")
    print("-" * 70)

    comparator = ComparatorAgent(name='comparator', config=default_config)

    # Generate comprehensive comparison
    comparison = comparator.execute(
        data=analysis_data,
        methods=['UPGD (Full)', 'UPGD (Output Only)', 'UPGD (Hidden Only)'],
        metrics=['accuracy', 'plasticity'],
        aggregation='final'
    )

    print("✓ Method Comparison Summary:")
    print(comparison['summary'][['method', 'metric', 'mean', 'std']].to_string(index=False))
    print()

    print("✓ Best Methods:")
    print(comparison['best_method'].to_string(index=False))
    print()

    print("✓ Win Matrix (wins on metrics):")
    print(comparison['win_matrix'])
    print()

    # Detailed two-method comparison
    if len(analysis_data['method'].unique()) >= 2:
        detailed = comparator.compare_two_methods(
            data=analysis_data,
            method_a='UPGD (Full)',
            method_b='UPGD (Output Only)',
            metrics=['accuracy', 'plasticity']
        )
        print("✓ Detailed Comparison: UPGD (Full) vs UPGD (Output Only)")
        print(detailed.to_string(index=False))
        print()

    # ========================================================================
    # 4. Plot Time Series
    # ========================================================================
    print("Step 4: Creating visualizations...")
    print("-" * 70)

    plotter = TimeSeriesPlotAgent(config=default_config)

    # Plot accuracy over tasks
    for metric in ['accuracy', 'plasticity']:
        # Use task_data for plotting (not aggregated across seeds yet)
        metric_data = task_data[task_data['metric'] == metric].copy()

        # Rename value_mean to value for TimeSeriesPlotAgent
        if 'value_mean' in metric_data.columns and 'value' not in metric_data.columns:
            metric_data['value'] = metric_data['value_mean']

        fig, path = plotter.execute(
            data=metric_data,
            methods=['UPGD (Full)', 'UPGD (Output Only)', 'UPGD (Hidden Only)'],
            metric=metric,
            x_axis='task',
            confidence_level=0.95,
            show_bands=False,  # Only 1 seed, no bands
            title=f'{metric.title()} Over Tasks (Mini-ImageNet)',
            subdir='mini_imagenet',
            filename=f'phase2_{metric}_comparison'
        )
        print(f"✓ Saved {metric} plot to: {path}")

    print()

    # ========================================================================
    # 5. Statistical Tests (if multiple seeds)
    # ========================================================================
    if len(data['seed'].unique()) > 1:
        print("Step 5: Hypothesis testing...")
        print("-" * 70)

        tester = StatisticalTestAgent(name='tester', config=default_config)

        test_results = tester.execute(
            data=analysis_data,
            baseline_method='UPGD (Full)',
            comparison_methods=['UPGD (Output Only)', 'UPGD (Hidden Only)'],
            metric='accuracy',
            test_type='wilcoxon',
            correction='holm'
        )

        print("✓ Statistical Test Results:")
        print(test_results[['method_b', 'p_value', 'p_value_corrected', 'significant', 'effect_size']].to_string(index=False))
        print()

        # Count significant differences
        n_sig = test_results['significant'].sum()
        print(f"  {n_sig}/{len(test_results)} comparisons are significant at α=0.05")
        print()
    else:
        print("Step 5: Skipped (requires multiple seeds for meaningful statistical tests)")
        print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Phase 2 Analysis Complete!")
    print("=" * 70)
    print()
    print("Generated outputs:")
    print("  - Summary statistics table")
    print("  - Win/loss matrix")
    print("  - Best method identification")
    print("  - Time series plots (accuracy and plasticity)")
    if len(data['seed'].unique()) > 1:
        print("  - Statistical significance tests")
    print()
    print("Next steps:")
    print("  1. Run with multiple seeds for robust statistical testing")
    print("  2. Try other datasets: 'input_mnist', 'emnist', 'cifar10'")
    print("  3. Add more methods to comparison")
    print("  4. Export results to LaTeX tables (Phase 3)")
    print()

if __name__ == '__main__':
    main()
