#!/usr/bin/env python3
"""
Utility Histogram Example

Demonstrates how to load and visualize utility histogram data showing
parameter importance distributions across UPGD methods.
"""

import sys
from pathlib import Path

# Add upgd_plots to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data import UtilityHistogramLoaderAgent
from agents.plot import UtilityHistogramPlotAgent
from config import default_config

def main():
    print("=" * 70)
    print("UPGD Utility Histogram Visualization")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. Load Utility Histogram Data
    # ========================================================================
    print("Step 1: Loading utility histogram data...")
    print("-" * 70)

    loader = UtilityHistogramLoaderAgent(config=default_config)

    # Load data for mini_imagenet
    hist_data = loader.execute(
        dataset='mini_imagenet',
        methods=['UPGD (Full)', 'UPGD (Output Only)', 'UPGD (Hidden Only)']
    )

    print(f"✓ Loaded histogram data for {len(hist_data)} methods")

    # Show available methods
    available = loader.get_available_methods('mini_imagenet')
    print(f"  Available methods: {len(available)}")

    # Show bin structure
    bins = loader.get_utility_bins()
    print(f"  Scaled utility bins: {len(bins['scaled'])}")
    print(f"  Raw utility bins: {len(bins['raw'])}")
    print()

    # ========================================================================
    # 2. Plot Global Utility Distributions
    # ========================================================================
    print("Step 2: Creating global utility distribution plots...")
    print("-" * 70)

    plotter = UtilityHistogramPlotAgent(config=default_config)

    # Scatter plot (linear scale)
    fig1, path1 = plotter.execute(
        histogram_data=hist_data,
        methods=['UPGD (Full)', 'UPGD (Output Only)', 'UPGD (Hidden Only)'],
        plot_type='scatter',
        log_scale=False,
        per_layer=False,
        title='Global Utility Distribution (Mini-ImageNet)',
        subdir='mini_imagenet',
        filename='utility_histogram_global'
    )
    print(f"✓ Saved global scatter plot (linear): {path1}")

    # Scatter plot (log scale)
    fig2, path2 = plotter.execute(
        histogram_data=hist_data,
        methods=['UPGD (Full)', 'UPGD (Output Only)', 'UPGD (Hidden Only)'],
        plot_type='scatter',
        log_scale=True,
        per_layer=False,
        title='Global Utility Distribution (Mini-ImageNet) [Log Scale]',
        subdir='mini_imagenet',
        filename='utility_histogram_global_log'
    )
    print(f"✓ Saved global scatter plot (log): {path2}")

    print()

    # ========================================================================
    # 3. Plot Per-Layer Comparison for Each Method
    # ========================================================================
    print("Step 3: Creating per-layer comparison plots...")
    print("-" * 70)

    for method in ['UPGD (Full)', 'UPGD (Output Only)']:
        fig, path = plotter.plot_per_layer_comparison(
            histogram_data=hist_data,
            method=method,
            log_scale=False,
            subdir='mini_imagenet'
        )
        print(f"✓ Saved per-layer plot for {method}: {path}")

    print()

    # ========================================================================
    # 4. Analyze Distribution
    # ========================================================================
    print("Step 4: Analyzing utility distributions...")
    print("-" * 70)

    for method, data in hist_data.items():
        if 'utility' in data and 'global' in data['utility']:
            global_hist = data['utility']['global']

            # Find bins with most parameters
            sorted_bins = sorted(global_hist.items(), key=lambda x: x[1], reverse=True)

            print(f"\n{method}:")
            print(f"  Top 3 bins:")
            for bin_label, pct in sorted_bins[:3]:
                print(f"    {bin_label}: {pct:.2f}%")

            # Check concentration around 0.5
            center_bins = ['[0.48, 0.52)']
            center_pct = sum(global_hist.get(b, 0) for b in center_bins)
            print(f"  Parameters in [0.48, 0.52): {center_pct:.2f}%")

    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Utility Histogram Analysis Complete!")
    print("=" * 70)
    print()
    print("Generated plots:")
    print("  - Global utility distribution (linear and log scale)")
    print("  - Per-layer comparisons for each method")
    print()
    print("Key insights:")
    print("  - Most UPGD variants concentrate ~99%+ of parameters near 0.5 utility")
    print("  - This indicates balanced parameter importance")
    print("  - Output-only gating shows slightly more spread")
    print()

if __name__ == '__main__':
    main()
