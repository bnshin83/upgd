#!/usr/bin/env python3
"""
Quick script to plot utility histograms for specific methods.

This recreates the utility_histogram_log.png plot but only for
UPGD (Full) and UPGD (Output Only) methods.
"""

import sys
from pathlib import Path

# Add upgd_plots to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data import UtilityHistogramLoaderAgent
from agents.plot import UtilityHistogramPlotAgent
from config import default_config

# Initialize agents
loader = UtilityHistogramLoaderAgent(config=default_config)
plotter = UtilityHistogramPlotAgent(config=default_config)

# Load utility histogram data for CIFAR-10
hist_data = loader.execute(
    dataset='cifar10',  # Change to 'mini_imagenet', 'input_mnist', 'emnist' as needed
    methods=['UPGD (Full)', 'UPGD (Output Only)']  # Only these two methods
)

# Create the plot (log scale)
fig, path = plotter.execute(
    histogram_data=hist_data,
    methods=['UPGD (Full)', 'UPGD (Output Only)'],
    plot_type='scatter',  # 'scatter' or 'bar'
    log_scale=True,       # Log scale like the original
    per_layer=False,      # Global histogram (not per-layer)
    title='Utility Distribution - CIFAR-10',
    subdir='cifar10',
    filename='utility_histogram_log'
)

print(f"Plot saved to: {path}")
