#!/usr/bin/env python3
"""
Cross-dataset analysis plots for UPGD experiments.

Generates 4 unified comparison plots:
1. Layer-Selective Gating Comparison (bar chart)
2. Dead Unit Patterns Across Datasets (grouped bar chart)
3. Utility Clamping Degradation (line chart)
4. Plasticity Comparison (grouped bar chart)

Usage:
    python plot_cross_dataset.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 12})

# Base paths
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PLOTS_DIR.parent
BASE_DIR = PROJECT_DIR
LOGS_DIR = BASE_DIR / 'logs'
PLOT_DIR = PLOTS_DIR / 'figures' / 'cross_dataset'

# Ensure output directory exists
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# COLOR SCHEME
# ============================================================================
COLORS = {
    'S&P': '#7f7f7f',
    'Full': '#1f77b4',
    'Output Only': '#2ca02c',
    'Hidden Only': '#ff7f0e',
    'Hidden+Output': '#9467bd',
    'Clamped 0.52': '#d62728',
    'Clamped 0.48-0.52': '#8c564b',
    'Clamped 0.44-0.56': '#e377c2',
}

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================
DATASETS = {
    'mini_imagenet': {
        'display_name': 'Mini-ImageNet',
        'logs_subdir': 'label_permuted_mini_imagenet_stats',
        'steps_per_task': 2500,
        'experiments': {
            'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.9_weight_decay_0.001_n_samples_1000000',
            'Full': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
            'Output Only': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
            'Hidden Only': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
            'Hidden+Output': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
            'Clamped 0.52': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
            'Clamped 0.48-0.52': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
            'Clamped 0.44-0.56': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        }
    },
    'cifar10': {
        'display_name': 'CIFAR-10',
        'logs_subdir': 'label_permuted_cifar10_stats',
        'steps_per_task': 2500,
        'experiments': {
            'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.999_weight_decay_0.001_n_samples_1000000',
            'Full': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_n_samples_1000000',
            'Output Only': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
            'Hidden Only': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
            'Hidden+Output': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
            'Clamped 0.52': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_n_samples_1000000',
            'Clamped 0.48-0.52': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
            'Clamped 0.44-0.56': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        }
    },
    'emnist': {
        'display_name': 'EMNIST',
        'logs_subdir': 'label_permuted_emnist_stats',
        'steps_per_task': 2500,
        'experiments': {
            'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.9_weight_decay_0.001_n_samples_1000000',
            'Full': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
            'Output Only': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
            'Hidden Only': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
            'Hidden+Output': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
            'Clamped 0.52': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
            'Clamped 0.48-0.52': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
            'Clamped 0.44-0.56': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        }
    },
    'input_mnist': {
        'display_name': 'Input-MNIST',
        'logs_subdir': 'input_permuted_mnist_stats',
        'steps_per_task': 5000,
        'experiments': {
            'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_weight_decay_0.01_beta_utility_0.9999_n_samples_1000000',
            'Full': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
            'Output Only': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_output_only_n_samples_1000000',
            'Hidden Only': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_hidden_only_n_samples_1000000',
            'Hidden+Output': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_hidden_and_output_n_samples_1000000',
            'Clamped 0.52': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
            'Clamped 0.48-0.52': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
            'Clamped 0.44-0.56': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        }
    },
}


def load_json_data(filepath: Path) -> dict:
    """Load JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def get_experiment_data(dataset_key: str, exp_name: str) -> list:
    """Load all seed data for an experiment."""
    config = DATASETS[dataset_key]
    exp_path = LOGS_DIR / config['logs_subdir'] / config['experiments'].get(exp_name, '')
    
    if not exp_path.exists():
        return []
    
    json_files = list(exp_path.glob('*.json'))
    all_data = []
    for jf in json_files:
        data = load_json_data(jf)
        if data:
            all_data.append(data)
    return all_data


def compute_final_metric(data_list: list, metric_key: str, steps_per_task: int,
                         last_n_tasks: int = 40) -> tuple:
    """Compute mean and std of final metric averaged over last N tasks."""
    if not data_list:
        return None, None
    
    all_final_avgs = []
    for d in data_list:
        if metric_key not in d:
            continue
        metric = np.array(d[metric_key])
        n_tasks = len(metric) // steps_per_task
        
        # Compute per-task averages
        per_task_avg = []
        for task_idx in range(n_tasks):
            start = task_idx * steps_per_task
            end = (task_idx + 1) * steps_per_task
            per_task_avg.append(np.mean(metric[start:end]))
        
        # Get final average over last N tasks
        per_task_avg = np.array(per_task_avg)
        if len(per_task_avg) >= last_n_tasks:
            final_avg = np.mean(per_task_avg[-last_n_tasks:])
        else:
            final_avg = np.mean(per_task_avg)
        all_final_avgs.append(final_avg)
    
    if not all_final_avgs:
        return None, None
    
    return np.mean(all_final_avgs), np.std(all_final_avgs)


def plot_layer_selective_gating():
    """
    Plot 1: Layer-Selective Gating Comparison
    Bar chart comparing accuracy across gating strategies for all 4 datasets.
    """
    print("\n[1/6] Plotting Layer-Selective Gating Comparison...")
    
    methods = ['Full', 'Output Only', 'Hidden Only', 'Hidden+Output']
    dataset_keys = ['mini_imagenet', 'cifar10', 'emnist', 'input_mnist']
    dataset_names = [DATASETS[k]['display_name'] for k in dataset_keys]
    
    # Collect data
    data = {method: {'means': [], 'stds': []} for method in methods}
    
    for dk in dataset_keys:
        steps_per_task = DATASETS[dk]['steps_per_task']
        for method in methods:
            exp_data = get_experiment_data(dk, method)
            mean, std = compute_final_metric(exp_data, 'accuracy_per_step', steps_per_task)
            data[method]['means'].append(mean if mean is not None else 0)
            data[method]['stds'].append(std if std is not None else 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(dataset_names))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    
    for i, method in enumerate(methods):
        means = data[method]['means']
        stds = data[method]['stds']
        bars = ax.bar(x + offsets[i] * width, means, width, 
                      label=method, color=COLORS.get(method, '#333333'),
                      yerr=stds, capsize=3, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Final Accuracy', fontsize=14)
    ax.set_title('Layer-Selective Gating Comparison Across Datasets', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = PLOT_DIR / 'layer_selective_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_dead_units_comparison():
    """
    Plot 2: Dead Unit Patterns Across Datasets
    Grouped bar chart showing dead unit fraction for S&P, Output Only, Full.
    """
    print("\n[2/6] Plotting Dead Unit Patterns Comparison...")
    
    methods = ['S&P', 'Output Only', 'Full']
    dataset_keys = ['mini_imagenet', 'cifar10', 'emnist', 'input_mnist']
    dataset_names = [DATASETS[k]['display_name'] for k in dataset_keys]
    
    # Network has ~3200 total units (3 hidden layers of 1024 each + output)
    # Normalize by approximate network size for interpretability
    total_units = 3200
    
    # Collect data
    data = {method: {'means': [], 'stds': []} for method in methods}
    
    for dk in dataset_keys:
        steps_per_task = DATASETS[dk]['steps_per_task']
        for method in methods:
            exp_data = get_experiment_data(dk, method)
            mean, std = compute_final_metric(exp_data, 'n_dead_units_per_step', steps_per_task)
            # Normalize to fraction
            if mean is not None:
                data[method]['means'].append(mean / total_units)
                data[method]['stds'].append(std / total_units if std else 0)
            else:
                data[method]['means'].append(0)
                data[method]['stds'].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(dataset_names))
    width = 0.25
    offsets = [-1, 0, 1]
    
    for i, method in enumerate(methods):
        means = data[method]['means']
        stds = data[method]['stds']
        bars = ax.bar(x + offsets[i] * width, means, width,
                      label=method, color=COLORS.get(method, '#333333'),
                      yerr=stds, capsize=3, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Dead Unit Fraction', fontsize=14)
    ax.set_title('Dead Unit Patterns Across Datasets', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation for label-permuted vs input-permuted
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.25, ax.get_ylim()[1] * 0.95, 'Label-Permuted', ha='center', fontsize=11, style='italic')
    ax.text(3.0, ax.get_ylim()[1] * 0.95, 'Input-Permuted', ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    output_path = PLOT_DIR / 'dead_units_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_clamping_degradation():
    """
    Plot 3: Utility Clamping Degradation
    Line chart showing accuracy vs clamping level for each dataset.
    """
    print("\n[3/6] Plotting Utility Clamping Degradation...")
    
    clamping_levels = ['Full', 'Clamped 0.44-0.56', 'Clamped 0.48-0.52', 'Clamped 0.52']
    clamping_x_labels = ['No Clamp', '[0.44, 0.56]', '[0.48, 0.52]', '≥0.52']
    
    # Use label-permuted datasets for this plot
    dataset_keys = ['mini_imagenet', 'cifar10', 'emnist']
    dataset_colors = {
        'mini_imagenet': '#1f77b4',
        'cifar10': '#ff7f0e',
        'emnist': '#2ca02c',
    }
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for dk in dataset_keys:
        steps_per_task = DATASETS[dk]['steps_per_task']
        display_name = DATASETS[dk]['display_name']
        
        means = []
        stds = []
        for level in clamping_levels:
            exp_data = get_experiment_data(dk, level)
            mean, std = compute_final_metric(exp_data, 'accuracy_per_step', steps_per_task)
            means.append(mean if mean is not None else 0)
            stds.append(std if std is not None else 0)
        
        x = np.arange(len(clamping_levels))
        ax.errorbar(x, means, yerr=stds, label=display_name,
                    color=dataset_colors[dk], marker='o', markersize=8,
                    linewidth=2, capsize=4)
    
    ax.set_xlabel('Clamping Range', fontsize=14)
    ax.set_ylabel('Final Accuracy', fontsize=14)
    ax.set_title('Accuracy Degradation with Tighter Utility Clamping', fontsize=16)
    ax.set_xticks(np.arange(len(clamping_levels)))
    ax.set_xticklabels(clamping_x_labels)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = PLOT_DIR / 'clamping_degradation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_plasticity_comparison():
    """
    Plot 4: Plasticity Comparison
    Grouped bar chart for S&P, Output Only, Full across all datasets.
    """
    print("\n[4/6] Plotting Plasticity Comparison...")
    
    methods = ['S&P', 'Output Only', 'Full']
    dataset_keys = ['mini_imagenet', 'cifar10', 'emnist', 'input_mnist']
    dataset_names = [DATASETS[k]['display_name'] for k in dataset_keys]
    
    # Collect data
    data = {method: {'means': [], 'stds': []} for method in methods}
    
    for dk in dataset_keys:
        steps_per_task = DATASETS[dk]['steps_per_task']
        for method in methods:
            exp_data = get_experiment_data(dk, method)
            mean, std = compute_final_metric(exp_data, 'plasticity_per_step', steps_per_task)
            data[method]['means'].append(mean if mean is not None else 0)
            data[method]['stds'].append(std if std is not None else 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(dataset_names))
    width = 0.25
    offsets = [-1, 0, 1]
    
    for i, method in enumerate(methods):
        means = data[method]['means']
        stds = data[method]['stds']
        bars = ax.bar(x + offsets[i] * width, means, width,
                      label=method, color=COLORS.get(method, '#333333'),
                      yerr=stds, capsize=3, alpha=0.85)
    
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Plasticity', fontsize=14)
    ax.set_title('Plasticity Comparison Across Datasets', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = PLOT_DIR / 'plasticity_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def get_utility_histogram_data(dataset_key: str, exp_name: str) -> dict:
    """Extract utility histogram data from experiment JSON files."""
    config = DATASETS[dataset_key]
    exp_path = LOGS_DIR / config['logs_subdir'] / config['experiments'].get(exp_name, '')
    
    if not exp_path.exists():
        return None
    
    json_files = list(exp_path.glob('*.json'))
    if not json_files:
        return None
    
    # Load first available seed
    for jf in json_files:
        data = load_json_data(jf)
        if data and 'utility_histogram_per_step' in data:
            return data['utility_histogram_per_step']
    return None


def plot_utility_distribution_comparison():
    """
    Plot 5: Utility Distribution Comparison
    2×2 grid showing utility histograms (log scale) for all 4 datasets.
    """
    print("\n[5/6] Plotting Utility Distribution Comparison...")
    
    dataset_keys = ['mini_imagenet', 'emnist', 'cifar10', 'input_mnist']
    methods = ['Full', 'Output Only', 'Hidden Only', 'Hidden+Output']
    
    # Actual histogram keys from JSON data structure
    hist_keys = [
        'hist_0_20_pct',      # <0.40 (0-20% utility mapped to ~0.0-0.4)
        'hist_20_40_pct',     # 0.20-0.40
        'hist_40_44_pct',     # 0.40-0.44
        'hist_44_48_pct',     # 0.44-0.48
        'hist_48_52_pct',     # 0.48-0.52
        'hist_52_56_pct',     # 0.52-0.56
        'hist_56_60_pct',     # 0.56-0.60
        'hist_60_80_pct',     # 0.60-0.80
        'hist_80_100_pct',    # >0.80
    ]
    
    # Display labels for bins
    bin_labels = ['<0.20', '0.20-0.40', '0.40-0.44', '0.44-0.48', 
                  '0.48-0.52', '0.52-0.56', '0.56-0.60', '0.60-0.80', '>0.80']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()
    
    for idx, dk in enumerate(dataset_keys):
        ax = axes_flat[idx]
        display_name = DATASETS[dk]['display_name']
        
        for method in methods:
            hist_data = get_utility_histogram_data(dk, method)
            if hist_data is None:
                continue
            
            try:
                # Extract histogram values using correct keys
                counts = []
                for key in hist_keys:
                    if key in hist_data:
                        vals = hist_data[key]
                        # Average last 100 samples or all if fewer
                        if isinstance(vals, list) and len(vals) > 0:
                            avg_val = np.mean(vals[-100:]) if len(vals) > 100 else np.mean(vals)
                            counts.append(avg_val * 100)  # Convert to percentage
                        else:
                            counts.append(0)
                    else:
                        counts.append(0)
                
                counts = np.array(counts)
                
                # Skip if all zeros
                if np.sum(counts) == 0:
                    print(f"    Warning: All zeros for {method} on {dk}")
                    continue
                
                x = np.arange(len(bin_labels))
                ax.plot(x, counts, marker='o', label=method, 
                       color=COLORS.get(method, '#333333'), linewidth=2, markersize=6)
            except Exception as e:
                print(f"    Warning: Could not process histogram for {method} on {dk}: {e}")
                continue
        
        ax.set_title(display_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Utility Bin', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_yscale('log')  # Log scale for y-axis
        ax.set_xticks(np.arange(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Utility Distribution Comparison Across Datasets', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = PLOT_DIR / 'utility_distribution_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def get_layer_utility_data(dataset_key: str, exp_name: str) -> dict:
    """Extract per-layer utility histogram data from experiment JSON files."""
    config = DATASETS[dataset_key]
    exp_path = LOGS_DIR / config['logs_subdir'] / config['experiments'].get(exp_name, '')
    
    if not exp_path.exists():
        return None
    
    json_files = list(exp_path.glob('*.json'))
    if not json_files:
        return None
    
    # Load first available seed
    for jf in json_files:
        data = load_json_data(jf)
        if data and 'layer_utility_histogram_per_step' in data:
            return data['layer_utility_histogram_per_step']
    return None


def compute_tail_mass(layer_hist: dict) -> float:
    """
    Compute the tail mass (proportion of utilities > 0.52) for a layer.
    High utility bins are: hist_52_56_pct, hist_56_60_pct, hist_60_80_pct, hist_80_100_pct
    """
    try:
        if not layer_hist:
            return 0.0
        
        # Keys for all bins
        all_bins = [
            'hist_0_20_pct', 'hist_20_40_pct', 'hist_40_44_pct', 'hist_44_48_pct',
            'hist_48_52_pct', 'hist_52_56_pct', 'hist_56_60_pct', 'hist_60_80_pct', 'hist_80_100_pct'
        ]
        # High utility bins (>0.52)
        high_bins = ['hist_52_56_pct', 'hist_56_60_pct', 'hist_60_80_pct', 'hist_80_100_pct']
        
        total = 0
        tail = 0
        for key in all_bins:
            if key in layer_hist:
                vals = layer_hist[key]
                if isinstance(vals, list) and len(vals) > 0:
                    avg_val = np.mean(vals[-100:]) if len(vals) > 100 else np.mean(vals)
                    total += avg_val
                    if key in high_bins:
                        tail += avg_val
        
        return (tail / total * 100) if total > 0 else 0.0
    except:
        return 0.0


def plot_per_layer_utility_comparison():
    """
    Plot 6: Per-Layer Utility Comparison
    Grouped bar chart comparing output vs hidden layer utility tail mass across datasets.
    """
    print("\n[6/6] Plotting Per-Layer Utility Comparison...")
    
    dataset_keys = ['mini_imagenet', 'cifar10', 'emnist', 'input_mnist']
    dataset_names = [DATASETS[dk]['display_name'] for dk in dataset_keys]
    
    # We'll compare Full UPGD's hidden layers avg vs output layer
    layer_types = ['Hidden Layers (avg)', 'Output Layer']
    
    # Collect data
    data = {lt: [] for lt in layer_types}
    
    for dk in dataset_keys:
        layer_hist = get_layer_utility_data(dk, 'Full')
        
        if layer_hist is None:
            data['Hidden Layers (avg)'].append(0)
            data['Output Layer'].append(0)
            continue
        
        try:
            # Layer keys are 'linear_1', 'linear_2', 'linear_3'
            # linear_3 is output layer, others are hidden layers
            hidden_tails = []
            output_tail = 0
            
            for layer_key in layer_hist:
                tail = compute_tail_mass(layer_hist[layer_key])
                if 'linear_3' in layer_key or layer_key == 'output':
                    output_tail = tail
                elif layer_key.startswith('linear_'):
                    hidden_tails.append(tail)
            
            hidden_avg = np.mean(hidden_tails) if hidden_tails else 0
            data['Hidden Layers (avg)'].append(hidden_avg)
            data['Output Layer'].append(output_tail)
        except Exception as e:
            print(f"    Warning: Could not process layer data for {dk}: {e}")
            data['Hidden Layers (avg)'].append(0)
            data['Output Layer'].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['Hidden Layers (avg)'], width, 
                   label='Hidden Layers (avg)', color='#ff7f0e', alpha=0.85)
    bars2 = ax.bar(x + width/2, data['Output Layer'], width,
                   label='Output Layer', color='#2ca02c', alpha=0.85)
    
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('High Utility Tail Mass (%)', fontsize=14)
    ax.set_title('Per-Layer Utility Distribution: Hidden vs Output (UPGD Full)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_yscale('log')  # Log scale for y-axis
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = PLOT_DIR / 'per_layer_utility_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("Cross-Dataset Analysis Plots")
    print("=" * 60)
    print(f"Output directory: {PLOT_DIR}")
    
    # Generate all 6 plots
    plot_layer_selective_gating()
    plot_dead_units_comparison()
    plot_clamping_degradation()
    plot_plasticity_comparison()
    plot_utility_distribution_comparison()
    plot_per_layer_utility_comparison()
    
    print("\n" + "=" * 60)
    print("All 6 plots generated successfully!")
    print(f"Output directory: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

