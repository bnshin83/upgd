#!/usr/bin/env python3
"""
Plotting script for UPGD scale ablation experiments on EMNIST.

Usage:
    python plot_ablation_scale.py

Compares:
- Scale ablation: scale=0.0, 0.27, 0.5, 0.73, 1.0 (hidden layer SGD scale)
- Freeze high utility (s >= 0.52): freeze parameters with scaled_utility >= 0.52
- Output frozen: output layer frozen, hidden layers full SGD
- Baselines: S&P, UPGD Full, UPGD Output Only
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

# Base paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent  # upgd_plots/
PROJECT_DIR = PLOTS_DIR.parent  # upgd/
BASE_DIR = PROJECT_DIR
PLOT_BASE_DIR = PLOTS_DIR / 'figures'

# EMNIST configuration
LOGS_DIR = BASE_DIR / 'logs' / 'label_permuted_emnist_stats'
PLOT_DIR = PLOT_BASE_DIR / 'emnist_ablation'
DISPLAY_NAME = 'Label-Permuted EMNIST'
STEPS_PER_TASK = 2500

# ============================================================================
# ABLATION EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENTS = {
    # Baselines
    'S&P': {
        'path': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.9_weight_decay_0.001_n_samples_1000000',
        'color': '#7f7f7f',
        'linestyle': '--',
    },
    'UPGD (Full)': {
        'path': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
        'color': '#1f77b4',
        'linestyle': '-',
    },
    'UPGD (Output Only)': {
        'path': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
        'color': '#2ca02c',
        'linestyle': '-',
    },
    # Scale ablation experiments
    'Scale=0.0 (hidden frozen)': {
        'path': 'upgd_fo_global_outputonly_scale0/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_non_gated_scale_0.0_n_samples_1000000',
        'color': '#d62728',
        'linestyle': '-',
    },
    'Scale=0.27 (max protection)': {
        'path': 'upgd_fo_global_outputonly_scale27/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_non_gated_scale_0.27_n_samples_1000000',
        'color': '#ff7f0e',
        'linestyle': '-',
    },
    'Scale=0.5 (neutral)': {
        'path': 'upgd_fo_global_outputonly_scale50/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_non_gated_scale_0.5_n_samples_1000000',
        'color': '#9467bd',
        'linestyle': '-',
    },
    'Scale=0.73 (min protection)': {
        'path': 'upgd_fo_global_outputonly_scale73/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_non_gated_scale_0.73_n_samples_1000000',
        'color': '#8c564b',
        'linestyle': '-',
    },
    'Scale=1.0 (full SGD)': {
        'path': 'upgd_fo_global_outputonly_scale1/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_non_gated_scale_1.0_n_samples_1000000',
        'color': '#e377c2',
        'linestyle': '-',
    },
    # Additional ablations
    'Freeze High (s>=0.52)': {
        'path': 'upgd_fo_global_freezehigh52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_full_freeze_high_utility_True_freeze_threshold_0.52_n_samples_1000000',
        'color': '#17becf',
        'linestyle': '-.',
    },
    'Output Frozen': {
        'path': 'upgd_fo_global_outputfrozen/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_frozen_non_gated_scale_1.0_n_samples_1000000',
        'color': '#bcbd22',
        'linestyle': '-.',
    },
}

# Subset for scale-only comparison
SCALE_EXPERIMENTS = {
    'S&P': EXPERIMENTS['S&P'],
    'UPGD (Output Only)': EXPERIMENTS['UPGD (Output Only)'],
    'Scale=0.0 (hidden frozen)': EXPERIMENTS['Scale=0.0 (hidden frozen)'],
    'Scale=0.27 (max protection)': EXPERIMENTS['Scale=0.27 (max protection)'],
    'Scale=0.5 (neutral)': EXPERIMENTS['Scale=0.5 (neutral)'],
    'Scale=0.73 (min protection)': EXPERIMENTS['Scale=0.73 (min protection)'],
    'Scale=1.0 (full SGD)': EXPERIMENTS['Scale=1.0 (full SGD)'],
}


def load_json_data(filepath: str) -> dict:
    """Load JSON data from file."""
    print(f"  Loading {Path(filepath).name}...")
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"    Error: {e}")
        return None


def get_experiment_data(exp_path: Path) -> list:
    """Load all seed data for an experiment."""
    json_files = list(exp_path.glob('*.json'))
    all_data = []
    for jf in json_files:
        data = load_json_data(str(jf))
        if data:
            all_data.append(data)
    return all_data


def compute_running_avg(data: np.ndarray, window: int) -> np.ndarray:
    """Compute running average."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_metric_comparison(experiments: dict, metric_key: str, ylabel: str, title: str,
                           output_name: str, window: int = 1,
                           ylim: tuple = None):
    """Generic function to plot any metric comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    results = {}

    for name, config in experiments.items():
        exp_path = LOGS_DIR / config['path']
        all_data = get_experiment_data(exp_path)

        if not all_data:
            print(f"  No data for {name}")
            continue

        # Extract metric
        all_metrics = []
        for d in all_data:
            if metric_key in d:
                all_metrics.append(np.array(d[metric_key]))

        if not all_metrics:
            continue

        # Align lengths
        min_len = min(len(m) for m in all_metrics)
        all_metrics = np.array([m[:min_len] for m in all_metrics])

        # Statistics
        mean_metric = np.mean(all_metrics, axis=0)
        std_metric = np.std(all_metrics, axis=0)

        # Convert per-step to per-task window averages
        n_tasks = len(mean_metric) // STEPS_PER_TASK

        per_task_avg = []
        for task_idx in range(n_tasks):
            start = task_idx * STEPS_PER_TASK
            end = (task_idx + 1) * STEPS_PER_TASK
            per_task_avg.append(np.mean(mean_metric[start:end]))

        per_task_avg = np.array(per_task_avg)

        # Convert task numbers to step numbers for x-axis
        task_steps = np.arange(n_tasks) * STEPS_PER_TASK

        # Store results
        overall_avg = np.mean(per_task_avg)
        results[name] = {
            'overall_avg': overall_avg,
            'final_avg': np.mean(per_task_avg[-20:]) if len(per_task_avg) >= 20 else per_task_avg[-1],
            'n_seeds': len(all_metrics),
        }

        # Plot with task-averaged data
        ax.plot(task_steps, per_task_avg,
                label=name,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=2)

    # Format x-axis as steps in thousands
    ax.set_xlabel('Steps', fontsize=14)
    ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000])
    ax.set_xticklabels(['0', '200k', '400k', '600k', '800k', '1M'])
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'{DISPLAY_NAME}: {title}', fontsize=16)
    ax.legend(loc='best', fontsize=9)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'{output_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results


def print_summary_table(all_results: dict):
    """Print summary table of all metrics."""
    print("\n" + "=" * 80)
    print(f"SUMMARY: {DISPLAY_NAME} Ablation Results")
    print("=" * 80)

    for metric_name, results in all_results.items():
        if not results:
            continue
        print(f"\n{metric_name}:")
        print("-" * 70)
        print(f"{'Method':<35} {'Overall Avg':>15} {'Final Avg':>15}")
        print("-" * 70)

        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('overall_avg', 0),
            reverse=True
        )
        for name, stats in sorted_results:
            avg = stats.get('overall_avg', 0)
            final = stats.get('final_avg', 0)
            print(f"{name:<35} {avg:>15.4f} {final:>15.4f}")

    print("\n" + "=" * 80)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {PLOT_DIR}")

    all_results = {}

    # =========================================================================
    # Plot 1: All experiments comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("PLOTTING ALL ABLATION EXPERIMENTS")
    print("=" * 60)

    print("\n[1/7] Plotting Accuracy (All)...")
    acc_results = plot_metric_comparison(
        EXPERIMENTS, 'accuracy_per_step', 'Accuracy',
        'Accuracy Comparison (All Ablations)',
        'accuracy_all_ablations', window=1
    )
    all_results['Accuracy (All)'] = acc_results

    print("\n[2/7] Plotting Loss (All)...")
    loss_results = plot_metric_comparison(
        EXPERIMENTS, 'losses_per_step', 'Loss',
        'Loss Comparison (All Ablations)',
        'loss_all_ablations', window=1
    )
    all_results['Loss (All)'] = loss_results

    print("\n[3/7] Plotting Plasticity (All)...")
    plast_results = plot_metric_comparison(
        EXPERIMENTS, 'plasticity_per_step', 'Plasticity',
        'Plasticity Comparison (All Ablations)',
        'plasticity_all_ablations', window=1
    )
    if plast_results:
        all_results['Plasticity (All)'] = plast_results

    print("\n[4/7] Plotting Dead Units (All)...")
    dead_results = plot_metric_comparison(
        EXPERIMENTS, 'n_dead_units_per_step', 'Dead Units',
        'Dead Units Comparison (All Ablations)',
        'dead_units_all_ablations', window=1
    )
    if dead_results:
        all_results['Dead Units (All)'] = dead_results

    # =========================================================================
    # Plot 2: Scale-only comparison (cleaner)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PLOTTING SCALE ABLATION ONLY")
    print("=" * 60)

    print("\n[5/7] Plotting Accuracy (Scale)...")
    acc_scale_results = plot_metric_comparison(
        SCALE_EXPERIMENTS, 'accuracy_per_step', 'Accuracy',
        'Scale Ablation: Accuracy',
        'accuracy_scale_ablation', window=1
    )
    all_results['Accuracy (Scale)'] = acc_scale_results

    print("\n[6/7] Plotting Plasticity (Scale)...")
    plast_scale_results = plot_metric_comparison(
        SCALE_EXPERIMENTS, 'plasticity_per_step', 'Plasticity',
        'Scale Ablation: Plasticity',
        'plasticity_scale_ablation', window=1
    )
    if plast_scale_results:
        all_results['Plasticity (Scale)'] = plast_scale_results

    print("\n[7/7] Plotting Dead Units (Scale)...")
    dead_scale_results = plot_metric_comparison(
        SCALE_EXPERIMENTS, 'n_dead_units_per_step', 'Dead Units',
        'Scale Ablation: Dead Units',
        'dead_units_scale_ablation', window=1
    )
    if dead_scale_results:
        all_results['Dead Units (Scale)'] = dead_scale_results

    # Print summary
    print_summary_table(all_results)

    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
