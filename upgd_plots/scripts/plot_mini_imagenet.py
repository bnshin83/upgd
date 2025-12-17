#!/usr/bin/env python3
"""
Plotting script for Mini-ImageNet experiments.
Compares UPGD variants: output-only, hidden-only, hidden+output, clamped versions, and baselines.
"""

import json
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

# Paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent  # upgd_plots/
PROJECT_DIR = PLOTS_DIR.parent  # upgd/
LOGS_DIR = PROJECT_DIR / 'logs' / 'label_permuted_mini_imagenet_stats'
PLOT_DIR = PLOTS_DIR / 'figures' / 'mini_imagenet'

# Define experiments to compare (matching the user's job IDs)
EXPERIMENTS = {
    'S&P': {
        'path': 'sgd/fully_connected_relu_with_hooks/lr_0.005_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
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
    'UPGD (Hidden Only)': {
        'path': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
        'color': '#ff7f0e',
        'linestyle': '-',
    },
    'UPGD (Hidden+Output)': {
        'path': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
        'color': '#9467bd',
        'linestyle': '-',
    },
    'UPGD (Clamped 0.52)': {
        'path': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
        'color': '#d62728',
        'linestyle': '-',
    },
    'UPGD (Clamped 0.48-0.52)': {
        'path': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
        'color': '#8c564b',
        'linestyle': '-',
    },
    'UPGD (Clamped 0.44-0.56)': {
        'path': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        'color': '#e377c2',
        'linestyle': '-',
    },
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


def plot_metric_comparison(metric_key: str, ylabel: str, title: str,
                           output_name: str, window: int = 1,
                           ylim: tuple = None):
    """Generic function to plot any metric comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    results = {}

    for name, config in EXPERIMENTS.items():
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

        # Convert per-step (1M points) to per-task window averages (400 tasks)
        # Each task = 2500 steps, matching original UPGD behavior
        steps_per_task = 2500
        n_tasks = len(mean_metric) // steps_per_task

        per_task_avg = []
        for task_idx in range(n_tasks):
            start = task_idx * steps_per_task
            end = (task_idx + 1) * steps_per_task
            per_task_avg.append(np.mean(mean_metric[start:end]))

        per_task_avg = np.array(per_task_avg)

        # Convert task numbers to step numbers for x-axis
        task_steps = np.arange(n_tasks) * steps_per_task

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
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=10)
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
    print("SUMMARY: Mini-ImageNet Experiment Results")
    print("=" * 80)

    methods = list(EXPERIMENTS.keys())

    for metric_name, results in all_results.items():
        if not results:
            continue
        print(f"\n{metric_name}:")
        print("-" * 60)
        print(f"{'Method':<30} {'Overall Avg':>15} {'Seeds':>10}")
        print("-" * 60)

        sorted_results = sorted(
            [(m, results.get(m, {})) for m in methods if m in results],
            key=lambda x: x[1].get('overall_avg', 0),
            reverse=True
        )
        for name, stats in sorted_results:
            avg = stats.get('overall_avg', 0)
            seeds = stats.get('n_seeds', 0)
            print(f"{name:<30} {avg:>15.4f} {seeds:>10}")

    print("\n" + "=" * 80)


def main():
    # Create output directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {PLOT_DIR}")

    all_results = {}

    # Compute per-task window averages from per-step data (matching original UPGD)
    # This converts 1M per-step points to 400 per-task window averages (2500 steps each)
    USE_PER_TASK_WINDOW = True

    if USE_PER_TASK_WINDOW:
        print("\nComputing per-task window averages (2500 steps per task)...")

    # 1. Accuracy comparison
    print("\n[1/7] Plotting Accuracy...")
    acc_results = plot_metric_comparison(
        'accuracy_per_step', 'Accuracy',
        'Mini-ImageNet: Accuracy Comparison',
        'accuracy_comparison', window=1
    )
    all_results['Accuracy'] = acc_results

    # 2. Loss comparison
    print("\n[2/7] Plotting Loss...")
    loss_results = plot_metric_comparison(
        'losses_per_step', 'Loss',
        'Mini-ImageNet: Loss Comparison',
        'loss_comparison', window=1
    )
    all_results['Loss'] = loss_results

    # 3. Plasticity comparison
    print("\n[3/7] Plotting Plasticity...")
    plast_results = plot_metric_comparison(
        'plasticity_per_step', 'Plasticity',
        'Mini-ImageNet: Plasticity Comparison',
        'plasticity_comparison', window=1
    )
    if plast_results:
        all_results['Plasticity'] = plast_results

    # 4. Dead units comparison
    print("\n[4/7] Plotting Dead Units...")
    dead_results = plot_metric_comparison(
        'n_dead_units_per_step', 'Dead Units',
        'Mini-ImageNet: Dead Units Comparison',
        'dead_units_comparison', window=1
    )
    if dead_results:
        all_results['Dead Units'] = dead_results

    # 5. Weight L2 norm
    print("\n[5/7] Plotting Weight L2 Norm...")
    weight_l2_results = plot_metric_comparison(
        'weight_l2_per_step', 'Weight L2 Norm',
        'Mini-ImageNet: Weight L2 Norm Comparison',
        'weight_l2_comparison', window=1
    )
    if weight_l2_results:
        all_results['Weight L2'] = weight_l2_results

    # 6. Weight L1 norm
    print("\n[6/7] Plotting Weight L1 Norm...")
    weight_l1_results = plot_metric_comparison(
        'weight_l1_per_step', 'Weight L1 Norm',
        'Mini-ImageNet: Weight L1 Norm Comparison',
        'weight_l1_comparison', window=1
    )
    if weight_l1_results:
        all_results['Weight L1'] = weight_l1_results

    # 7. Gradient L2 norm
    print("\n[7/7] Plotting Gradient L2 Norm...")
    grad_l2_results = plot_metric_comparison(
        'grad_l2_per_step', 'Gradient L2 Norm',
        'Mini-ImageNet: Gradient L2 Norm Comparison',
        'grad_l2_comparison', window=1
    )
    if grad_l2_results:
        all_results['Gradient L2'] = grad_l2_results

    # Print summary
    print_summary_table(all_results)

    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
