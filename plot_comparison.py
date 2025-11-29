#!/usr/bin/env python3
"""
Advanced plotting script for comparing multiple input_permuted_mnist experimental runs.
This script can handle multiple algorithms, seeds, and configurations.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple
import glob
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 12})
sns.set_palette("husl")

def find_all_experiments(base_dir: str) -> Dict[str, List[str]]:
    """Find all experimental JSON files organized by algorithm."""
    experiments = {}
    
    # Search for all JSON files
    json_files = glob.glob(f"{base_dir}/**/*.json", recursive=True)
    
    for filepath in json_files:
        # Extract algorithm name from path
        parts = Path(filepath).parts
        if 'logs' in parts:
            idx = parts.index('logs')
            if idx + 2 < len(parts):
                algo_name = parts[idx + 2]  # e.g., 'pgd', 'sgd', etc.
                if algo_name not in experiments:
                    experiments[algo_name] = []
                experiments[algo_name].append(filepath)
    
    return experiments

def load_experiment_data(filepath: str) -> Dict[str, Any]:
    """Load and validate experimental data."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None

def plot_algorithm_comparison(experiments: Dict[str, List[str]], output_dir: str, 
                            metric: str = 'losses', smooth_window: int = 100):
    """Compare different algorithms on the same plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = sns.color_palette("husl", len(experiments))
    
    for idx, (algo_name, filepaths) in enumerate(experiments.items()):
        all_data = []
        
        # Load all seeds for this algorithm
        for filepath in filepaths:
            data = load_experiment_data(filepath)
            if data and metric in data:
                all_data.append(data[metric])
        
        if not all_data:
            print(f"No valid data found for {algo_name}")
            continue
        
        # Ensure all runs have the same length (truncate to minimum)
        min_length = min(len(d) for d in all_data)
        all_data = [d[:min_length] for d in all_data]
        all_data = np.array(all_data)
        
        # Calculate statistics
        mean_data = np.mean(all_data, axis=0)
        std_data = np.std(all_data, axis=0)
        se_data = std_data / np.sqrt(len(all_data))
        
        # Plot raw data
        ax1.plot(mean_data, label=f'{algo_name} (n={len(all_data)})', 
                color=colors[idx], linewidth=2, alpha=0.8)
        ax1.fill_between(range(len(mean_data)), 
                        mean_data - se_data, 
                        mean_data + se_data, 
                        alpha=0.2, color=colors[idx])
        
        # Plot smoothed data
        if smooth_window > 1 and len(mean_data) > smooth_window:
            smoothed = np.convolve(mean_data, 
                                  np.ones(smooth_window)/smooth_window, 
                                  mode='valid')
            ax2.plot(smoothed, label=f'{algo_name} (smoothed)', 
                    color=colors[idx], linewidth=2)
    
    # Configure axes
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss' if metric == 'losses' else 'Accuracy')
    ax1.set_title(f'Raw {metric.capitalize()} Comparison')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss' if metric == 'losses' else 'Accuracy')
    ax2.set_title(f'Smoothed {metric.capitalize()} (window={smooth_window})')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    if metric == 'accuracies':
        ax1.set_ylim([0, 1.05])
        ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path = output_dir / f'algorithm_comparison_{metric}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'algorithm_comparison_{metric}.pdf', bbox_inches='tight')
    print(f"Saved algorithm comparison to {output_path}")
    plt.close()

def plot_learning_dynamics(experiments: Dict[str, List[str]], output_dir: str,
                          n_tasks: int = 100, samples_per_task: int = 5000):
    """Analyze learning dynamics across tasks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = sns.color_palette("husl", len(experiments))
    
    for idx, (algo_name, filepaths) in enumerate(experiments.items()):
        all_losses = []
        all_accuracies = []
        
        for filepath in filepaths:
            data = load_experiment_data(filepath)
            if data:
                if 'losses' in data:
                    all_losses.append(data['losses'])
                if 'accuracies' in data:
                    all_accuracies.append(data['accuracies'])
        
        if not all_losses:
            continue
        
        # Ensure consistent length
        min_length = min(len(d) for d in all_losses)
        all_losses = [d[:min_length] for d in all_losses]
        
        # Calculate per-task metrics
        n_actual_tasks = min(n_tasks, min_length // samples_per_task)
        
        task_final_losses = []
        task_avg_losses = []
        task_improvement = []
        
        for seed_losses in all_losses:
            seed_task_final = []
            seed_task_avg = []
            seed_task_improve = []
            
            for t in range(n_actual_tasks):
                start = t * samples_per_task
                end = min((t + 1) * samples_per_task, len(seed_losses))
                task_data = seed_losses[start:end]
                
                if len(task_data) > 0:
                    seed_task_final.append(task_data[-1])
                    seed_task_avg.append(np.mean(task_data))
                    if len(task_data) > 1:
                        seed_task_improve.append(task_data[0] - task_data[-1])
                    else:
                        seed_task_improve.append(0)
            
            task_final_losses.append(seed_task_final)
            task_avg_losses.append(seed_task_avg)
            task_improvement.append(seed_task_improve)
        
        # Plot metrics
        task_numbers = range(1, n_actual_tasks + 1)
        
        # Final loss per task
        ax = axes[0, 0]
        mean_final = np.mean(task_final_losses, axis=0)
        se_final = np.std(task_final_losses, axis=0) / np.sqrt(len(task_final_losses))
        ax.plot(task_numbers, mean_final, marker='o', label=algo_name, 
               color=colors[idx], linewidth=2)
        ax.fill_between(task_numbers, mean_final - se_final, mean_final + se_final,
                       alpha=0.2, color=colors[idx])
        
        # Average loss per task
        ax = axes[0, 1]
        mean_avg = np.mean(task_avg_losses, axis=0)
        se_avg = np.std(task_avg_losses, axis=0) / np.sqrt(len(task_avg_losses))
        ax.plot(task_numbers, mean_avg, marker='s', label=algo_name,
               color=colors[idx], linewidth=2)
        ax.fill_between(task_numbers, mean_avg - se_avg, mean_avg + se_avg,
                       alpha=0.2, color=colors[idx])
        
        # Improvement per task
        ax = axes[1, 0]
        mean_improve = np.mean(task_improvement, axis=0)
        se_improve = np.std(task_improvement, axis=0) / np.sqrt(len(task_improvement))
        ax.plot(task_numbers, mean_improve, marker='^', label=algo_name,
               color=colors[idx], linewidth=2)
        ax.fill_between(task_numbers, mean_improve - se_improve, mean_improve + se_improve,
                       alpha=0.2, color=colors[idx])
        
        # Plasticity metric (variance of improvements)
        ax = axes[1, 1]
        plasticity = np.std(task_improvement, axis=0)
        ax.plot(task_numbers, plasticity, marker='d', label=algo_name,
               color=colors[idx], linewidth=2)
    
    # Configure all axes
    axes[0, 0].set_xlabel('Task Number')
    axes[0, 0].set_ylabel('Final Loss')
    axes[0, 0].set_title('Final Loss per Task')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Task Number')
    axes[0, 1].set_ylabel('Average Loss')
    axes[0, 1].set_title('Average Loss per Task')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Task Number')
    axes[1, 0].set_ylabel('Loss Improvement')
    axes[1, 0].set_title('Within-Task Improvement')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    axes[1, 1].set_xlabel('Task Number')
    axes[1, 1].set_ylabel('Std Dev of Improvement')
    axes[1, 1].set_title('Plasticity Metric (Variance in Learning)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'learning_dynamics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'learning_dynamics.pdf', bbox_inches='tight')
    print(f"Saved learning dynamics to {output_path}")
    plt.close()

def plot_final_performance_summary(experiments: Dict[str, List[str]], output_dir: str):
    """Create a summary bar plot of final performance metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = {}
    
    for algo_name, filepaths in experiments.items():
        final_losses = []
        final_accs = []
        avg_losses = []
        avg_accs = []
        
        for filepath in filepaths:
            data = load_experiment_data(filepath)
            if data:
                if 'losses' in data and len(data['losses']) > 0:
                    final_losses.append(data['losses'][-1])
                    avg_losses.append(np.mean(data['losses']))
                if 'accuracies' in data and len(data['accuracies']) > 0:
                    final_accs.append(data['accuracies'][-1])
                    avg_accs.append(np.mean(data['accuracies']))
        
        summary_data[algo_name] = {
            'final_loss': (np.mean(final_losses), np.std(final_losses)) if final_losses else (None, None),
            'final_acc': (np.mean(final_accs), np.std(final_accs)) if final_accs else (None, None),
            'avg_loss': (np.mean(avg_losses), np.std(avg_losses)) if avg_losses else (None, None),
            'avg_acc': (np.mean(avg_accs), np.std(avg_accs)) if avg_accs else (None, None),
            'n_seeds': len(filepaths)
        }
    
    # Create bar plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    algorithms = list(summary_data.keys())
    x_pos = np.arange(len(algorithms))
    
    # Final Loss
    ax = axes[0, 0]
    means = [summary_data[a]['final_loss'][0] for a in algorithms if summary_data[a]['final_loss'][0] is not None]
    stds = [summary_data[a]['final_loss'][1] for a in algorithms if summary_data[a]['final_loss'][0] is not None]
    valid_algos = [a for a in algorithms if summary_data[a]['final_loss'][0] is not None]
    if means:
        bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(valid_algos, rotation=45)
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Comparison')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Final Accuracy
    ax = axes[0, 1]
    means = [summary_data[a]['final_acc'][0] for a in algorithms if summary_data[a]['final_acc'][0] is not None]
    stds = [summary_data[a]['final_acc'][1] for a in algorithms if summary_data[a]['final_acc'][0] is not None]
    valid_algos = [a for a in algorithms if summary_data[a]['final_acc'][0] is not None]
    if means:
        bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5, color='green')
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(valid_algos, rotation=45)
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Final Accuracy Comparison')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
    
    # Average Loss
    ax = axes[1, 0]
    means = [summary_data[a]['avg_loss'][0] for a in algorithms if summary_data[a]['avg_loss'][0] is not None]
    stds = [summary_data[a]['avg_loss'][1] for a in algorithms if summary_data[a]['avg_loss'][0] is not None]
    valid_algos = [a for a in algorithms if summary_data[a]['avg_loss'][0] is not None]
    if means:
        bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5, color='orange')
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(valid_algos, rotation=45)
        ax.set_ylabel('Average Loss')
        ax.set_title('Average Loss Throughout Training')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Number of seeds
    ax = axes[1, 1]
    n_seeds = [summary_data[a]['n_seeds'] for a in algorithms]
    bars = ax.bar(range(len(algorithms)), n_seeds, color='purple')
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=45)
    ax.set_ylabel('Number of Seeds')
    ax.set_title('Experiment Replication Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, n_seeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = output_dir / 'performance_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_summary.pdf', bbox_inches='tight')
    print(f"Saved performance summary to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Advanced plotting for input_permuted_mnist experiments')
    parser.add_argument('--logs-dir', type=str, 
                       default='/scratch/gautschi/shin283/upgd/logs/input_permuted_mnist',
                       help='Base directory containing experimental logs')
    parser.add_argument('--output-dir', type=str, default='./plots/comparison',
                       help='Directory to save plots')
    parser.add_argument('--smooth-window', type=int, default=100,
                       help='Window size for smoothing')
    parser.add_argument('--n-tasks', type=int, default=100,
                       help='Number of tasks to analyze')
    parser.add_argument('--samples-per-task', type=int, default=5000,
                       help='Number of samples per task')
    
    args = parser.parse_args()
    
    print(f"Searching for experiments in: {args.logs_dir}")
    
    # Find all experiments
    experiments = find_all_experiments(args.logs_dir)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"Found experiments for algorithms: {list(experiments.keys())}")
    for algo, files in experiments.items():
        print(f"  {algo}: {len(files)} runs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all comparison plots
    print("\nGenerating comparison plots...")
    
    # Loss comparison
    plot_algorithm_comparison(experiments, output_dir, 'losses', args.smooth_window)
    
    # Accuracy comparison if available
    has_accuracy = False
    for algo, files in experiments.items():
        for f in files:
            data = load_experiment_data(f)
            if data and 'accuracies' in data:
                has_accuracy = True
                break
        if has_accuracy:
            break
    
    if has_accuracy:
        plot_algorithm_comparison(experiments, output_dir, 'accuracies', args.smooth_window)
    
    # Learning dynamics
    plot_learning_dynamics(experiments, output_dir, args.n_tasks, args.samples_per_task)
    
    # Performance summary
    plot_final_performance_summary(experiments, output_dir)
    
    print(f"\nAll comparison plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
