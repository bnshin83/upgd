#!/usr/bin/env python3
"""
Curvature Analysis Script

Analyzes the results from input-aware UPGD experiments with curvature tracking.
This script demonstrates how to load, process, and visualize curvature data.

Usage:
    python analyze_curvature_results.py --log_dir logs/ --experiment_name input_aware_emnist
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import seaborn as sns

def load_curvature_data(log_file):
    """Load curvature data from experiment logs."""
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        return None

def analyze_curvature_statistics(data):
    """Analyze basic statistics of curvature data."""
    if 'input_curvature_per_task' not in data:
        print("No curvature data found in logs")
        return
    
    curvature_values = data['input_curvature_per_task']
    lambda_values = data.get('lambda_values_per_task', [])
    avg_curvature = data.get('avg_curvature_per_task', [])
    
    # New statistics
    curvature_max = data.get('curvature_max_per_task', [])
    curvature_min = data.get('curvature_min_per_task', [])
    curvature_std = data.get('curvature_std_per_task', [])
    
    print("=== Curvature Statistics ===")
    print(f"Number of tasks: {len(curvature_values)}")
    print(f"Curvature mean range: [{np.min(curvature_values):.6f}, {np.max(curvature_values):.6f}]")
    print(f"Overall mean curvature: {np.mean(curvature_values):.6f}")
    print(f"Overall std curvature: {np.std(curvature_values):.6f}")
    
    # Enhanced statistics if available
    if curvature_max and curvature_min and curvature_std:
        print(f"Per-task max curvature range: [{np.min(curvature_max):.6f}, {np.max(curvature_max):.6f}]")
        print(f"Per-task min curvature range: [{np.min(curvature_min):.6f}, {np.max(curvature_min):.6f}]")
        print(f"Average within-task std: {np.mean(curvature_std):.6f}")
        print(f"Max within-task variation: {np.max(curvature_std):.6f}")
    
    if lambda_values:
        print(f"Lambda range: [{np.min(lambda_values):.6f}, {np.max(lambda_values):.6f}]")
        print(f"Lambda mean: {np.mean(lambda_values):.6f}")
    
    # Identify high-curvature tasks
    threshold = np.percentile(curvature_values, 90)  # Top 10%
    high_curvature_tasks = [i for i, c in enumerate(curvature_values) if c > threshold]
    print(f"High-curvature tasks (top 10%): {len(high_curvature_tasks)} tasks")
    print(f"High-curvature threshold: {threshold:.6f}")
    
    return {
        'curvature_values': curvature_values,
        'lambda_values': lambda_values,
        'avg_curvature': avg_curvature,
        'curvature_max': curvature_max,
        'curvature_min': curvature_min,
        'curvature_std': curvature_std,
        'high_curvature_tasks': high_curvature_tasks,
        'threshold': threshold
    }

def plot_curvature_evolution(data, save_path=None):
    """Plot curvature evolution over tasks."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Expanded to show more statistics
    
    curvature_values = data['curvature_values']
    lambda_values = data['lambda_values']
    avg_curvature = data['avg_curvature']
    curvature_max = data.get('curvature_max', [])
    curvature_min = data.get('curvature_min', [])
    curvature_std = data.get('curvature_std', [])
    
    # Plot 1: Input curvature over tasks with error bars
    if curvature_max and curvature_min:
        # Plot mean with min/max range
        axes[0, 0].plot(curvature_values, 'b-', linewidth=2, label='Mean Curvature')
        axes[0, 0].fill_between(range(len(curvature_values)), curvature_min, curvature_max, 
                                alpha=0.3, color='blue', label='Min-Max Range')
    else:
        axes[0, 0].plot(curvature_values, 'b-', linewidth=2, label='Input Curvature')
        
    axes[0, 0].axhline(y=data['threshold'], color='r', linestyle='--', 
                       label=f'90th percentile ({data["threshold"]:.6f})')
    axes[0, 0].set_xlabel('Task')
    axes[0, 0].set_ylabel('Input Curvature')
    axes[0, 0].set_title('Input Curvature Evolution with Variation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Lambda values over tasks
    if lambda_values:
        axes[0, 1].plot(lambda_values, 'g-', linewidth=2, label='Lambda Values')
        axes[0, 1].set_xlabel('Task')
        axes[0, 1].set_ylabel('Protection Strength λ(x)')
        axes[0, 1].set_title('Dynamic Protection Strength')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Within-task curvature variation (std)
    if curvature_std:
        axes[0, 2].plot(curvature_std, 'purple', linewidth=2, label='Within-Task Std')
        axes[0, 2].set_xlabel('Task')
        axes[0, 2].set_ylabel('Curvature Standard Deviation')
        axes[0, 2].set_title('Within-Task Curvature Variation')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Curvature distribution
    axes[1, 0].hist(curvature_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(x=data['threshold'], color='r', linestyle='--', 
                       label=f'90th percentile')
    axes[1, 0].set_xlabel('Mean Input Curvature')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Curvature Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # Plot 5: Curvature vs Lambda correlation
    if lambda_values and len(lambda_values) == len(curvature_values):
        axes[1, 1].scatter(curvature_values, lambda_values, alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Input Curvature')
        axes[1, 1].set_ylabel('Protection Strength λ(x)')
        axes[1, 1].set_title('Curvature vs Protection Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(curvature_values, lambda_values)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=axes[1, 1].transAxes, fontsize=12,
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # Plot 6: Max vs Min curvature comparison
    if curvature_max and curvature_min:
        axes[1, 2].plot(curvature_max, 'r-', linewidth=2, label='Max Curvature', alpha=0.8)
        axes[1, 2].plot(curvature_min, 'b-', linewidth=2, label='Min Curvature', alpha=0.8)
        axes[1, 2].plot(curvature_values, 'g-', linewidth=2, label='Mean Curvature', alpha=0.8)
        axes[1, 2].set_xlabel('Task')
        axes[1, 2].set_ylabel('Curvature Value')
        axes[1, 2].set_title('Max/Mean/Min Curvature per Task')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig

def analyze_memorization_patterns(data, performance_data=None):
    """Analyze patterns that suggest memorization vs generalization."""
    curvature_values = data['curvature_values']
    lambda_values = data['lambda_values']
    high_curvature_tasks = data['high_curvature_tasks']
    
    print("\\n=== Memorization Analysis ===")
    
    # Curvature patterns
    curvature_trend = np.polyfit(range(len(curvature_values)), curvature_values, 1)[0]
    print(f"Curvature trend: {'Increasing' if curvature_trend > 0 else 'Decreasing'} "
          f"({curvature_trend:.8f} per task)")
    
    # Protection activation analysis
    if lambda_values:
        protection_active = sum(1 for l in lambda_values if l > 0.1)  # Threshold for "active"
        protection_rate = protection_active / len(lambda_values) * 100
        print(f"Protection activation rate: {protection_rate:.1f}% of tasks")
        
        avg_protection = np.mean(lambda_values)
        print(f"Average protection strength: {avg_protection:.6f}")
    
    # High-curvature task analysis
    if len(high_curvature_tasks) > 0:
        print(f"High-curvature tasks appear at: {high_curvature_tasks[:10]}...")  # First 10
        
        # Check if high-curvature tasks cluster at certain points
        if len(high_curvature_tasks) > 1:
            gaps = np.diff(high_curvature_tasks)
            avg_gap = np.mean(gaps)
            print(f"Average gap between high-curvature tasks: {avg_gap:.1f} tasks")
    
    # Performance correlation (if available)
    if performance_data and 'accuracies' in performance_data:
        accuracies = performance_data['accuracies']
        if len(accuracies) == len(curvature_values):
            # Correlation between curvature and performance
            acc_curv_corr = np.corrcoef(curvature_values, accuracies)[0, 1]
            print(f"Curvature-Accuracy correlation: {acc_curv_corr:.3f}")
            
            # Performance on high-curvature vs low-curvature tasks
            high_curv_acc = [accuracies[i] for i in high_curvature_tasks if i < len(accuracies)]
            low_curv_tasks = [i for i in range(len(curvature_values)) if i not in high_curvature_tasks]
            low_curv_acc = [accuracies[i] for i in low_curv_tasks if i < len(accuracies)]
            
            if high_curv_acc and low_curv_acc:
                print(f"High-curvature task accuracy: {np.mean(high_curv_acc):.3f} ± {np.std(high_curv_acc):.3f}")
                print(f"Low-curvature task accuracy: {np.mean(low_curv_acc):.3f} ± {np.std(low_curv_acc):.3f}")

def compare_with_baseline(input_aware_data, baseline_data=None):
    """Compare input-aware results with baseline (if available)."""
    if not baseline_data:
        print("\\nNo baseline data provided for comparison")
        return
    
    print("\\n=== Baseline Comparison ===")
    
    # Compare performance metrics
    if 'accuracies' in input_aware_data and 'accuracies' in baseline_data:
        ia_acc = np.mean(input_aware_data['accuracies'])
        baseline_acc = np.mean(baseline_data['accuracies'])
        improvement = ((ia_acc - baseline_acc) / baseline_acc) * 100
        print(f"Input-aware accuracy: {ia_acc:.3f}")
        print(f"Baseline accuracy: {baseline_acc:.3f}")
        print(f"Improvement: {improvement:+.1f}%")
    
    # Compare plasticity
    if 'plasticity_per_task' in input_aware_data and 'plasticity_per_task' in baseline_data:
        ia_plasticity = np.mean(input_aware_data['plasticity_per_task'])
        baseline_plasticity = np.mean(baseline_data['plasticity_per_task'])
        improvement = ((ia_plasticity - baseline_plasticity) / baseline_plasticity) * 100
        print(f"Input-aware plasticity: {ia_plasticity:.3f}")
        print(f"Baseline plasticity: {baseline_plasticity:.3f}")
        print(f"Improvement: {improvement:+.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze curvature tracking results')
    parser.add_argument('--log_dir', default='logs/', help='Directory containing log files')
    parser.add_argument('--experiment_name', default='input_aware', help='Experiment name pattern')
    parser.add_argument('--baseline_name', default=None, help='Baseline experiment name for comparison')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    # Find log files
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return
    
    # Look for input-aware experiment logs
    log_files = list(log_dir.glob(f"*{args.experiment_name}*.json"))
    if not log_files:
        print(f"No log files found matching pattern '*{args.experiment_name}*.json'")
        return
    
    print(f"Found {len(log_files)} log files:")
    for f in log_files:
        print(f"  {f}")
    
    # Load the most recent input-aware experiment
    log_file = max(log_files, key=lambda f: f.stat().st_mtime)
    print(f"\\nAnalyzing: {log_file}")
    
    data = load_curvature_data(log_file)
    if not data:
        return
    
    # Print experiment configuration
    print("\\n=== Experiment Configuration ===")
    print(f"Learner: {data.get('learner', 'unknown')}")
    print(f"Task: {data.get('task', 'unknown')}")
    print(f"Network: {data.get('network', 'unknown')}")
    print(f"Seed: {data.get('seed', 'unknown')}")
    print(f"Samples: {data.get('n_samples', 'unknown')}")
    
    if 'optimizer_hps' in data:
        hps = data['optimizer_hps']
        print(f"Curvature threshold: {hps.get('curvature_threshold', 'N/A')}")
        print(f"Lambda max: {hps.get('lambda_max', 'N/A')}")
        print(f"Hutchinson samples: {hps.get('hutchinson_samples', 'N/A')}")
        print(f"Compute frequency: {data.get('compute_curvature_every', 'N/A')}")
    
    # Analyze curvature data
    curvature_stats = analyze_curvature_statistics(data)
    if not curvature_stats:
        return
    
    # Create plots
    save_path = f"curvature_analysis_{args.experiment_name}.png" if args.save_plots else None
    plot_curvature_evolution(curvature_stats, save_path)
    
    # Analyze memorization patterns
    analyze_memorization_patterns(curvature_stats, data)
    
    # Compare with baseline if provided
    if args.baseline_name:
        baseline_files = list(log_dir.glob(f"*{args.baseline_name}*.json"))
        if baseline_files:
            baseline_file = max(baseline_files, key=lambda f: f.stat().st_mtime)
            baseline_data = load_curvature_data(baseline_file)
            compare_with_baseline(data, baseline_data)
    
    print(f"\\nAnalysis complete. Check the plots {'(saved)' if args.save_plots else '(displayed)'} for visual insights.")
    
    if not args.save_plots:
        plt.show()

if __name__ == "__main__":
    main()