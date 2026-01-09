#!/usr/bin/env python3
"""
Plot experimental results for input_permuted_mnist experiments.
This script reads JSON data files from the logs directory and creates visualizations.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import argparse
from typing import List, Dict, Any

# Set matplotlib parameters for publication-quality plots
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['figure.figsize'] = (10, 6)

def load_json_data(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_curves(data: Dict[str, Any], output_dir: str = '.'):
    """
    Plot training curves including loss and accuracy over time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    losses = data.get('losses', [])
    accuracies = data.get('accuracies', [])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot losses
    if losses:
        ax = axes[0]
        # Apply smoothing for better visualization
        window_size = min(100, len(losses) // 20)
        if window_size > 1:
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(len(losses)), losses, alpha=0.3, color='blue', label='Raw Loss')
            ax.plot(range(len(smoothed_losses)), smoothed_losses, color='blue', linewidth=2, label='Smoothed Loss')
        else:
            ax.plot(losses, color='blue', linewidth=2, label='Loss')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot accuracies
    if accuracies:
        ax = axes[1]
        # Apply smoothing for better visualization
        window_size = min(100, len(accuracies) // 20)
        if window_size > 1:
            smoothed_acc = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(len(accuracies)), accuracies, alpha=0.3, color='green', label='Raw Accuracy')
            ax.plot(range(len(smoothed_acc)), smoothed_acc, color='green', linewidth=2, label='Smoothed Accuracy')
        else:
            ax.plot(accuracies, color='green', linewidth=2, label='Accuracy')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy over Time')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'training_curves.pdf', bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()

def plot_task_performance(data: Dict[str, Any], output_dir: str = '.', tasks_per_plot: int = 5000):
    """
    Plot performance across different tasks/permutations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    losses = data.get('losses', [])
    accuracies = data.get('accuracies', [])
    
    if not losses and not accuracies:
        print("No data to plot for task performance")
        return
    
    # Calculate task-wise metrics
    n_samples = len(losses) if losses else len(accuracies)
    n_tasks = n_samples // tasks_per_plot
    
    if n_tasks == 0:
        print(f"Not enough samples for task-wise analysis (samples: {n_samples}, required: {tasks_per_plot})")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Task-wise average loss
    if losses:
        task_losses = []
        for i in range(n_tasks):
            start_idx = i * tasks_per_plot
            end_idx = min((i + 1) * tasks_per_plot, len(losses))
            task_losses.append(np.mean(losses[start_idx:end_idx]))
        
        ax = axes[0]
        ax.plot(range(1, n_tasks + 1), task_losses, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Task Number')
        ax.set_ylabel('Average Loss')
        ax.set_title('Average Loss per Task')
        ax.grid(True, alpha=0.3)
    
    # Task-wise average accuracy
    if accuracies:
        task_accuracies = []
        for i in range(n_tasks):
            start_idx = i * tasks_per_plot
            end_idx = min((i + 1) * tasks_per_plot, len(accuracies))
            task_accuracies.append(np.mean(accuracies[start_idx:end_idx]))
        
        ax = axes[1]
        ax.plot(range(1, n_tasks + 1), task_accuracies, marker='o', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Task Number')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Average Accuracy per Task')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'task_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'task_performance.pdf', bbox_inches='tight')
    print(f"Saved task performance to {output_path}")
    plt.close()

def plot_plasticity_analysis(data: Dict[str, Any], output_dir: str = '.', window_size: int = 1000):
    """
    Analyze and plot plasticity-related metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    losses = data.get('losses', [])
    if not losses:
        print("No loss data available for plasticity analysis")
        return
    
    # Calculate rolling statistics
    n_windows = len(losses) // window_size
    if n_windows < 2:
        print(f"Not enough data for plasticity analysis (need at least {2 * window_size} samples)")
        return
    
    rolling_mean = []
    rolling_std = []
    rolling_max = []
    rolling_min = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window_data = losses[start_idx:end_idx]
        rolling_mean.append(np.mean(window_data))
        rolling_std.append(np.std(window_data))
        rolling_max.append(np.max(window_data))
        rolling_min.append(np.min(window_data))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rolling mean
    ax = axes[0, 0]
    ax.plot(rolling_mean, linewidth=2)
    ax.set_xlabel(f'Window Number (size={window_size})')
    ax.set_ylabel('Mean Loss')
    ax.set_title('Rolling Mean Loss')
    ax.grid(True, alpha=0.3)
    
    # Rolling std
    ax = axes[0, 1]
    ax.plot(rolling_std, linewidth=2, color='orange')
    ax.set_xlabel(f'Window Number (size={window_size})')
    ax.set_ylabel('Std Dev of Loss')
    ax.set_title('Loss Variability (Plasticity Indicator)')
    ax.grid(True, alpha=0.3)
    
    # Range (max - min)
    ax = axes[1, 0]
    loss_range = np.array(rolling_max) - np.array(rolling_min)
    ax.plot(loss_range, linewidth=2, color='red')
    ax.set_xlabel(f'Window Number (size={window_size})')
    ax.set_ylabel('Loss Range')
    ax.set_title('Loss Range per Window')
    ax.grid(True, alpha=0.3)
    
    # Gradient magnitude (approximation)
    ax = axes[1, 1]
    gradient_approx = np.abs(np.diff(rolling_mean))
    ax.plot(gradient_approx, linewidth=2, color='purple')
    ax.set_xlabel(f'Window Transition')
    ax.set_ylabel('|Δ Mean Loss|')
    ax.set_title('Learning Rate Indicator')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'plasticity_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'plasticity_analysis.pdf', bbox_inches='tight')
    print(f"Saved plasticity analysis to {output_path}")
    plt.close()

def plot_summary_statistics(data: Dict[str, Any], output_dir: str = '.'):
    """
    Create a summary plot with key statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract all available metrics
    metrics = {}
    if 'losses' in data and data['losses']:
        metrics['Final Loss'] = data['losses'][-1]
        metrics['Min Loss'] = np.min(data['losses'])
        metrics['Mean Loss'] = np.mean(data['losses'])
    
    if 'accuracies' in data and data['accuracies']:
        metrics['Final Accuracy'] = data['accuracies'][-1]
        metrics['Max Accuracy'] = np.max(data['accuracies'])
        metrics['Mean Accuracy'] = np.mean(data['accuracies'])
    
    # Add other metrics if available
    for key in ['learner', 'lr', 'sigma', 'network']:
        if key in data:
            metrics[key.capitalize()] = data[key]
    
    # Create summary figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in metrics.items()]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        table[(i, 0)].set_facecolor('#E8E8E8')
        table[(i, 1)].set_facecolor('#F5F5F5')
    
    plt.title('Experiment Summary Statistics', fontsize=16, pad=20)
    output_path = output_dir / 'summary_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary statistics to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot input_permuted_mnist experimental results')
    parser.add_argument('--data-path', type=str, 
                       default='/scratch/gautschi/shin283/upgd/logs/input_permuted_mnist/pgd/fully_connected_relu/lr_0.01_sigma_0.001/0.json',
                       help='Path to JSON data file')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Directory to save plots')
    parser.add_argument('--tasks-per-plot', type=int, default=5000,
                       help='Number of samples per task for task-wise analysis')
    parser.add_argument('--window-size', type=int, default=1000,
                       help='Window size for rolling statistics')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {args.data_path}")
    
    # Load data
    try:
        data = load_json_data(args.data_path)
        print(f"Successfully loaded data with keys: {list(data.keys())}")
        
        # Generate all plots
        print("\nGenerating plots...")
        plot_training_curves(data, output_dir)
        plot_task_performance(data, output_dir, args.tasks_per_plot)
        plot_plasticity_analysis(data, output_dir, args.window_size)
        plot_summary_statistics(data, output_dir)
        
        print(f"\nAll plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error loading or plotting data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
