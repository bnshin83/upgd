#!/usr/bin/env python3
"""
Plot and compare input curvature vs steps for multiple continual learning experiments.

Usage:
    python plot_curvature_comparison.py --emnist_log <path> --mnist_log <path> --output <output_file>
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(log_path):
    """Parse curvature data from log file."""
    steps = []
    curvatures = []
    task_boundaries = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse step data: "Step X: Input curvature = Y.YYYYYY, Lambda = Z.ZZZZZZ"
            step_match = re.match(r'Step (\d+): Input curvature = ([\d.]+)', line.strip())
            if step_match:
                step = int(step_match.group(1))
                curvature = float(step_match.group(2))
                steps.append(step)
                curvatures.append(curvature)
            
            # Parse task boundaries: "Task N: Curvature stats - Mean: X, Max: Y, Min: Z, Std: W"
            task_match = re.match(r'Task (\d+): Curvature stats', line.strip())
            if task_match:
                task_num = int(task_match.group(1))
                task_boundaries.append(task_num)
    
    return np.array(steps), np.array(curvatures), task_boundaries


def plot_comparison(emnist_data, mnist_data, output_path):
    """Plot comparison of curvature between EMNIST and Input MNIST."""
    
    emnist_steps, emnist_curvatures, emnist_tasks = emnist_data
    mnist_steps, mnist_curvatures, mnist_tasks = mnist_data
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # EMNIST plot (top)
    ax1.scatter(emnist_steps, emnist_curvatures, c='blue', alpha=0.6, s=1, label='Input Curvature')
    
    # Add task boundaries for EMNIST (every 2500 steps)
    task_freq_emnist = 2500
    max_step_emnist = emnist_steps.max() if len(emnist_steps) > 0 else 0
    for task_num in emnist_tasks[:10]:  # Show first 10 boundaries
        boundary_step = (task_num - 1) * task_freq_emnist
        if boundary_step <= max_step_emnist:
            ax1.axvline(x=boundary_step, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Create custom tick labels for EMNIST
    current_ticks_1 = ax1.get_xticks()
    tick_labels_1 = []
    for tick in current_ticks_1:
        task_num = int(tick // task_freq_emnist) + 1
        if tick >= 0 and task_num <= len(emnist_tasks):
            tick_labels_1.append(f'{int(tick)}\n(Task {task_num})')
        else:
            tick_labels_1.append(f'{int(tick)}')
    ax1.set_xticks(current_ticks_1)
    ax1.set_xticklabels(tick_labels_1, fontsize=8)
    
    ax1.set_ylabel('Input Curvature')
    ax1.set_title('EMNIST - Input Curvature vs Steps (Task every 2500 steps)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if emnist_curvatures.min() > 0:
        ax1.set_yscale('log')
    
    # Input MNIST plot (bottom)
    ax2.scatter(mnist_steps, mnist_curvatures, c='green', alpha=0.6, s=1, label='Input Curvature')
    
    # Add task boundaries for Input MNIST (every 5000 steps)
    task_freq_mnist = 5000
    max_step_mnist = mnist_steps.max() if len(mnist_steps) > 0 else 0
    for task_num in mnist_tasks[:10]:  # Show first 10 boundaries
        boundary_step = (task_num - 1) * task_freq_mnist
        if boundary_step <= max_step_mnist:
            ax2.axvline(x=boundary_step, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Create custom tick labels for Input MNIST
    current_ticks_2 = ax2.get_xticks()
    tick_labels_2 = []
    for tick in current_ticks_2:
        task_num = int(tick // task_freq_mnist) + 1
        if tick >= 0 and task_num <= len(mnist_tasks):
            tick_labels_2.append(f'{int(tick)}\n(Task {task_num})')
        else:
            tick_labels_2.append(f'{int(tick)}')
    ax2.set_xticks(current_ticks_2)
    ax2.set_xticklabels(tick_labels_2, fontsize=8)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Input Curvature')
    ax2.set_title('Input MNIST - Input Curvature vs Steps (Task every 5000 steps)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    if mnist_curvatures.min() > 0:
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")
    print(f"EMNIST - Steps: {len(emnist_steps)}, Tasks: {len(emnist_tasks)}, Curvature range: {emnist_curvatures.min():.6f} - {emnist_curvatures.max():.6f}")
    print(f"Input MNIST - Steps: {len(mnist_steps)}, Tasks: {len(mnist_tasks)}, Curvature range: {mnist_curvatures.min():.6f} - {mnist_curvatures.max():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Compare curvature plots for EMNIST and Input MNIST')
    parser.add_argument('--emnist_log', required=True, help='Path to EMNIST log file')
    parser.add_argument('--mnist_log', required=True, help='Path to Input MNIST log file')
    parser.add_argument('--output', default='curvature_comparison.png', help='Output plot file')
    
    args = parser.parse_args()
    
    emnist_path = Path(args.emnist_log)
    mnist_path = Path(args.mnist_log)
    
    if not emnist_path.exists():
        raise FileNotFoundError(f"EMNIST log file not found: {emnist_path}")
    if not mnist_path.exists():
        raise FileNotFoundError(f"Input MNIST log file not found: {mnist_path}")
    
    # Parse log data
    emnist_data = parse_log_file(emnist_path)
    mnist_data = parse_log_file(mnist_path)
    
    if len(emnist_data[0]) == 0:
        raise ValueError("No curvature data found in EMNIST log file")
    if len(mnist_data[0]) == 0:
        raise ValueError("No curvature data found in Input MNIST log file")
    
    output_path = Path(args.output)
    
    # Create comparison plot
    plot_comparison(emnist_data, mnist_data, output_path)


if __name__ == "__main__":
    main()