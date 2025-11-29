#!/usr/bin/env python3
"""
Plot input curvature vs steps with task boundaries for continual learning experiments.

Usage:
    python plot_curvature.py --log_file <path_to_log> --task_freq <steps_per_task> --output <output_file>
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


def plot_curvature(steps, curvatures, task_boundaries, task_freq, title, output_path):
    """Plot curvature with task boundaries."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot curvature as dots
    ax.scatter(steps, curvatures, c='blue', alpha=0.6, s=1, label='Input Curvature')
    
    # Add task boundaries
    max_step = steps.max() if len(steps) > 0 else 0
    for task_num in task_boundaries:
        boundary_step = (task_num - 1) * task_freq
        if boundary_step <= max_step:
            ax.axvline(x=boundary_step, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Create custom tick labels with task numbers
    # Get current x-ticks
    current_ticks = ax.get_xticks()
    tick_labels = []
    
    for tick in current_ticks:
        # Find which task this tick belongs to
        task_num = int(tick // task_freq) + 1
        if tick >= 0 and task_num <= len(task_boundaries):
            tick_labels.append(f'{int(tick)}\n(Task {task_num})')
        else:
            tick_labels.append(f'{int(tick)}')
    
    ax.set_xticks(current_ticks)
    ax.set_xticklabels(tick_labels, fontsize=8)
    
    # Formatting
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Input Curvature')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Log scale for y-axis if needed (curvature values can be very small)
    if curvatures.min() > 0:
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    print(f"Total steps: {len(steps)}")
    print(f"Total tasks: {len(task_boundaries)}")
    print(f"Curvature range: {curvatures.min():.6f} - {curvatures.max():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Plot curvature vs steps with task boundaries')
    parser.add_argument('--log_file', required=True, help='Path to log file')
    parser.add_argument('--task_freq', type=int, required=True, 
                       help='Steps per task (2500 for EMNIST, 5000 for Input MNIST)')
    parser.add_argument('--output', help='Output plot file (default: auto-generated)')
    parser.add_argument('--title', help='Plot title (default: auto-generated)')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    # Parse log data
    steps, curvatures, task_boundaries = parse_log_file(log_path)
    
    if len(steps) == 0:
        raise ValueError("No curvature data found in log file")
    
    # Generate output filename if not provided
    if args.output is None:
        output_path = log_path.parent / f"{log_path.stem}_curvature_plot.png"
    else:
        output_path = Path(args.output)
    
    # Generate title if not provided
    if args.title is None:
        dataset = "EMNIST" if args.task_freq == 2500 else "Input MNIST" if args.task_freq == 5000 else "Unknown"
        title = f"Input Curvature vs Steps - {dataset} (Task freq: {args.task_freq})"
    else:
        title = args.title
    
    # Create plot
    plot_curvature(steps, curvatures, task_boundaries, args.task_freq, title, output_path)


if __name__ == "__main__":
    main()