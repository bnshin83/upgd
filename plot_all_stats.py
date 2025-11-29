#!/usr/bin/env python3
"""
Plot experimental results for all statistics directories.
This script reads JSON data files from multiple statistics directories and creates comprehensive visualizations.
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

# Set matplotlib parameters for publication-quality plots
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['figure.figsize'] = (12, 8)

def find_all_json_files(base_paths: List[str]) -> Dict[str, List[str]]:
    """Find all JSON files in the specified directories."""
    all_files = {}
    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            print(f"Warning: Directory {base_path} not found")
            continue
            
        # Find all JSON files recursively
        json_files = glob.glob(os.path.join(base_path, "**/*.json"), recursive=True)
        
        if json_files:
            # Use the directory name as key
            dir_name = os.path.basename(base_path.rstrip('/'))
            all_files[dir_name] = json_files
            print(f"Found {len(json_files)} JSON files in {dir_name}")
        else:
            print(f"No JSON files found in {base_path}")
    
    return all_files

def load_json_data(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def extract_experiment_info(filepath: str) -> Dict[str, str]:
    """Extract experiment information from file path."""
    parts = Path(filepath).parts
    info = {
        'dataset': 'unknown',
        'learner': 'unknown', 
        'network': 'unknown',
        'hyperparams': 'unknown'
    }
    
    # Extract dataset from path
    for part in parts:
        if 'mnist' in part.lower():
            info['dataset'] = part
        elif 'cifar' in part.lower():
            info['dataset'] = part
        elif 'emnist' in part.lower():
            info['dataset'] = part
        elif 'imagenet' in part.lower():
            info['dataset'] = part
    
    # Extract learner type
    for part in parts:
        if 'pgd' in part.lower():
            info['learner'] = 'PGD'
        elif 'upgd' in part.lower():
            info['learner'] = 'UPGD'
        elif 'input_aware' in part.lower():
            info['learner'] = 'Input-Aware UPGD'
    
    # Extract network type
    for part in parts:
        if 'fully_connected' in part:
            info['network'] = 'Fully Connected'
        elif 'convolutional' in part:
            info['network'] = 'Convolutional'
    
    # Extract hyperparameters
    for part in parts:
        if part.startswith('lr_'):
            info['hyperparams'] = part
    
    return info

def plot_comparative_training_curves(all_data: Dict[str, Dict[str, Any]], output_dir: str):
    """Plot comparative training curves across all experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data)))
    
    for idx, (exp_name, data) in enumerate(all_data.items()):
        if not data or 'losses' not in data:
            continue
            
        color = colors[idx % len(colors)]
        losses = data['losses']
        accuracies = data.get('accuracies', [])
        
        # Plot losses
        ax = axes[0]
        if losses:
            # Apply smoothing
            window_size = min(100, len(losses) // 20)
            if window_size > 1:
                smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                ax.plot(smoothed, color=color, label=exp_name, linewidth=2)
            else:
                ax.plot(losses, color=color, label=exp_name, linewidth=2)
        
        # Plot accuracies
        ax = axes[1]
        if accuracies:
            window_size = min(100, len(accuracies) // 20)
            if window_size > 1:
                smoothed = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
                ax.plot(smoothed, color=color, label=exp_name, linewidth=2)
            else:
                ax.plot(accuracies, color=color, label=exp_name, linewidth=2)
    
    # Configure loss plot
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Configure accuracy plot
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy Comparison')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Final performance comparison
    ax = axes[2]
    final_losses = []
    final_accs = []
    exp_names = []
    
    for exp_name, data in all_data.items():
        if not data:
            continue
        if 'losses' in data and data['losses']:
            final_losses.append(data['losses'][-1])
            exp_names.append(exp_name)
        if 'accuracies' in data and data['accuracies']:
            final_accs.append(data['accuracies'][-1])
    
    if final_losses:
        bars = ax.bar(range(len(final_losses)), final_losses, color=colors[:len(final_losses)])
        ax.set_xlabel('Experiments')
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Comparison')
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    # Final accuracy comparison
    ax = axes[3]
    if final_accs:
        bars = ax.bar(range(len(final_accs)), final_accs, color=colors[:len(final_accs)])
        ax.set_xlabel('Experiments')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Final Accuracy Comparison')
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'comparative_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'comparative_analysis.pdf', bbox_inches='tight')
    print(f"Saved comparative analysis to {output_path}")
    plt.close()

def plot_individual_experiments(all_data: Dict[str, Dict[str, Any]], output_dir: str):
    """Create individual plots for each experiment."""
    output_dir = Path(output_dir)
    
    for exp_name, data in all_data.items():
        if not data:
            continue
            
        exp_dir = output_dir / exp_name.replace('/', '_')
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Training curves
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        losses = data.get('losses', [])
        accuracies = data.get('accuracies', [])
        
        # Plot losses
        if losses:
            ax = axes[0]
            window_size = min(100, len(losses) // 20)
            if window_size > 1:
                smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                ax.plot(range(len(losses)), losses, alpha=0.3, color='blue', label='Raw Loss')
                ax.plot(range(len(smoothed)), smoothed, color='blue', linewidth=2, label='Smoothed Loss')
            else:
                ax.plot(losses, color='blue', linewidth=2, label='Loss')
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.set_title(f'{exp_name} - Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot accuracies
        if accuracies:
            ax = axes[1]
            window_size = min(100, len(accuracies) // 20)
            if window_size > 1:
                smoothed = np.convolve(accuracies, np.ones(window_size)/window_size, mode='valid')
                ax.plot(range(len(accuracies)), accuracies, alpha=0.3, color='green', label='Raw Accuracy')
                ax.plot(range(len(smoothed)), smoothed, color='green', linewidth=2, label='Smoothed Accuracy')
            else:
                ax.plot(accuracies, color='green', linewidth=2, label='Accuracy')
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{exp_name} - Training Accuracy')
            ax.set_ylim([0, 1.05])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(exp_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.savefig(exp_dir / 'training_curves.pdf', bbox_inches='tight')
        plt.close()
        
        # Plasticity analysis
        if losses:
            plot_plasticity_analysis_individual(losses, exp_dir, exp_name)

def plot_plasticity_analysis_individual(losses: List[float], output_dir: Path, exp_name: str, window_size: int = 1000):
    """Create plasticity analysis for individual experiment."""
    if len(losses) < 2 * window_size:
        return
    
    n_windows = len(losses) // window_size
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
    axes[0, 0].plot(rolling_mean, linewidth=2)
    axes[0, 0].set_xlabel(f'Window Number (size={window_size})')
    axes[0, 0].set_ylabel('Mean Loss')
    axes[0, 0].set_title(f'{exp_name} - Rolling Mean Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rolling std
    axes[0, 1].plot(rolling_std, linewidth=2, color='orange')
    axes[0, 1].set_xlabel(f'Window Number (size={window_size})')
    axes[0, 1].set_ylabel('Std Dev of Loss')
    axes[0, 1].set_title(f'{exp_name} - Loss Variability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Range
    loss_range = np.array(rolling_max) - np.array(rolling_min)
    axes[1, 0].plot(loss_range, linewidth=2, color='red')
    axes[1, 0].set_xlabel(f'Window Number (size={window_size})')
    axes[1, 0].set_ylabel('Loss Range')
    axes[1, 0].set_title(f'{exp_name} - Loss Range')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient approximation
    if len(rolling_mean) > 1:
        gradient_approx = np.abs(np.diff(rolling_mean))
        axes[1, 1].plot(gradient_approx, linewidth=2, color='purple')
        axes[1, 1].set_xlabel(f'Window Transition')
        axes[1, 1].set_ylabel('|Δ Mean Loss|')
        axes[1, 1].set_title(f'{exp_name} - Learning Rate Indicator')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plasticity_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'plasticity_analysis.pdf', bbox_inches='tight')
    plt.close()

def create_summary_table(all_data: Dict[str, Dict[str, Any]], output_dir: str):
    """Create a summary table with key statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect statistics
    summary_data = []
    
    for exp_name, data in all_data.items():
        if not data:
            continue
            
        row = {'Experiment': exp_name}
        
        if 'losses' in data and data['losses']:
            losses = data['losses']
            row['Final Loss'] = f"{losses[-1]:.4f}"
            row['Min Loss'] = f"{np.min(losses):.4f}"
            row['Mean Loss'] = f"{np.mean(losses):.4f}"
            row['Loss Std'] = f"{np.std(losses):.4f}"
        
        if 'accuracies' in data and data['accuracies']:
            accuracies = data['accuracies']
            row['Final Accuracy'] = f"{accuracies[-1]:.4f}"
            row['Max Accuracy'] = f"{np.max(accuracies):.4f}"
            row['Mean Accuracy'] = f"{np.mean(accuracies):.4f}"
        
        # Add other available metrics
        for key in ['learner', 'lr', 'sigma', 'network']:
            if key in data:
                row[key.capitalize()] = str(data[key])
        
        summary_data.append(row)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(16, max(8, len(summary_data) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    if summary_data:
        # Get all unique columns
        all_columns = set()
        for row in summary_data:
            all_columns.update(row.keys())
        columns = ['Experiment'] + [col for col in sorted(all_columns) if col != 'Experiment']
        
        # Create table data
        table_data = []
        for row in summary_data:
            table_row = [row.get(col, 'N/A') for col in columns]
            table_data.append(table_row)
        
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(len(columns)):
                if i == 0:  # Header
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#F5F5F5' if i % 2 == 0 else '#FFFFFF')
    
    plt.title('Experiment Summary Statistics', fontsize=16, pad=20)
    output_path = output_dir / 'summary_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary table to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot statistics experiment results')
    parser.add_argument('--base-dirs', nargs='+', 
                       default=[
                           '/scratch/gautschi/shin283/upgd/logs/input_permuted_mnist_stats',
                           '/scratch/gautschi/shin283/upgd/logs/label_permuted_cifar10_stats',
                           '/scratch/gautschi/shin283/upgd/logs/label_permuted_emnist_stats',
                           '/scratch/gautschi/shin283/upgd/logs/label_permuted_mini_imagenet_stats'
                       ],
                       help='Base directories containing experiment logs')
    parser.add_argument('--json-file', type=str,
                       help='Single JSON file to plot (saves plots in same directory)')
    parser.add_argument('--output-dir', type=str, default='./plots_all_stats',
                       help='Directory to save plots (only used with --base-dirs)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching for JSON files in: {args.base_dirs}")
    
    # Find all JSON files
    all_json_files = find_all_json_files(args.base_dirs)
    
    if not all_json_files:
        print("No JSON files found!")
        return
    
    # Load all data
    all_data = {}
    
    for dir_name, json_files in all_json_files.items():
        for json_file in json_files:
            print(f"Loading: {json_file}")
            data = load_json_data(json_file)
            
            if data:
                # Create a meaningful name from the path
                rel_path = os.path.relpath(json_file, '/scratch/gautschi/shin283/upgd/logs/')
                exp_name = rel_path.replace('.json', '').replace('/', '_')
                all_data[exp_name] = data
                
                # Add experiment info
                exp_info = extract_experiment_info(json_file)
                all_data[exp_name].update(exp_info)
    
    if not all_data:
        print("No valid data found!")
        return
    
    print(f"\nLoaded {len(all_data)} experiments")
    for name in all_data.keys():
        print(f"  - {name}")
    
    # Generate plots
    print("\nGenerating comparative plots...")
    plot_comparative_training_curves(all_data, output_dir)
    
    print("Generating individual experiment plots...")
    plot_individual_experiments(all_data, output_dir)
    
    print("Creating summary table...")
    create_summary_table(all_data, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()