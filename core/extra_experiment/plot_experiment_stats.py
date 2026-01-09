#!/usr/bin/env python3
"""
Plot script for experiment statistics results.
This script generates various diagnostic plots from any statistics JSON files
(works for MNIST, EMNIST, CIFAR-10, Mini-ImageNet, etc.)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Set matplotlib styling
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams.update({'font.size': 12})

def plot_stats_from_json(json_file_path, output_dir=None):
    """
    Create plots from a single statistics JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        output_dir (str): Directory to save plots (if None, saves in same directory as JSON file)
    """
    
    # If no output directory specified, use the same directory as the JSON file
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract basic info
    learner = data.get('learner', 'unknown')
    network = data.get('network', 'unknown')
    lr = data.get('optimizer_hps', {}).get('lr', 'unknown')
    sigma = data.get('optimizer_hps', {}).get('sigma', 'unknown')
    
    title_suffix = f"{learner} - lr={lr}, σ={sigma}"
    
    # Calculate number of tasks based on data length
    n_samples = len(data['losses'])
    n_samples_per_task = data.get('n_samples', 1000000) // 200  # 200 tasks typical for input permuted MNIST
    if n_samples_per_task == 0:
        n_samples_per_task = n_samples // 200
    
    task_indices = np.arange(len(data['losses']))
    task_numbers = task_indices / (n_samples_per_task // 200) if n_samples_per_task > 0 else task_indices
    
    # 1. Plot Learning Curves (Loss and Accuracy)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Loss
    ax1.plot(task_numbers, data['losses'], 'b-', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Learning Curves - {title_suffix}')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(task_numbers, data['accuracies'], 'r-', alpha=0.7, linewidth=1)
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Task Number')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Plot Plasticity
    if 'plasticity_per_task' in data:
        plt.figure(figsize=(10, 6))
        task_nums = np.arange(len(data['plasticity_per_task']))
        plt.plot(task_nums, data['plasticity_per_task'], 'g-', linewidth=2)
        plt.ylabel('Plasticity')
        plt.xlabel('Task Number')
        plt.title(f'Plasticity over Tasks - {title_suffix}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plasticity.pdf', bbox_inches='tight', dpi=300)
        plt.close()
    
    # 3. Plot Dead Units
    if 'n_dead_units_per_task' in data:
        plt.figure(figsize=(10, 6))
        task_nums = np.arange(len(data['n_dead_units_per_task']))
        plt.plot(task_nums, data['n_dead_units_per_task'], 'orange', linewidth=2)
        plt.ylabel('Fraction of Dead Units')
        plt.xlabel('Task Number')
        plt.title(f'Dead Units over Tasks - {title_suffix}')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dead_units.pdf', bbox_inches='tight', dpi=300)
        plt.close()
    
    # 4. Plot Weight Statistics
    if 'weight_l2_per_task' in data and 'weight_l1_per_task' in data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        task_nums = np.arange(len(data['weight_l2_per_task']))
        
        # L2 norm
        ax1.plot(task_nums, data['weight_l2_per_task'], 'purple', linewidth=2)
        ax1.set_ylabel('Weight L2 Norm')
        ax1.set_title(f'Weight Norms over Tasks - {title_suffix}')
        ax1.grid(True, alpha=0.3)
        
        # L1 norm
        ax2.plot(task_nums, data['weight_l1_per_task'], 'brown', linewidth=2)
        ax2.set_ylabel('Weight L1 Norm')
        ax2.set_xlabel('Task Number')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/weight_norms.pdf', bbox_inches='tight', dpi=300)
        plt.close()
    
    # 5. Plot Gradient Statistics
    if 'grad_l2_per_task' in data and 'grad_l1_per_task' in data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        task_nums = np.arange(len(data['grad_l2_per_task']))
        
        # Gradient L2 norm
        ax1.plot(task_nums, data['grad_l2_per_task'], 'cyan', linewidth=2)
        ax1.set_ylabel('Gradient L2 Norm')
        ax1.set_title(f'Gradient Norms over Tasks - {title_suffix}')
        ax1.grid(True, alpha=0.3)
        
        # Gradient L1 norm
        ax2.plot(task_nums, data['grad_l1_per_task'], 'magenta', linewidth=2)
        ax2.set_ylabel('Gradient L1 Norm')
        ax2.set_xlabel('Task Number')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gradient_norms.pdf', bbox_inches='tight', dpi=300)
        plt.close()
    
    # 6. Plot Gradient Sparsity (L0 norm)
    if 'grad_l0_per_task' in data:
        plt.figure(figsize=(10, 6))
        task_nums = np.arange(len(data['grad_l0_per_task']))
        plt.plot(task_nums, data['grad_l0_per_task'], 'red', linewidth=2)
        plt.ylabel('Gradient Sparsity (L0 norm)')
        plt.xlabel('Task Number')
        plt.title(f'Gradient Sparsity over Tasks - {title_suffix}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gradient_sparsity.pdf', bbox_inches='tight', dpi=300)
        plt.close()
    
    # 7. Summary plot with key metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    ax1.plot(task_numbers, data['accuracies'], 'r-', linewidth=2)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Plasticity
    if 'plasticity_per_task' in data:
        task_nums = np.arange(len(data['plasticity_per_task']))
        ax2.plot(task_nums, data['plasticity_per_task'], 'g-', linewidth=2)
        ax2.set_ylabel('Plasticity')
        ax2.set_title('Plasticity')
        ax2.grid(True, alpha=0.3)
    
    # Dead Units
    if 'n_dead_units_per_task' in data:
        task_nums = np.arange(len(data['n_dead_units_per_task']))
        ax3.plot(task_nums, data['n_dead_units_per_task'], 'orange', linewidth=2)
        ax3.set_ylabel('Fraction of Dead Units')
        ax3.set_xlabel('Task Number')
        ax3.set_title('Dead Units')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # Weight L2 norm
    if 'weight_l2_per_task' in data:
        task_nums = np.arange(len(data['weight_l2_per_task']))
        ax4.plot(task_nums, data['weight_l2_per_task'], 'purple', linewidth=2)
        ax4.set_ylabel('Weight L2 Norm')
        ax4.set_xlabel('Task Number')
        ax4.set_title('Weight L2 Norm')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'UPGD Statistics Summary - {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"All plots saved to {output_dir}/")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Plot experiment statistics results (MNIST, EMNIST, CIFAR-10, etc.)')
    parser.add_argument('--json-file', help='Path to the JSON statistics file (optional)')
    parser.add_argument('--output-dir', help='Output directory for plots (optional - defaults to same directory as JSON file)')
    
    args = parser.parse_args()
    
    # If no file provided, ask for it interactively
    if not args.json_file:
        print("Enter the path to the JSON statistics file:")
        json_file = input().strip()
        # Remove quotes if user copied path with quotes
        json_file = json_file.strip('"').strip("'")
    else:
        json_file = args.json_file
    
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} not found")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        print(f"Processing: {json_file}")
        print(f"Output directory: {output_dir}")
    else:
        output_dir = None
        print(f"Processing: {json_file}")
        print(f"Output directory: Same as JSON file location")
    
    plot_stats_from_json(json_file, output_dir)

if __name__ == "__main__":
    main()