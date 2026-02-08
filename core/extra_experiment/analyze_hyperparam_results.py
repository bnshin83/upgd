#!/usr/bin/env python3
"""
Analyze Hyperparameter Search Results from WandB

This script fetches results from WandB and finds the best hyperparameters
for CIFAR-10 and EMNIST datasets.

Usage:
    python analyze_hyperparam_results.py [--metric avg_accuracy] [--top 10]
"""

import argparse
import os
import sys

try:
    import wandb
    import pandas as pd
except ImportError:
    print("Installing required packages...")
    os.system("pip install wandb pandas tabulate")
    import wandb
    import pandas as pd

def fetch_runs(project="upgd-hyperparam-search", entity=None):
    """Fetch all runs from the WandB project."""
    api = wandb.Api()
    
    # Get project path
    if entity:
        path = f"{entity}/{project}"
    else:
        path = project
    
    print(f"Fetching runs from: {path}")
    runs = api.runs(path)
    
    data = []
    for run in runs:
        # Skip failed/crashed runs
        if run.state != "finished":
            continue
        
        config = run.config
        summary = run.summary._json_dict
        
        row = {
            'run_name': run.name,
            'dataset': config.get('task', '').replace('label_permuted_', '').replace('_stats', ''),
            'lr': config.get('lr', None),
            'sigma': config.get('sigma', None),
            'beta_utility': config.get('beta_utility', None),
            'weight_decay': config.get('weight_decay', None),
            'seed': config.get('seed', None),
            # Metrics
            'final_loss': summary.get('summary/final_loss', None),
            'avg_loss': summary.get('summary/avg_loss', None),
            'min_loss': summary.get('summary/min_loss', None),
            'final_accuracy': summary.get('summary/final_accuracy', None),
            'avg_accuracy': summary.get('summary/avg_accuracy', None),
            'max_accuracy': summary.get('summary/max_accuracy', None),
            'final_plasticity': summary.get('summary/final_plasticity', None),
            'avg_plasticity': summary.get('summary/avg_plasticity', None),
            'final_dead_units': summary.get('summary/final_dead_units', None),
        }
        data.append(row)
    
    return pd.DataFrame(data)

def analyze_results(df, metric='avg_accuracy', top_n=10):
    """Analyze and rank hyperparameter combinations."""
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 Hyperparameter Search Results Analysis")
    print(f"{'='*80}")
    print(f"Total runs: {len(df)}")
    print(f"Metric: {metric}")
    print(f"{'='*80}\n")
    
    # Group by hyperparameters and compute mean/std across seeds
    hp_cols = ['dataset', 'lr', 'sigma', 'beta_utility', 'weight_decay']
    
    for dataset in df['dataset'].unique():
        print(f"\n{'='*80}")
        print(f"📈 {dataset.upper()}")
        print(f"{'='*80}")
        
        df_dataset = df[df['dataset'] == dataset]
        
        if df_dataset.empty:
            print("No data for this dataset")
            continue
        
        # Aggregate across seeds
        agg_df = df_dataset.groupby(['lr', 'sigma', 'beta_utility', 'weight_decay']).agg({
            metric: ['mean', 'std', 'count'],
            'final_loss': ['mean', 'std'],
            'final_plasticity': ['mean'] if 'final_plasticity' in df_dataset.columns else [],
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                         for col in agg_df.columns]
        
        # Sort by mean metric (higher is better for accuracy, lower for loss)
        ascending = 'loss' in metric.lower()
        sort_col = f'{metric}_mean'
        
        if sort_col in agg_df.columns:
            agg_df = agg_df.sort_values(sort_col, ascending=ascending)
        
        # Print top N
        print(f"\nTop {top_n} configurations (by {metric}):\n")
        print("-" * 80)
        
        for i, row in agg_df.head(top_n).iterrows():
            print(f"Rank {agg_df.head(top_n).index.get_loc(i) + 1}:")
            print(f"  LR: {row['lr']}, Sigma: {row['sigma']}, "
                  f"Beta: {row['beta_utility']}, WD: {row['weight_decay']}")
            
            if f'{metric}_mean' in row:
                mean_val = row[f'{metric}_mean']
                std_val = row.get(f'{metric}_std', 0)
                count = row.get(f'{metric}_count', 0)
                print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f} (n={int(count)})")
            
            if 'final_loss_mean' in row:
                print(f"  Final Loss: {row['final_loss_mean']:.4f}")
            
            print()
        
        # Print best configuration
        print("-" * 80)
        best = agg_df.iloc[0]
        print(f"\n🏆 BEST for {dataset.upper()}:")
        print(f"   --lr {best['lr']} --sigma {best['sigma']} "
              f"--beta_utility {best['beta_utility']} --weight_decay {best['weight_decay']}")
        
    # Overall summary
    print(f"\n{'='*80}")
    print("📋 SUMMARY: Best Hyperparameters")
    print(f"{'='*80}\n")
    
    for dataset in df['dataset'].unique():
        df_dataset = df[df['dataset'] == dataset]
        if df_dataset.empty:
            continue
            
        agg_df = df_dataset.groupby(['lr', 'sigma', 'beta_utility', 'weight_decay']).agg({
            metric: 'mean'
        }).reset_index()
        
        ascending = 'loss' in metric.lower()
        best_idx = agg_df[metric].idxmin() if ascending else agg_df[metric].idxmax()
        best = agg_df.loc[best_idx]
        
        print(f"{dataset.upper()}:")
        print(f"  python3 core/run/run_stats_with_curvature.py \\")
        print(f"      --task label_permuted_{dataset}_stats \\")
        print(f"      --learner upgd_fo_global \\")
        print(f"      --lr {best['lr']} \\")
        print(f"      --sigma {best['sigma']} \\")
        print(f"      --beta_utility {best['beta_utility']} \\")
        print(f"      --weight_decay {best['weight_decay']} \\")
        print(f"      --network fully_connected_relu_with_hooks \\")
        print(f"      --n_samples 50000 --seed 0")
        print()

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('--metric', type=str, default='avg_accuracy',
                       choices=['avg_accuracy', 'final_accuracy', 'max_accuracy',
                               'avg_loss', 'final_loss', 'min_loss',
                               'avg_plasticity', 'final_plasticity'],
                       help='Metric to optimize (default: avg_accuracy)')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top configurations to show (default: 10)')
    parser.add_argument('--project', type=str, default='upgd-hyperparam-search',
                       help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='WandB entity (username/team)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Export results to CSV file')
    
    args = parser.parse_args()
    
    # Fetch data
    print("Connecting to WandB...")
    df = fetch_runs(project=args.project, entity=args.entity)
    
    if df.empty:
        print("No finished runs found in the project.")
        print("Make sure:")
        print("  1. The project name is correct")
        print("  2. Some experiments have completed")
        print("  3. You're logged into WandB (run: wandb login)")
        return
    
    # Analyze
    analyze_results(df, metric=args.metric, top_n=args.top)
    
    # Export if requested
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n💾 Results exported to: {args.csv}")

if __name__ == "__main__":
    main()

