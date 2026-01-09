#!/usr/bin/env python3
"""
Fetch per-layer utility histogram data from WandB for baseline experiments.

This script uses the WandB API to extract per-layer utility histogram data
for SGD and UPGD (Full) experiments.

Usage:
    python fetch_layer_histograms_wandb.py

Requires: pip install wandb
"""

import wandb
import json
from pathlib import Path

# Output directory
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PLOTS_DIR / "data" / "utility_histograms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# WandB project info
WANDB_ENTITY = "shin283-purdue-university"
WANDB_PROJECT = "upgd"

# Target runs for Mini-ImageNet
# Format: display_name -> run_name or run_id
MINI_IMAGENET_RUNS = {
    "UPGD (Full)": "10093434_upgd_fo_global_mini_imagenet_seed_2_lr_0.01_sigma_0.001_beta_0.9_wd_0.0",
    "S&P": "10095043_snp_mini_imagenet_seed_2",
}

# Utility histogram bins (9 bins)
UTILITY_BIN_SUFFIXES = ['0_20', '20_40', '40_44', '44_48', '48_52', '52_56', '56_60', '60_80', '80_100']
UTILITY_BIN_LABELS = [
    '[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.44)', '[0.44, 0.48)', '[0.48, 0.52)',
    '[0.52, 0.56)', '[0.56, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]'
]

LAYERS = ['linear_1', 'linear_2', 'linear_3']


def find_run_by_name(api, run_name):
    """Find a run by its name in the project."""
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", 
                    filters={"display_name": run_name})
    for run in runs:
        return run
    return None


def extract_layer_histograms_from_run(run):
    """Extract per-layer utility histogram from a WandB run."""
    result = {
        'utility': {
            'global': {},
            'layers': {layer: {} for layer in LAYERS}
        },
        'raw_utility': {'global': {}},
        'run_id': run.id,
        'run_name': run.name,
    }
    
    # Get summary data (final values)
    summary = run.summary
    
    # Global utility histogram
    for i, suffix in enumerate(UTILITY_BIN_SUFFIXES):
        key = f'utility/hist_{suffix}_pct'
        if key in summary:
            result['utility']['global'][UTILITY_BIN_LABELS[i]] = summary[key]
    
    # Per-layer utility histograms
    for layer in LAYERS:
        for i, suffix in enumerate(UTILITY_BIN_SUFFIXES):
            key = f'layer/{layer}/hist_{suffix}_pct'
            if key in summary:
                result['utility']['layers'][layer][UTILITY_BIN_LABELS[i]] = summary[key]
    
    # Raw utility histogram
    raw_bins = ['lt_m001', 'm001_m0002', 'm0002_p0002', 'p0002_p001', 'gt_p001']
    raw_labels = ['< -0.001', '[-0.001, -0.0002)', '[-0.0002, 0.0002]', '(0.0002, 0.001]', '> 0.001']
    for suffix, label in zip(raw_bins, raw_labels):
        key = f'raw_utility/hist_{suffix}_pct'
        if key in summary:
            result['raw_utility']['global'][label] = summary[key]
    
    return result


def main():
    print("=" * 70)
    print("Fetching Per-Layer Utility Histograms from WandB")
    print("=" * 70)
    
    # Initialize WandB API
    api = wandb.Api()
    
    all_data = {}
    
    for exp_name, run_name in MINI_IMAGENET_RUNS.items():
        print(f"\n{exp_name}")
        print(f"  Looking for run: {run_name}")
        
        run = find_run_by_name(api, run_name)
        if not run:
            print(f"  ✗ Run not found!")
            continue
        
        print(f"  ✓ Found run: {run.id}")
        
        # Extract histogram data
        hist_data = extract_layer_histograms_from_run(run)
        all_data[exp_name] = hist_data
        
        # Print summary
        print(f"  Global utility histogram:")
        for bin_label, val in hist_data['utility']['global'].items():
            if val > 0.01:
                print(f"    {bin_label}: {val:.4f}%")
        
        print(f"  Per-layer data:")
        for layer in LAYERS:
            layer_data = hist_data['utility']['layers'][layer]
            if layer_data:
                # Show the dominant bin
                max_bin = max(layer_data, key=layer_data.get)
                print(f"    {layer}: {max_bin} = {layer_data[max_bin]:.2f}%")
            else:
                print(f"    {layer}: No data")
    
    if all_data:
        # Load existing data and merge
        output_file = OUTPUT_DIR / "mini_imagenet_utility_histograms.json"
        existing_data = {}
        if output_file.exists():
            with open(output_file) as f:
                existing_data = json.load(f)
        
        # Update with new per-layer data
        for exp_name, new_data in all_data.items():
            if exp_name in existing_data:
                # Merge: keep existing, update layers if empty
                if not existing_data[exp_name].get('utility', {}).get('layers', {}).get('linear_1'):
                    existing_data[exp_name]['utility']['layers'] = new_data['utility']['layers']
            else:
                existing_data[exp_name] = new_data
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        print(f"\nUpdated: {output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
