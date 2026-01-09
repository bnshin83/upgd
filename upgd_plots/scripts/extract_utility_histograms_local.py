#!/usr/bin/env python3
"""
Extract utility and raw utility histogram data from local WandB files.

Usage:
    python extract_utility_histograms_local.py [dataset]

Datasets: mini_imagenet, input_mnist, emnist, cifar10

Extracts:
1. Scaled utility histogram (9 bins in [0, 1])
2. Raw utility histogram (5 bins centered at 0)
3. Per-layer histograms if available
"""

import json
import sys
from pathlib import Path

# Directories (relative to this script's location)
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent  # upgd_plots/
PROJECT_DIR = PLOTS_DIR.parent  # upgd/
WANDB_DIR = PROJECT_DIR / "wandb"
OUTPUT_DIR = PLOTS_DIR / "data" / "utility_histograms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Scaled utility histogram bins (9 bins)
UTILITY_BIN_SUFFIXES = ['0_20', '20_40', '40_44', '44_48', '48_52', '52_56', '56_60', '60_80', '80_100']
UTILITY_BIN_LABELS = [
    '[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.44)', '[0.44, 0.48)', '[0.48, 0.52)',
    '[0.52, 0.56)', '[0.56, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]'
]

# Raw utility histogram bins (5 bins)
RAW_UTILITY_BINS = ['lt_m001', 'm001_m0002', 'm0002_p0002', 'p0002_p001', 'gt_p001']
RAW_UTILITY_BIN_LABELS = ['< -0.001', '[-0.001, -0.0002)', '[-0.0002, 0.0002]', '(0.0002, 0.001]', '> 0.001']

LAYERS = ['linear_1', 'linear_2', 'linear_3']

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================
# Each dataset has its own set of WandB run IDs
# Run IDs can be found in WandB dashboard or from wandb folder names

DATASET_CONFIGS = {
    'mini_imagenet': {
        'display_name': 'Mini-ImageNet',
        'use_json': True,  # Extract from experiment JSON files
        'logs_subdir': 'label_permuted_mini_imagenet_stats',
        'seed': 2,
        'experiments': {
            'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.9_weight_decay_0.001_n_samples_1000000',
            'UPGD (Full)': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
            'UPGD (Output Only)': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
            'UPGD (Hidden Only)': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
            'UPGD (Hidden+Output)': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
            'UPGD (Clamped 0.52)': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
            'UPGD (Clamped 0.48-0.52)': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
            'UPGD (Clamped 0.44-0.56)': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        }
    },
    'input_mnist': {
        'display_name': 'Input-Permuted MNIST',
        'use_json': True,  # Extract from experiment JSON files instead of WandB
        'logs_subdir': 'input_permuted_mnist_stats',
        'seed': 2,
        'experiments': {
            'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.001_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
            'UPGD (Full)': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
            'UPGD (Output Only)': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_output_only_n_samples_1000000',
            'UPGD (Hidden Only)': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_hidden_only_n_samples_1000000',
            'UPGD (Hidden+Output)': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_hidden_and_output_n_samples_1000000',
            'UPGD (Clamped 0.52)': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
            'UPGD (Clamped 0.48-0.52)': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
            'UPGD (Clamped 0.44-0.56)': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        }
    },
    'emnist': {
        'display_name': 'Label-Permuted EMNIST',
        'use_json': True,
        'logs_subdir': 'label_permuted_emnist_stats',
        'seed': 2,
        'experiments': {
            'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.9_weight_decay_0.001_n_samples_1000000',
            'UPGD (Full)': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
            'UPGD (Output Only)': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
            'UPGD (Hidden Only)': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
            'UPGD (Hidden+Output)': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
            'UPGD (Clamped 0.52)': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
            'UPGD (Clamped 0.48-0.52)': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
            'UPGD (Clamped 0.44-0.56)': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        }
    },
    'cifar10': {
        'display_name': 'Label-Permuted CIFAR-10',
        'use_json': True,
        'logs_subdir': 'label_permuted_cifar10_stats',
        'seed': 2,
        'experiments': {
            'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.999_weight_decay_0.001_n_samples_1000000',
            'UPGD (Full)': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_n_samples_1000000',
            'UPGD (Output Only)': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
            'UPGD (Hidden Only)': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
            'UPGD (Hidden+Output)': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
            'UPGD (Clamped 0.52)': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_n_samples_1000000',
            'UPGD (Clamped 0.48-0.52)': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
            'UPGD (Clamped 0.44-0.56)': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
        }
    },
}


def find_run_dir(run_id):
    """Find the wandb run directory for a given run ID."""
    for run_dir in WANDB_DIR.glob(f'run-*-{run_id}'):
        return run_dir
    return None


def read_summary_file(run_dir):
    """Read the wandb-summary.json file."""
    summary_file = run_dir / 'files' / 'wandb-summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return {}


def extract_histograms_from_json(json_data):
    """Extract utility histograms from experiment JSON file (final time point)."""
    result = {
        'utility': {'global': {}, 'layers': {layer: {} for layer in LAYERS}},
        'raw_utility': {'global': {}},
    }

    # Extract utility histogram data (final time point)
    if 'utility_histogram_per_step' in json_data:
        hist_data = json_data['utility_histogram_per_step']

        # Get final values (last element in each array)
        for i, suffix in enumerate(UTILITY_BIN_SUFFIXES):
            key = f'hist_{suffix}_pct'
            if key in hist_data and len(hist_data[key]) > 0:
                result['utility']['global'][UTILITY_BIN_LABELS[i]] = hist_data[key][-1]

        # Extract total_params if available
        if 'total_params' in hist_data:
            result['total_params'] = hist_data['total_params']

    # Extract per-layer utility histograms
    if 'layer_utility_histogram_per_step' in json_data:
        layer_data = json_data['layer_utility_histogram_per_step']

        for layer in LAYERS:
            if layer in layer_data:
                for i, suffix in enumerate(UTILITY_BIN_SUFFIXES):
                    key = f'hist_{suffix}_pct'
                    if key in layer_data[layer] and len(layer_data[layer][key]) > 0:
                        result['utility']['layers'][layer][UTILITY_BIN_LABELS[i]] = layer_data[layer][key][-1]

    return result


def extract_histograms(summary):
    """Extract utility and raw utility histograms from WandB summary."""
    result = {
        'utility': {'global': {}, 'layers': {layer: {} for layer in LAYERS}},
        'raw_utility': {'global': {}},
    }

    # === SCALED UTILITY HISTOGRAM ===
    # Try global first
    has_global_utility = False
    for i, suffix in enumerate(UTILITY_BIN_SUFFIXES):
        key = f'utility/hist_{suffix}_pct'
        if key in summary:
            result['utility']['global'][UTILITY_BIN_LABELS[i]] = summary[key]
            has_global_utility = True

    # Extract per-layer utility histograms
    layer_counts = {layer: {} for layer in LAYERS}
    layer_totals = {layer: 0 for layer in LAYERS}

    for layer in LAYERS:
        for i, suffix in enumerate(UTILITY_BIN_SUFFIXES):
            pct_key = f'layer/{layer}/hist_{suffix}_pct'
            count_key = f'layer/{layer}/hist_{suffix}'

            if pct_key in summary:
                result['utility']['layers'][layer][UTILITY_BIN_LABELS[i]] = summary[pct_key]

            if count_key in summary:
                layer_counts[layer][UTILITY_BIN_LABELS[i]] = summary[count_key]
                layer_totals[layer] += summary[count_key]

    # Compute global from layers if not available
    if not has_global_utility and any(layer_totals.values()):
        total_params = sum(layer_totals.values())
        for bin_label in UTILITY_BIN_LABELS:
            total_in_bin = sum(layer_counts[layer].get(bin_label, 0) for layer in LAYERS)
            result['utility']['global'][bin_label] = (total_in_bin / total_params) * 100 if total_params > 0 else 0

    # === RAW UTILITY HISTOGRAM ===
    for i, suffix in enumerate(RAW_UTILITY_BINS):
        key = f'raw_utility/hist_{suffix}_pct'
        if key in summary:
            result['raw_utility']['global'][RAW_UTILITY_BIN_LABELS[i]] = summary[key]

    # Also extract the full histogram if available (for detailed plotting)
    if 'histograms/raw_utility' in summary:
        hist_data = summary['histograms/raw_utility']
        if isinstance(hist_data, dict) and 'values' in hist_data and 'bins' in hist_data:
            result['raw_utility']['full_histogram'] = {
                'values': hist_data['values'],
                'bins': hist_data['bins']
            }

    # Extract 64-bin scaled utility histogram (clamped or unclamped)
    for hist_key in ['histograms/scaled_utility_clamped', 'histograms/scaled_utility_unclamped']:
        if hist_key in summary:
            hist_data = summary[hist_key]
            if isinstance(hist_data, dict) and 'values' in hist_data and 'bins' in hist_data:
                result['utility']['scaled_histogram'] = {
                    'values': hist_data['values'],
                    'bins': hist_data['bins'],
                    'source': hist_key
                }
                break  # prefer clamped if both exist

    # Extract total_params if available
    if 'utility/total_params' in summary:
        result['total_params'] = summary['utility/total_params']

    return result


def extract_for_dataset(dataset):
    """Extract utility histograms for a specific dataset."""
    if dataset not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        return

    config = DATASET_CONFIGS[dataset]
    display_name = config['display_name']

    print("=" * 70)
    print(f"Extracting Utility Histograms for {display_name}")
    print("=" * 70)

    all_data = {}

    # Check if using JSON extraction or WandB extraction
    if config.get('use_json', False):
        # Extract from experiment JSON files
        logs_dir = PROJECT_DIR / 'logs' / config['logs_subdir']
        seed = config.get('seed', 2)
        experiments = config.get('experiments', {})

        if not experiments:
            print(f"No experiments configured for {display_name}")
            return

        for exp_name, exp_path in experiments.items():
            print(f"\n{exp_name}")

            json_path = logs_dir / exp_path / f'{seed}.json'
            if not json_path.exists():
                print(f"  File not found: {json_path}")
                continue

            print(f"  Loading {json_path.name}...")
            with open(json_path) as f:
                json_data = json.load(f)

            hist_data = extract_histograms_from_json(json_data)
            all_data[exp_name] = hist_data

            # Print summary
            if hist_data['utility']['global']:
                print("  Scaled utility (global):")
                for label, val in hist_data['utility']['global'].items():
                    if val > 0.1:
                        print(f"    {label}: {val:.2f}%")

            if hist_data['utility']['layers']:
                has_layer_data = any(hist_data['utility']['layers'].get(layer) for layer in LAYERS)
                if has_layer_data:
                    print("  ✓ Per-layer data available")

    else:
        # Extract from WandB summary files (original behavior)
        target_runs = config.get('runs', {})

        if not target_runs:
            print(f"No run IDs configured for {display_name}")
            print("Please add run IDs to DATASET_CONFIGS in this script")
            return

        for exp_name, run_id in target_runs.items():
            print(f"\n{exp_name} (run: {run_id})")

            run_dir = find_run_dir(run_id)
            if not run_dir:
                print("  Run directory not found!")
                continue

            summary = read_summary_file(run_dir)
            if not summary:
                print("  No summary file found")
                continue

            hist_data = extract_histograms(summary)
            hist_data['run_id'] = run_id

            all_data[exp_name] = hist_data

            # Print summary
            if hist_data['utility']['global']:
                print("  Scaled utility (global):")
                for label, val in hist_data['utility']['global'].items():
                    if val > 0.1:
                        print(f"    {label}: {val:.2f}%")

            if hist_data['raw_utility']['global']:
                print("  Raw utility (global):")
                for label, val in hist_data['raw_utility']['global'].items():
                    if val > 0.1:
                        print(f"    {label}: {val:.2f}%")

    # Save to JSON
    output_file = OUTPUT_DIR / f"{dataset}_utility_histograms.json"
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to: {output_file}")

    # Print comparison tables
    if all_data:
        print("\n" + "=" * 100)
        print("SCALED UTILITY DISTRIBUTION (% of parameters)")
        print("=" * 100)
        header = f"{'Method':<25}"
        for label in UTILITY_BIN_LABELS:
            header += f"{label:>10}"
        print(header)
        print("-" * 100)

        for exp_name in all_data.keys():
            row = f"{exp_name:<25}"
            for label in UTILITY_BIN_LABELS:
                val = all_data[exp_name]['utility']['global'].get(label, 0)
                row += f"{val:>10.2f}"
            print(row)

        print("\n" + "=" * 80)
        print("RAW UTILITY DISTRIBUTION (% of parameters)")
        print("=" * 80)
        header = f"{'Method':<25}"
        for label in RAW_UTILITY_BIN_LABELS:
            header += f"{label:>12}"
        print(header)
        print("-" * 80)

        for exp_name in all_data.keys():
            row = f"{exp_name:<25}"
            for label in RAW_UTILITY_BIN_LABELS:
                val = all_data[exp_name]['raw_utility']['global'].get(label, 0)
                row += f"{val:>12.2f}"
            print(row)


def main():
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = 'mini_imagenet'  # default

    extract_for_dataset(dataset)


if __name__ == "__main__":
    main()
