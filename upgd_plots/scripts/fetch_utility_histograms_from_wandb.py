#!/usr/bin/env python3
"""
Fetch utility histogram data from WandB and save locally as JSON.

This fetches the 9-bin utility histogram percentages over training steps
for all target runs.
"""

import json
import wandb
from pathlib import Path

# Directories (relative to this script's location)
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent  # upgd_plots/
OUTPUT_DIR = PLOTS_DIR / "data" / "utility_histograms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# WandB project
PROJECT = 'shin283-purdue-university/upgd-input-aware'

# Target runs
TARGET_RUNS = {
    'UPGD (Full)': '2rorn0u1',
    'UPGD (Output Only)': 'lv4hrwao',
    'UPGD (Hidden Only)': 'ddyu1m95',
    'UPGD (Hidden+Output)': 'tval9wyc',
    'UPGD (Clamped 0.52)': 'ap0ll118',
    'S&P': '4q72hdev',
}

# Utility histogram keys (9 bins)
UTILITY_HIST_KEYS = [
    'utility/hist_0_20_pct',
    'utility/hist_20_40_pct',
    'utility/hist_40_44_pct',
    'utility/hist_44_48_pct',
    'utility/hist_48_52_pct',
    'utility/hist_52_56_pct',
    'utility/hist_56_60_pct',
    'utility/hist_60_80_pct',
    'utility/hist_80_100_pct',
    'utility/global_max',
]

# Per-layer keys (for all 3 layers)
LAYERS = ['linear_1', 'linear_2', 'linear_3']
LAYER_HIST_BINS = ['hist_0_20_pct', 'hist_20_40_pct', 'hist_40_44_pct', 'hist_44_48_pct',
                   'hist_48_52_pct', 'hist_52_56_pct', 'hist_56_60_pct', 'hist_60_80_pct', 'hist_80_100_pct']


def fetch_run_history(run_id, keys):
    """Fetch history for specific keys from a WandB run."""
    api = wandb.Api()
    run = api.run(f'{PROJECT}/{run_id}')
    print(f"  Run name: {run.name}")

    # Fetch history with specified keys
    history = run.scan_history(keys=keys + ['_step'])

    data = {'steps': []}
    for key in keys:
        data[key] = []

    for row in history:
        step = row.get('_step')
        if step is None:
            continue

        # Check if this row has utility data
        has_data = any(key in row and row[key] is not None for key in keys)
        if not has_data:
            continue

        data['steps'].append(step)
        for key in keys:
            data[key].append(row.get(key))

    return data


def main():
    print("=" * 70)
    print("Fetching Utility Histogram Data from WandB")
    print("=" * 70)

    all_data = {}

    for method, run_id in TARGET_RUNS.items():
        print(f"\n{method} (run: {run_id})")

        try:
            # Build list of keys to fetch
            keys_to_fetch = UTILITY_HIST_KEYS.copy()
            for layer in LAYERS:
                for bin_name in LAYER_HIST_BINS:
                    keys_to_fetch.append(f'layer/{layer}/{bin_name}')

            # Fetch history
            data = fetch_run_history(run_id, keys_to_fetch)

            print(f"  Fetched {len(data['steps'])} data points")

            if data['steps']:
                all_data[method] = {
                    'run_id': run_id,
                    'utility_histogram': {
                        'steps': data['steps'],
                        'hist_0_20_pct': data.get('utility/hist_0_20_pct', []),
                        'hist_20_40_pct': data.get('utility/hist_20_40_pct', []),
                        'hist_40_44_pct': data.get('utility/hist_40_44_pct', []),
                        'hist_44_48_pct': data.get('utility/hist_44_48_pct', []),
                        'hist_48_52_pct': data.get('utility/hist_48_52_pct', []),
                        'hist_52_56_pct': data.get('utility/hist_52_56_pct', []),
                        'hist_56_60_pct': data.get('utility/hist_56_60_pct', []),
                        'hist_60_80_pct': data.get('utility/hist_60_80_pct', []),
                        'hist_80_100_pct': data.get('utility/hist_80_100_pct', []),
                        'global_max': data.get('utility/global_max', []),
                    },
                    'layer_histogram': {}
                }

                # Add per-layer data
                for layer in LAYERS:
                    layer_data = {'steps': data['steps']}
                    for bin_name in LAYER_HIST_BINS:
                        key = f'layer/{layer}/{bin_name}'
                        layer_data[bin_name] = data.get(key, [])
                    all_data[method]['layer_histogram'][layer] = layer_data

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save to JSON
    output_file = OUTPUT_DIR / 'utility_histogram_timeseries.json'
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for method, data in all_data.items():
        n_points = len(data['utility_histogram']['steps'])
        print(f"  {method}: {n_points} data points")


if __name__ == "__main__":
    main()
