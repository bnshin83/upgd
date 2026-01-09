#!/usr/bin/env python3
"""
Plot per-layer high-utility tail mass over training time from local JSON logs.

Primary plot:
  - layer tail mass = % of params in [0.52, 0.56) (hist_52_56_pct)
  - three panels (linear_1, linear_2, linear_3), lines = methods

Optional (if present in JSON):
  - per-layer raw utility max time series (layer_utility_max_per_step[layer]['raw_utility_max'])

Usage:
  python plot_layer_tail_over_time.py emnist
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Style (match other plotting scripts)
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({"font.size": 12})


SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent  # upgd_plots/
PROJECT_DIR = PLOTS_DIR.parent  # upgd/

# Reuse dataset + experiment path configs from the existing extractor
from extract_utility_histograms_local import DATASET_CONFIGS

DATASET_DISPLAY_NAMES = {
    "mini_imagenet": "Mini-ImageNet",
    "input_mnist": "Input-Permuted MNIST",
    "emnist": "Label-Permuted EMNIST",
    "cifar10": "Label-Permuted CIFAR-10",
}

LAYERS = ["linear_1", "linear_2", "linear_3"]

COLORS = {
    "S&P": "#7f7f7f",
    "UPGD (Full)": "#1f77b4",
    "UPGD (Output Only)": "#2ca02c",
    "UPGD (Hidden Only)": "#ff7f0e",
    "UPGD (Hidden+Output)": "#9467bd",
    "UPGD (Clamped 0.52)": "#d62728",
    "UPGD (Clamped 0.48-0.52)": "#8c564b",
    "UPGD (Clamped 0.44-0.56)": "#e377c2",
}


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _get_run_json_path(dataset: str, method_name: str) -> Path | None:
    cfg = DATASET_CONFIGS.get(dataset)
    if not cfg:
        return None
    logs_dir = PROJECT_DIR / "logs" / cfg["logs_subdir"]
    seed = cfg.get("seed", 2)
    exp_path = cfg.get("experiments", {}).get(method_name)
    if not exp_path:
        return None
    return logs_dir / exp_path / f"{seed}.json"


def plot_tail_mass(dataset: str, methods: list[str], log_scale: bool):
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    plot_dir = PLOTS_DIR / "figures" / dataset
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for layer_idx, layer in enumerate(LAYERS):
        ax = axes[layer_idx]
        for method in methods:
            json_path = _get_run_json_path(dataset, method)
            if not json_path or not json_path.exists():
                continue
            data = _load_json(json_path)
            layer_series = (data.get("layer_utility_histogram_per_step") or {}).get(layer, {})
            steps = layer_series.get("steps", [])
            tail = layer_series.get("hist_52_56_pct", [])
            if not steps or not tail:
                continue

            ax.plot(
                np.array(steps),
                np.array(tail),
                label=method,
                color=COLORS.get(method),
                linewidth=2.0,
                alpha=0.9,
            )

        ax.set_title(layer)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3, axis="y")
        if log_scale:
            ax.set_yscale("log")

    axes[0].set_ylabel("% params in [0.52, 0.56)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, framealpha=0.95, fontsize=11)

    suffix = "_log" if log_scale else ""
    fig.suptitle(f"Per-layer high-utility tail over time ({display_name})", fontsize=16, y=1.02)
    plt.tight_layout()
    out_path = plot_dir / f"layer_tail_over_time{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_raw_umax(dataset: str, methods: list[str], log_scale: bool):
    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    plot_dir = PLOTS_DIR / "figures" / dataset
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    any_data = False
    for layer_idx, layer in enumerate(LAYERS):
        ax = axes[layer_idx]
        for method in methods:
            json_path = _get_run_json_path(dataset, method)
            if not json_path or not json_path.exists():
                continue
            data = _load_json(json_path)
            layer_series = (data.get("layer_utility_max_per_step") or {}).get(layer, {})
            steps = layer_series.get("steps", [])
            raw_umax = layer_series.get("raw_utility_max", [])
            if not steps or not raw_umax:
                continue
            any_data = True
            ax.plot(
                np.array(steps),
                np.array(raw_umax),
                label=method,
                color=COLORS.get(method),
                linewidth=2.0,
                alpha=0.9,
            )

        ax.set_title(layer)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3, axis="y")
        if log_scale:
            ax.set_yscale("log")

    if not any_data:
        plt.close()
        print("No per-layer raw utility max time-series found in logs (layer_utility_max_per_step missing).")
        return

    axes[0].set_ylabel("Raw utility max (bias-corrected EMA)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, framealpha=0.95, fontsize=11)

    suffix = "_log" if log_scale else ""
    fig.suptitle(f"Per-layer raw utility max over time ({display_name})", fontsize=16, y=1.02)
    plt.tight_layout()
    out_path = plot_dir / f"layer_raw_umax_over_time{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "mini_imagenet"
    # Backwards/alias compatibility (common shorthand in this repo)
    dataset_aliases = {
        "imnist": "input_mnist",
    }
    dataset = dataset_aliases.get(dataset, dataset)
    cfg = DATASET_CONFIGS.get(dataset)
    if not cfg:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {sorted(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    methods = list(cfg.get("experiments", {}).keys())
    if not methods:
        print(f"No experiments configured for {dataset}")
        sys.exit(1)

    print(f"Plotting tail-over-time for {dataset}: {len(methods)} methods")
    plot_tail_mass(dataset, methods=methods, log_scale=False)
    plot_tail_mass(dataset, methods=methods, log_scale=True)
    plot_raw_umax(dataset, methods=methods, log_scale=False)
    plot_raw_umax(dataset, methods=methods, log_scale=True)


if __name__ == "__main__":
    main()


