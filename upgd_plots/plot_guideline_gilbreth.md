# UPGD Plotting Guide - Gilbreth Cluster

## Summary

This document tracks all plotting work done on the Gilbreth cluster for UPGD experiments, including Input-MNIST and Mini-ImageNet datasets with seed 2.

**Date:** December 18, 2025
**Location:** `/scratch/gilbreth/shin283/upgd/upgd_plots/`
**Datasets Completed:** Input-MNIST, Mini-ImageNet

---

## 1. Folder Structure

```
/scratch/gilbreth/shin283/upgd_plots/
    scripts/
        plot_training_metrics.py           # Multi-dataset training metrics (UPDATED)
        extract_utility_histograms_local.py # Extract from JSON files (UPDATED)
        plot_utility_histograms.py         # Plot utility histograms (FIXED)
        plot_mini_imagenet.py              # Legacy script (not updated)
        fetch_utility_histograms_from_wandb.py # WandB API fetcher (not used)
    data/
        utility_histograms/
            input_mnist_utility_histograms.json   # Extracted from JSON
            mini_imagenet_utility_histograms.json # Extracted from JSON
    figures/
        input_mnist/                       # 21 plots (3.3MB)
            accuracy_comparison.png
            loss_comparison.png
            plasticity_comparison.png
            dead_units_comparison.png
            weight_l2_comparison.png
            weight_l1_comparison.png
            grad_l2_comparison.png
            utility_histogram.png          # 9-bin global
            utility_histogram_log.png      # 9-bin global (log scale)
            raw_utility_histogram.png      # 5-bin raw utility
            raw_utility_histogram_log.png  # 5-bin raw utility (log)
            layer_utility_*.png            # 10 per-layer plots (5 methods ◊ 2 scales)
        mini_imagenet/                     # 21 plots (4.0MB)
            [same structure as input_mnist]
    plot_guideline_gilbreth.md             # This file
```

---

## 2. Experiments Plotted

### Input-MNIST (Seed 2)

| Method | Task ID | Accuracy | Plasticity | Status |
|--------|---------|----------|------------|--------|
| S&P | 10093423 | 0.7814 | 0.3066 |  Complete |
| UPGD (Full) | 10090773 | 0.7797 | 0.4966 |  Complete |
| UPGD (Output Only) | 10090770 | 0.7796 | 0.4965 |  Complete |
| UPGD (Hidden Only) | 10090771 | 0.7799 | 0.4987 |  Complete |
| UPGD (Hidden+Output) | 10090772 | 0.7807 | 0.4966 |  Complete |
| UPGD (Clamped 0.52) | 10090774 | 0.7794 | 0.4976 |  Complete |
| UPGD (Clamped 0.48-0.52) | 10090775 | 0.7797 | 0.4980 |  Complete |
| UPGD (Clamped 0.44-0.56) | 10093422 | 0.7801 | 0.4972 |  Complete |

**Key Finding:** S&P achieves highest accuracy but UPGD maintains 63% better plasticity.

### Mini-ImageNet (Seed 2)

| Method | Task ID | Accuracy | Plasticity | Status |
|--------|---------|----------|------------|--------|
| S&P | 10095043 | 0.2884 | 0.4236 |  Complete |
| UPGD (Full) | 10093434 | 0.5474 | 0.5505 |  Complete |
| UPGD (Output Only) | 10093424 | **0.5901** | 0.5370 |  Complete |
| UPGD (Hidden Only) | 10093428 | 0.2198 | 0.2154 |  Complete |
| UPGD (Hidden+Output) | 10093433 | 0.5443 | 0.5523 |  Complete |
| UPGD (Clamped 0.52) | 10095040 | 0.4118 | 0.4260 |  Complete |
| UPGD (Clamped 0.48-0.52) | 10095041 | 0.3106 | 0.3440 |  Complete |
| UPGD (Clamped 0.44-0.56) | 10095042 | 0.3739 | 0.3974 |  Complete |

**Key Finding:** UPGD (Output Only) achieves best accuracy (0.5901), 104% higher than S&P.

---

## 3. Path Fixes Applied

### Problem
The `plot_training_metrics.py` script had incorrect paths for gated UPGD variants. The script assumed simple paths without gating mode parameters, but actual experiment directories include these parameters.

### Fixes Applied

#### Input-MNIST Path Updates

**Before:**
```python
'UPGD (Output Only)': {
    'path': '.../lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
```

**After:**
```python
'UPGD (Output Only)': {
    'path': '.../lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_output_only_n_samples_1000000',
```

Similar fixes for:
- `UPGD (Hidden Only)`: Added `gating_mode_hidden_only`
- `UPGD (Hidden+Output)`: Added `gating_mode_hidden_and_output`
- `UPGD (Clamped 0.48-0.52)`: Added `min_clamp_0.48_max_clamp_0.52`
- `UPGD (Clamped 0.44-0.56)`: Added `min_clamp_0.44_max_clamp_0.56`

#### Mini-ImageNet Path Updates

Same pattern of fixes applied. Paths now correctly include:
- `gating_mode_output_only`
- `gating_mode_hidden_only`
- `gating_mode_hidden_and_output`
- `min_clamp_0.48_max_clamp_0.52`
- `min_clamp_0.44_max_clamp_0.56`

#### Matplotlib Style Fix

**Before:**
```python
plt.style.use('seaborn-v0_8-whitegrid')  # Not available in environment
```

**After:**
```python
plt.style.use('seaborn-whitegrid')  # Available style
```

Applied to:
- `plot_training_metrics.py`
- `plot_utility_histograms.py`

---

## 4. Utility Histogram Extraction

### Method: JSON File Extraction (Not WandB)

The original `extract_utility_histograms_local.py` was designed to read from WandB summary files. We modified it to support direct extraction from experiment JSON files.

### Changes Made

#### 1. Added `extract_histograms_from_json()` Function

```python
def extract_histograms_from_json(json_data):
    """Extract utility histograms from experiment JSON file (final time point)."""
    result = {
        'utility': {'global': {}, 'layers': {layer: {} for layer in LAYERS}},
        'raw_utility': {'global': {}},
    }

    # Extract from utility_histogram_per_step (100,000 time points)
    if 'utility_histogram_per_step' in json_data:
        hist_data = json_data['utility_histogram_per_step']
        # Get final values (last element in each array)
        for i, suffix in enumerate(UTILITY_BIN_SUFFIXES):
            key = f'hist_{suffix}_pct'
            if key in hist_data and len(hist_data[key]) > 0:
                result['utility']['global'][UTILITY_BIN_LABELS[i]] = hist_data[key][-1]

    # Extract per-layer data
    if 'layer_utility_histogram_per_step' in json_data:
        layer_data = json_data['layer_utility_histogram_per_step']
        for layer in LAYERS:
            if layer in layer_data:
                for i, suffix in enumerate(UTILITY_BIN_SUFFIXES):
                    key = f'hist_{suffix}_pct'
                    if key in layer_data[layer] and len(layer_data[layer][key]) > 0:
                        result['utility']['layers'][layer][UTILITY_BIN_LABELS[i]] = layer_data[layer][key][-1]

    return result
```

#### 2. Updated Dataset Configs

Added `use_json: True` flag and `experiments` dictionary with full paths:

```python
'input_mnist': {
    'display_name': 'Input-Permuted MNIST',
    'use_json': True,  # Extract from experiment JSON files
    'logs_subdir': 'input_permuted_mnist_stats',
    'seed': 2,
    'experiments': {
        'S&P': 'sgd/fully_connected_relu_with_hooks/lr_0.001_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
        'UPGD (Full)': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
        # ... all 8 methods
    }
},
```

Similar configuration for Mini-ImageNet.

#### 3. Modified `extract_for_dataset()` Function

Added conditional logic to use JSON extraction when `use_json: True`:

```python
def extract_for_dataset(dataset):
    config = DATASET_CONFIGS[dataset]

    if config.get('use_json', False):
        # Extract from experiment JSON files
        logs_dir = PROJECT_DIR / 'logs' / config['logs_subdir']
        seed = config.get('seed', 2)
        experiments = config.get('experiments', {})

        for exp_name, exp_path in experiments.items():
            json_path = logs_dir / exp_path / f'{seed}.json'
            with open(json_path) as f:
                json_data = json.load(f)
            hist_data = extract_histograms_from_json(json_data)
            all_data[exp_name] = hist_data
    else:
        # Original WandB extraction (for backward compatibility)
        # ...
```

### Data Format

The JSON files contain utility histogram time-series data with ~100,000 time points:

```json
{
  "utility_histogram_per_step": {
    "steps": [0, 10, 20, ...],  // ~100,000 steps (logged every 10)
    "hist_0_20_pct": [0.0, 0.0, ...],
    "hist_20_40_pct": [0.0, 0.0, ...],
    "hist_40_44_pct": [0.0, 0.0, ...],
    "hist_44_48_pct": [2.3, 2.5, ...],
    "hist_48_52_pct": [97.7, 97.5, ...],  // Most parameters here
    "hist_52_56_pct": [0.0, 0.0, ...],
    "hist_56_60_pct": [0.0, 0.0, ...],
    "hist_60_80_pct": [0.0, 0.0, ...],
    "hist_80_100_pct": [0.0, 0.0, ...],
    "global_max": [0.52, 0.52, ...],
    "total_params": 674950  // Mini-ImageNet FCN
  },
  "layer_utility_histogram_per_step": {
    "linear_1": { "steps": [...], "hist_0_20_pct": [...], ... },
    "linear_2": { ... },
    "linear_3": { ... }
  }
}
```

We extract the **final time point** (last value in each array) to get the end-of-training utility distribution.

---

## 5. Plots Generated

### Training Metrics (7 plots per dataset)

Generated by `plot_training_metrics.py`:

1. **Accuracy Comparison** - Classification accuracy per task
2. **Loss Comparison** - Cross-entropy loss per task
3. **Plasticity Comparison** - Learning ability (1 - L_after/L_before)
4. **Dead Units Comparison** - Fraction of neurons with zero activation
5. **Weight L2 Norm** - L2 norm of all weights
6. **Weight L1 Norm** - L1 norm of all weights
7. **Gradient L2 Norm** - L2 norm of gradients

**Processing:** Each plot computes per-task window averages (2500 steps) from per-step data (1M points) in JSON files.

### Utility Histograms (4 global + per-layer plots)

Generated by `plot_utility_histograms.py`:

**Global Histograms (4 plots):**
1. `utility_histogram.png` - 9-bin distribution (linear scale)
2. `utility_histogram_log.png` - 9-bin distribution (log scale)
3. `raw_utility_histogram.png` - 5-bin raw utility (linear scale)
4. `raw_utility_histogram_log.png` - 5-bin raw utility (log scale)

**Per-Layer Histograms (2 plots per method with layer data):**

Methods with per-layer data:
- UPGD (Full)
- UPGD (Output Only)
- UPGD (Hidden Only)
- UPGD (Hidden+Output)
- UPGD (Clamped 0.52)

For each method:
- `layer_utility_{method}.png` - 3 panels (linear_1, linear_2, linear_3)
- `layer_utility_{method}_log.png` - Same with log scale

**Total per-layer plots:**
- Input-MNIST: 5 methods ◊ 2 scales = 10 plots
- Mini-ImageNet: 5 methods ◊ 2 scales = 10 plots

### Utility Histogram Bins

#### Scaled Utility (9 bins)
| Bin | Range |
|-----|-------|
| 1 | [0, 0.2) |
| 2 | [0.2, 0.4) |
| 3 | [0.4, 0.44) |
| 4 | [0.44, 0.48) |
| 5 | [0.48, 0.52) ê **Central bin** |
| 6 | [0.52, 0.56) |
| 7 | [0.56, 0.6) |
| 8 | [0.6, 0.8) |
| 9 | [0.8, 1.0] |

**Finding:** 99.7-100% of parameters concentrate in bin 5 [0.48, 0.52).

#### Raw Utility (5 bins)
| Bin | Range |
|-----|-------|
| 1 | < -0.001 |
| 2 | [-0.001, -0.0002) |
| 3 | [-0.0002, 0.0002] ê **Center** |
| 4 | (0.0002, 0.001] |
| 5 | > 0.001 |

---

## 6. Execution Steps

### For Input-MNIST

```bash
cd /scratch/gilbreth/shin283/upgd/upgd_plots
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# 1. Generate training metric plots
python scripts/plot_training_metrics.py input_mnist

# 2. Extract utility histograms from JSON files
python scripts/extract_utility_histograms_local.py input_mnist

# 3. Plot utility histograms
python scripts/plot_utility_histograms.py input_mnist
```

**Output:**
- 7 training metric plots (2.4MB)
- 4 global utility histograms (0.3MB)
- 10 per-layer utility histograms (0.6MB)
- **Total: 21 plots, 3.3MB**

### For Mini-ImageNet

```bash
cd /scratch/gilbreth/shin283/upgd/upgd_plots
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# 1. Generate training metric plots
python scripts/plot_training_metrics.py mini_imagenet

# 2. Extract utility histograms from JSON files
python scripts/extract_utility_histograms_local.py mini_imagenet

# 3. Plot utility histograms
python scripts/plot_utility_histograms.py mini_imagenet
```

**Output:**
- 7 training metric plots (2.6MB)
- 4 global utility histograms (0.4MB)
- 10 per-layer utility histograms (1.0MB)
- **Total: 21 plots, 4.0MB**

### Runtime

- **Training metrics:** ~2-3 minutes per dataset (loading 8 ◊ 300MB JSON files)
- **Utility extraction:** ~30 seconds per dataset
- **Utility plotting:** ~10 seconds per dataset

---

## 7. Key Results Summary

### Input-MNIST
- **Best Accuracy:** S&P (0.7814) - baseline performs well on easier task
- **Best Plasticity:** UPGD variants (~0.50) vs S&P (0.31) - 63% better
- **Utility Concentration:** 99.7-100% in [0.48, 0.52) bin

### Mini-ImageNet
- **Best Accuracy:** UPGD (Output Only) (0.5901) - 104% better than S&P
- **Best Plasticity:** UPGD (Hidden+Output) (0.5523) - 23% better than S&P
- **Gating Impact:** Output Only > Full > Hidden+Output >> Hidden Only
- **Clamping Impact:** 25-43% accuracy reduction vs unclamped
- **Utility Concentration:** 99.97-99.99% in [0.48, 0.52) bin

---

## 8. Network Architecture

**Fully Connected Network (FCN):**
- **Input:** 784 (Input-MNIST) or 2048 (Mini-ImageNet ResNet50 features)
- **Layer 1 (linear_1):** input_dim í 300
- **Layer 2 (linear_2):** 300 í 150
- **Layer 3 (linear_3):** 150 í 100 (output)
- **Activation:** ReLU
- **Total Parameters:**
  - Input-MNIST: ~375K parameters
  - Mini-ImageNet: ~675K parameters

---

## 9. Conda Environment

**Name:** `/scratch/gilbreth/shin283/conda_envs/upgd`

**Key Packages:**
- Python 3.8
- PyTorch (for loading JSON data)
- Matplotlib (plotting)
- NumPy (data processing)
- Pandas (optional)

**Activation:**
```bash
conda activate /scratch/gilbreth/shin283/conda_envs/upgd
```

---

## 10. File Locations

### Source Data
```
/scratch/gilbreth/shin283/upgd/logs/
   input_permuted_mnist_stats/
      sgd/fully_connected_relu_with_hooks/lr_0.001_sigma_0.1_.../2.json (303MB)
      upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_.../2.json (329MB)
      ... (8 experiments total)
   label_permuted_mini_imagenet_stats/
       sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_.../2.json (306MB)
       upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_.../2.json (330MB)
       ... (8 experiments total)
```

### Generated Plots
```
/scratch/gilbreth/shin283/upgd/upgd_plots/figures/
   input_mnist/    (21 plots, 3.3MB)
   mini_imagenet/  (21 plots, 4.0MB)
```

### Extracted Data
```
/scratch/gilbreth/shin283/upgd/upgd_plots/data/utility_histograms/
   input_mnist_utility_histograms.json (extracted from 8 JSON files)
   mini_imagenet_utility_histograms.json (extracted from 8 JSON files)
```

---

## 11. Paper Integration

Results have been documented in:
- **Paper Body:** `/scratch/gilbreth/shin283/upgd/paper_body_gilbreh.md`
  - Section: Main Results í Input-Permuted MNIST
  - Section: Main Results í Mini-ImageNet

**Key Sections Added:**
1. Experimental setup and benchmark description
2. Accuracy and loss analysis
3. Plasticity maintenance comparison
4. Dead units and network capacity
5. Weight and gradient norms
6. Impact of gating strategy (Mini-ImageNet)
7. Utility clamping analysis (Mini-ImageNet)

**Figure References:**
- `figures/input_mnist/accuracy_comparison.png`
- `figures/mini_imagenet/accuracy_comparison.png`

---

## 12. Notes

### Clamped 0.52 Completion
- **Initial Status:** Experiment 10095040 was still running (only 90 bytes)
- **Completed:** December 18, 2025 at 15:57
- **Action Taken:** Re-ran all Mini-ImageNet plotting with Clamped 0.52 included
- **Result:** All 8 methods now complete

### Utility Histogram Time-Series Data
- JSON files contain **100,000 time points** (every 10 steps out of 1M)
- We extract the **final time point** for end-of-training distribution
- Future work could analyze utility evolution over time using full time-series

### Plotting Performance
- Large JSON files (300-330MB each) take ~20-30 seconds to load
- Plotting 7 metrics ◊ 8 methods requires loading each file 7 times
- Total time: ~2-3 minutes per dataset
- Consider caching loaded data for faster iteration

---

## 13. Future Work

### Additional Datasets
- **EMNIST** (label-permuted) - experiments ready, need plotting
- **CIFAR-10** (label-permuted) - only hyperparameter search done, need full runs

### Additional Analysis
- Utility evolution over time (use full 100K time-series)
- Per-task utility distribution changes
- Correlation between utility spread and task difficulty
- Layer-wise plasticity analysis

### Plot Enhancements
- Add error bars (need multiple seeds)
- Significance testing between methods
- Utility heatmaps over training
- Parameter importance visualization

---

## Document History

- **2025-12-18:** Initial creation
  - Completed Input-MNIST plots (21 plots)
  - Completed Mini-ImageNet plots (21 plots)
  - Fixed paths in plot_training_metrics.py
  - Modified extract_utility_histograms_local.py for JSON extraction
  - Fixed matplotlib style compatibility
  - Added results to paper_body_gilbreh.md
