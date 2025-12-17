# UPGD Plotting Guidelines

## Summary of Work Done

This document tracks the plotting setup and fixes applied to the UPGD experiments.

---

## 1. Folder Structure Created

```
/scratch/gautschi/shin283/upgd_plots/
    scripts/
        plot_training_metrics.py           # Training metrics (multi-dataset)
        extract_utility_histograms_local.py # Extract utility histograms from WandB (multi-dataset)
        plot_utility_histograms.py         # Plot utility histogram figures (multi-dataset)
        plot_mini_imagenet.py              # (legacy) Mini-ImageNet only version
    data/
        utility_histograms/
            {dataset}_utility_histograms.json  # Extracted histogram data per dataset
    figures/
        {dataset}/                         # Separate folder per dataset (PNG only)
            accuracy_comparison.png
            loss_comparison.png
            plasticity_comparison.png
            dead_units_comparison.png
            weight_l2_comparison.png
            weight_l1_comparison.png
            grad_l2_comparison.png
            utility_histogram.png          # 9-bin utility distribution
            utility_histogram_log.png      # 9-bin utility distribution (log scale)
            raw_utility_histogram.png      # 5-bin raw utility distribution
            raw_utility_histogram_log.png  # 5-bin raw utility distribution (log scale)
            layer_utility_*.png            # Per-layer utility distribution
    plot_guideline.md                      # This file

Supported datasets: mini_imagenet, input_mnist, emnist, cifar10
```

---

## 2. Bug Fixes Applied

### Bug #1: Per-Task Averaging (CRITICAL)
**Location:** `/scratch/gautschi/shin283/upgd/core/run/run_stats_with_curvature.py`

**Problem:**
- Per-step arrays were NOT reset after each task boundary
- This caused per-task metrics to compute **cumulative averages** instead of **per-task window averages**
- Original UPGD behavior: reset lists after each task to get 2500-step window averages

**Fix Applied (lines 519-535):**
```python
# Reset per-step arrays after each task (matches original UPGD behavior)
losses_per_step = []
if self.task.criterion == 'cross_entropy':
    accuracy_per_step = []
plasticity_per_step = []
n_dead_units_per_step = []
weight_rank_per_step = []
weight_l2_per_step = []
weight_l1_per_step = []
grad_l2_per_step = []
grad_l1_per_step = []
grad_l0_per_step = []
```

**Impact:**
- Future experiments will log correct per-task window averages
- Existing JSON files retain original per-step data (all_*_per_step arrays with 1M points)

### Bug #2: torch.torch Typo (MINOR - doesn't affect FCN)
**Location:**
- `/scratch/gautschi/shin283/upgd/core/run/run_stats.py` line 106
- `/scratch/gautschi/shin283/upgd/core/run/run_stats_with_curvature.py` line 335

**Problem:**
- Double torch reference: `torch.torch.linalg.matrix_rank()`
- Only triggered for CNN experiments (not fully connected networks)

**Fix Applied:**
```python
# Changed from:
sample_weight_rank += torch.torch.linalg.matrix_rank(param.data).float().mean()

# To:
sample_weight_rank += torch.linalg.matrix_rank(param.data).float().mean()
```

### Bug #3: Test Script Mislabeling (DOCUMENTATION)
**Location:** `/scratch/gautschi/shin283/upgd/test_upgd_fo_global_mini_imagenet_stats.sh`

**Problem:**
- Job name and echo messages claimed `sigma=0.01, wd=0.001` (paper values)
- Actual parameters: `sigma=0.001, wd=0.0`

**Fix Applied:**
- Updated job name, output paths, WandB run name, and echo messages to reflect actual parameters

---

## 3. Data Understanding

### JSON File Structure

Each experiment produces a JSON file with:

#### Per-Step Data (1,000,000 points - CORRECT)
- `losses_per_step`: loss at each step
- `accuracy_per_step`: accuracy at each step
- `plasticity_per_step`: plasticity at each step
- `n_dead_units_per_step`: dead units at each step
- `weight_l2_per_step`, `weight_l1_per_step`: weight norms at each step
- `grad_l2_per_step`, `grad_l1_per_step`, `grad_l0_per_step`: gradient stats at each step

#### Per-Task Data (400 points)
**In existing JSON files (before fix):**
- `losses`: cumulative average from step 0 to current task (WRONG)
- `accuracies`: cumulative average from step 0 to current task (WRONG)
- Similar for other per-task metrics

**After fix (future experiments):**
- `losses`: 2500-step window average for each task (CORRECT)
- `accuracies`: 2500-step window average for each task (CORRECT)

---

## 4. Plotting Approach

### Current Implementation

**Script:** `/scratch/gautschi/shin283/upgd_plots/scripts/plot_mini_imagenet.py`

**Method:**
1. Loads per-step data from JSON (1M points)
2. Converts to per-task window averages (400 points):
   ```python
   steps_per_task = 2500
   n_tasks = len(per_step_data) // steps_per_task

   for task_idx in range(n_tasks):
       start = task_idx * steps_per_task
       end = (task_idx + 1) * steps_per_task
       per_task_avg.append(np.mean(per_step_data[start:end]))
   ```
3. Plots 400 points (one per task) with x-axis in steps

**Matches:** Original UPGD paper plotting style

### Experiments Plotted

| Name | Path | Description |
|------|------|-------------|
| S&P | sgd/...lr_0.005_sigma_0.001... | Shrink & Perturb baseline |
| UPGD (Full) | upgd_fo_global/...lr_0.01_sigma_0.001... | Standard UPGD (all layers gated) |
| UPGD (Output Only) | upgd_fo_global_outputonly/.../gating_mode_output_only | Gating only on linear_3 |
| UPGD (Hidden Only) | upgd_fo_global_hiddenonly/.../gating_mode_hidden_only | Gating only on linear_1, linear_2 |
| UPGD (Hidden+Output) | upgd_fo_global_hiddenandoutput/.../gating_mode_hidden_and_output | Gating on all layers |
| UPGD (Clamped 0.52) | upgd_fo_global_clamped052/... | Utility clamped to max 0.52 |
| UPGD (Clamped 0.48-0.52) | upgd_fo_global_clamped_48_52/.../min_clamp_0.48_max_clamp_0.52 | Utility in [0.48, 0.52] |
| UPGD (Clamped 0.44-0.56) | upgd_fo_global_clamped_44_56/.../min_clamp_0.44_max_clamp_0.56 | Utility in [0.44, 0.56] |

### Metrics Plotted

1. **Accuracy** - Classification accuracy per task
2. **Loss** - Cross-entropy loss per task
3. **Plasticity** - Learning ability (1 - L_after/L_before)
4. **Dead Units** - Fraction of neurons with zero activation
5. **Weight L2** - L2 norm of all weights
6. **Weight L1** - L1 norm of all weights
7. **Gradient L2** - L2 norm of gradients

### Plot Style

- Figure size: 12x7 inches
- X-axis: Steps (0, 200k, 400k, 600k, 800k, 1M)
- Line width: 2.0
- S&P uses dashed line, others solid
- Color-coded by method
- Legends show method name only (no avg values)
- Output: PNG only (no PDF)

---

## 5. Key Results (Mini-ImageNet, Seed 2)

| Method | Accuracy | Loss | Plasticity | Dead Units |
|--------|----------|------|------------|------------|
| **UPGD (Output Only)** | **0.5919** | **1.9582** | 0.5340 | 0.8296 |
| UPGD (Hidden+Output) | 0.5491 | 2.0663 | 0.5480 | 0.7467 |
| UPGD (Full) | 0.5434 | 2.0866 | **0.5521** | 0.7484 |
| UPGD (Clamped 0.52) | 0.4107 | 2.6956 | 0.4243 | 0.8920 |
| UPGD (Clamped 0.44-0.56) | 0.3771 | 2.8668 | 0.3973 | 0.9023 |
| UPGD (Clamped 0.48-0.52) | 0.3086 | 3.2696 | 0.3427 | 0.9332 |
| UPGD (Hidden Only) | 0.2331 | 3.6393 | 0.2283 | 0.9535 |
| S&P | 0.1116 | 4.1459 | 0.1337 | **0.9641** |

**Key Finding:** UPGD (Output Only) achieves best accuracy and loss, confirming that utility gating on the output layer is most effective.

---

## 6. WandB vs JSON Comparison

### WandB Logs (during training)
- Logs every 10 steps -> ~100,000 points for 1M steps
- Per-step metrics: `training/loss`, `training/accuracy`, etc.
- Per-task metrics: `task_level/loss`, `task_level/accuracy`, etc.
- CSV exports have irregular sampling (WandB aggregation)

### JSON Files (saved at end)
- Per-step: all 1,000,000 points (every step)
- Per-task: 400 points (per-task window averages after fix)
- Complete data, no sampling

### For Plotting
- **Use JSON per-step data** (more complete)
- **Compute per-task window averages** (matches original UPGD)
- Results match WandB trends

---

## 7. How to Generate Plots

```bash
cd /scratch/gautschi/shin283/upgd_plots
conda activate plasticity
python scripts/plot_mini_imagenet.py
```

Output: `/scratch/gautschi/shin283/upgd_plots/figures/mini_imagenet/*.png`

---

## 8. Next Steps for Other Datasets

To create plots for CIFAR-10, EMNIST, Input-MNIST:

1. Copy `plot_mini_imagenet.py`
2. Update paths:
   - `LOGS_DIR = BASE_DIR / 'logs' / 'label_permuted_cifar10_stats'`
   - `PLOT_DIR = Path('/scratch/gautschi/shin283/upgd_plots/figures/cifar10')`
3. Update experiment configs to match available runs
4. Run the script

---

## 9. Future Experiment Checklist

Before running new experiments:

- [ ] Verify bug fixes are in place (per-task reset, torch.torch typo)
- [ ] Check test script parameters match job name
- [ ] Set WandB environment variables correctly
- [ ] Ensure `run_stats_with_curvature.py` is used (not `run_stats.py`)
- [ ] Verify seed is set consistently
- [ ] Check hyperparameters match intended configuration

After experiments complete:

- [ ] Verify JSON files have both per-step (1M) and per-task (400) data
- [ ] Check per-task data is window average (not cumulative)
- [ ] Generate plots using per-task window averaging
- [ ] Compare with WandB dashboard for validation

---

## 10. Utility Histogram Analysis

### Overview

Added utility histogram analysis to visualize the distribution of scaled and raw utility values across parameters.

### Data Source

Utility histograms are logged to WandB during training:
- **Scaled utility**: 9 bins in [0, 1] range after sigmoid normalization
- **Raw utility**: 5 bins centered at 0 (before scaling)
- **Per-layer data**: Available for some experiments

### Scripts Created

#### `extract_utility_histograms_local.py`
Extracts histogram data from local WandB summary files.

```bash
cd /scratch/gautschi/shin283/upgd_plots
conda activate plasticity
python scripts/extract_utility_histograms_local.py
```

**Features:**
- Reads from `/scratch/gautschi/shin283/upgd/wandb/run-*/files/wandb-summary.json`
- Extracts scaled utility histogram (9 bins)
- Extracts raw utility histogram (5 bins)
- Extracts per-layer histograms when available
- Computes global histogram from per-layer data if global not logged
- Saves to JSON for plotting

**Output:** `/scratch/gautschi/shin283/upgd_plots/data/utility_histograms/mini_imagenet_utility_histograms.json`

#### `plot_utility_histograms.py`
Creates histogram visualization plots.

```bash
cd /scratch/gautschi/shin283/upgd_plots
conda activate plasticity
python scripts/plot_utility_histograms.py
```

**Output:** `/scratch/gautschi/shin283/upgd_plots/figures/mini_imagenet/` (dataset-specific folder)

### Histogram Bins

#### Scaled Utility (9 bins)
| Bin | Range |
|-----|-------|
| 1 | [0, 0.2) |
| 2 | [0.2, 0.4) |
| 3 | [0.4, 0.44) |
| 4 | [0.44, 0.48) |
| 5 | [0.48, 0.52) |
| 6 | [0.52, 0.56) |
| 7 | [0.56, 0.6) |
| 8 | [0.6, 0.8) |
| 9 | [0.8, 1.0] |

#### Raw Utility (5 bins)
| Bin | Range |
|-----|-------|
| 1 | < -0.001 |
| 2 | [-0.001, -0.0002) |
| 3 | [-0.0002, 0.0002] |
| 4 | (0.0002, 0.001] |
| 5 | > 0.001 |

### Figures Generated

| File | Description |
|------|-------------|
| `utility_histogram.png` | Scatter plot comparing utility distribution across methods (9 bins) |
| `utility_histogram_log.png` | Same as above with log scale y-axis |
| `raw_utility_histogram.png` | Scatter plot comparing raw utility distribution (5 bins) |
| `raw_utility_histogram_log.png` | Same as above with log scale y-axis |

### Key Findings

1. **Scaled utility concentrated at 0.5**: All methods show ~100% of parameters in [0.48, 0.52) bin
2. **Raw utility differences**:
   - S&P: Tightly concentrated at zero (99.99% in [-0.0002, 0.0002])
   - UPGD (Full): More spread (88.5% center, ~5% each in adjacent bins)
   - UPGD (Clamped 0.52): Similar to Full but slightly tighter

### WandB Run IDs (Dec 1, 2025)

| Method | Run ID |
|--------|--------|
| UPGD (Full) | 2rorn0u1 |
| UPGD (Output Only) | lv4hrwao |
| UPGD (Hidden Only) | ddyu1m95 |
| UPGD (Hidden+Output) | tval9wyc |
| UPGD (Clamped 0.52) | ap0ll118 |
| UPGD (Clamped 0.48-0.52) | b1n4yksn |
| UPGD (Clamped 0.44-0.56) | jqlsiz5h |
| S&P | 4q72hdev |

### Notes

- Some runs (Clamped 0.48-0.52, Clamped 0.44-0.56) don't have histogram data in WandB summary
- Per-layer data available for: Output Only, Hidden Only, Hidden+Output, Clamped 0.52
- Full raw utility histogram (64 bins) available for: S&P, UPGD (Full), Clamped 0.52

---

## 11. Utility Histogram Data Storage Fix

### Problem Identified

Utility histogram data (9 bins) was only being logged to WandB during training, but **not saved to local JSON files**. This meant:
- Historical data required slow WandB API queries (~100,000 data points)
- No local backup of utility distribution over time
- Inconsistent with other metrics that are saved to JSON

### Solution Implemented

Modified `/scratch/gautschi/shin283/upgd/core/run/run_stats_with_curvature.py` to collect and save utility histogram time-series data.

#### Change 1: Add Tracking Dictionaries (lines 162-181)

```python
# Utility histogram tracking (9 bins) - logged every 10 steps
all_utility_hist_per_step = {
    'steps': [],  # Which steps have utility data
    'hist_0_20_pct': [],
    'hist_20_40_pct': [],
    'hist_40_44_pct': [],
    'hist_44_48_pct': [],
    'hist_48_52_pct': [],
    'hist_52_56_pct': [],
    'hist_56_60_pct': [],
    'hist_60_80_pct': [],
    'hist_80_100_pct': [],
    'global_max': [],
}

# Per-layer utility histograms
all_layer_utility_hist_per_step = {
    'linear_1': {'steps': [], 'hist_0_20_pct': [], 'hist_20_40_pct': [], ...},
    'linear_2': {'steps': [], 'hist_0_20_pct': [], 'hist_20_40_pct': [], ...},
    'linear_3': {'steps': [], 'hist_0_20_pct': [], 'hist_20_40_pct': [], ...},
}
```

#### Change 2: Collect During Training (lines 467-488)

```python
# Add utility metrics if available (use optimizer instance, not class)
if hasattr(optimizer, 'get_utility_stats'):
    utility_stats = optimizer.get_utility_stats()
    step_metrics.update(utility_stats)

    # Save utility histogram data to local tracking (for JSON export)
    if 'utility/hist_48_52_pct' in utility_stats:
        all_utility_hist_per_step['steps'].append(i)
        all_utility_hist_per_step['hist_0_20_pct'].append(utility_stats.get('utility/hist_0_20_pct', 0))
        all_utility_hist_per_step['hist_20_40_pct'].append(utility_stats.get('utility/hist_20_40_pct', 0))
        # ... (all 9 bins)
        all_utility_hist_per_step['global_max'].append(utility_stats.get('utility/global_max', 0))

        # Per-layer utility histograms
        for layer in ['linear_1', 'linear_2', 'linear_3']:
            key_48_52 = f'layer/{layer}/hist_48_52_pct'
            if key_48_52 in utility_stats:
                all_layer_utility_hist_per_step[layer]['steps'].append(i)
                all_layer_utility_hist_per_step[layer]['hist_48_52_pct'].append(utility_stats[key_48_52])
                # ... (all bins for this layer)
```

#### Change 3: Save to JSON (lines 680-683)

```python
# Add utility histogram data (9 bins) if collected
if all_utility_hist_per_step['steps']:
    log_data['utility_histogram_per_step'] = all_utility_hist_per_step
    log_data['layer_utility_histogram_per_step'] = all_layer_utility_hist_per_step
```

### Impact

**Future Experiments:**
- Utility histogram time-series (~100,000 points) will be saved to JSON automatically
- No need to query WandB API for historical utility distribution data
- Consistent with other per-step metrics

**Existing Experiments:**
- Already completed runs only have utility data in WandB
- Must use WandB export or API to retrieve historical utility distributions
- Script created: `fetch_utility_histograms_from_wandb.py` (slow, recommend manual export)

### Expected JSON Format

```json
{
  "utility_histogram_per_step": {
    "steps": [0, 10, 20, ...],  // ~100,000 steps (logged every 10)
    "hist_0_20_pct": [0.0, 0.0, ...],
    "hist_20_40_pct": [0.0, 0.0, ...],
    "hist_40_44_pct": [0.0, 0.0, ...],
    "hist_44_48_pct": [2.3, 2.5, ...],
    "hist_48_52_pct": [97.7, 97.5, ...],  // Most parameters in this bin
    "hist_52_56_pct": [0.0, 0.0, ...],
    "hist_56_60_pct": [0.0, 0.0, ...],
    "hist_60_80_pct": [0.0, 0.0, ...],
    "hist_80_100_pct": [0.0, 0.0, ...],
    "global_max": [0.52, 0.52, ...],
    "total_params": 674950  // Total number of parameters in the network
  },
  "layer_utility_histogram_per_step": {
    "linear_1": {
      "steps": [0, 10, 20, ...],
      "hist_0_20_pct": [...],
      // ... all 9 bins
    },
    "linear_2": { ... },
    "linear_3": { ... }
  }
}
```

---

## 12. Utility Histogram Plotting (Current)

### Plot Style

`/scratch/gautschi/shin283/upgd_plots/scripts/plot_utility_histograms.py` creates:

1. **Scatter plots with 9 bins on x-axis** (utility distribution)
2. **Scatter plots with 5 bins on x-axis** (raw utility distribution)
3. **Both linear and log-scale versions**

### Data Source

**Data extracted from WandB summary files** using `extract_utility_histograms_local.py`:
- Reads from `/scratch/gautschi/shin283/upgd/wandb/run-*/files/wandb-summary.json`
- Saves to `/scratch/gautschi/shin283/upgd_plots/data/utility_histograms/{dataset}_utility_histograms.json`

### Plot Types Generated

| File | Description |
|------|-------------|
| `utility_histogram.png` | 9-bin utility distribution (linear scale) |
| `utility_histogram_log.png` | 9-bin utility distribution (log scale) |
| `raw_utility_histogram.png` | 5-bin raw utility distribution (linear scale) |
| `raw_utility_histogram_log.png` | 5-bin raw utility distribution (log scale) |
| `layer_utility_{method}.png` | Per-layer utility distribution (3 panels: linear_1, linear_2, linear_3) |
| `layer_utility_{method}_log.png` | Per-layer utility distribution (log scale) |

**Per-layer data available for:** UPGD (Output Only), UPGD (Hidden Only), UPGD (Hidden+Output), UPGD (Clamped 0.52)

### Total Parameters

- **Mini-ImageNet FCN**: 674,950 parameters
  - Input: 2048 (ResNet50 bottleneck features)
  - linear_1: 2048 → 300 = 614,700 params
  - linear_2: 300 → 150 = 45,150 params
  - linear_3: 150 → 100 = 15,100 params
- **Note**: WandB histogram visualization samples 100,000 points, but percentages are computed from all parameters

### Key Findings

1. **Utility concentration at 0.5:** All methods keep ~99%+ parameters in [0.48, 0.52] bin
2. **Log-scale reveals differences:** Small changes > 0.52 (< 0.1% of parameters) correlate with performance
3. **UPGD (Output Only)** shows most spread outside [0.48, 0.52] range

---

## 13. Multi-Dataset Support

All scripts now support multiple datasets via command-line argument.

### Supported Datasets

| Dataset | Display Name | Log Directory |
|---------|--------------|---------------|
| `mini_imagenet` | Mini-ImageNet | `label_permuted_mini_imagenet_stats` |
| `input_mnist` | Input-Permuted MNIST | `input_permuted_mnist_stats` |
| `emnist` | Label-Permuted EMNIST | `label_permuted_emnist_stats` |
| `cifar10` | Label-Permuted CIFAR-10 | `label_permuted_cifar10_stats` |

### Usage

```bash
# Training metrics plots
python scripts/plot_training_metrics.py mini_imagenet
python scripts/plot_training_metrics.py input_mnist
python scripts/plot_training_metrics.py emnist
python scripts/plot_training_metrics.py cifar10

# Extract utility histograms from WandB
python scripts/extract_utility_histograms_local.py mini_imagenet
python scripts/extract_utility_histograms_local.py input_mnist

# Plot utility histograms
python scripts/plot_utility_histograms.py mini_imagenet
python scripts/plot_utility_histograms.py input_mnist
```

### Adding New Dataset Configurations

1. **For training metrics** (`plot_training_metrics.py`):
   - Add experiment paths to `DATASET_CONFIGS[dataset]['experiments']`
   - Each experiment needs: `path`, `color`, `linestyle`

2. **For utility histograms** (`extract_utility_histograms_local.py`):
   - Add WandB run IDs to `DATASET_CONFIGS[dataset]['runs']`
   - Run IDs can be found in WandB dashboard or `wandb/run-*-{run_id}` folder names

---

## Document History

- 2025-12-16: Initial creation, documented all fixes and plotting approach
- 2025-12-16: Added utility histogram analysis (Section 10)
- 2025-12-16: Added utility histogram data storage fix (Section 11) and plotting updates (Section 12)
- 2025-12-16: Updated plotting to use 9-bin scatter charts (not time-series), output to dataset-specific folder
- 2025-12-16: Added `total_params` tracking to JSON output for future experiments (674,950 for Mini-ImageNet)
- 2025-12-16: Added multi-dataset support to all scripts (Section 13)
