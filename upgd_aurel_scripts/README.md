# UPGD Incremental CIFAR-100 Variants - SLURM Scripts

This directory contains SLURM scripts for running all UPGD variants on the incremental CIFAR-100 continual learning benchmark.

## Overview

These scripts run UPGD (Utility-based Perturbed Gradient Descent) with different layer-selective gating configurations on the Gautschi cluster.

## Variants

### Layer-Selective Gating Modes

1. **Full Gating** (`upgd_full`): Utility gating applied to ALL layers
   - Standard UPGD behavior
   - All parameters use utility-based gating

2. **Output-Only Gating** (`upgd_output_only`): Utility gating ONLY on final FC layer
   - Gated: `fc.weight`, `fc.bias` (classification head)
   - Non-gated: All conv/bn layers (fixed scale 0.5)
   - Hypothesis: High plasticity for new classes, stable features

3. **Hidden-Only Gating** (`upgd_hidden_only`): Utility gating on ALL layers EXCEPT final FC
   - Gated: All conv/bn layers
   - Non-gated: `fc.weight`, `fc.bias` (fixed scale 0.5)
   - Hypothesis: Adaptive features, stable output mapping

### Combinations with CBP

Each gating mode can be combined with Continual Backpropagation (CBP):
- **CBP**: Replaces dead neurons with fresh initializations
- **Synergy**: UPGD provides adaptive gating, CBP replaces non-contributing neurons

## Scripts

### Individual Variant Scripts

- `slurm_incr_cifar_sgd_baseline.sh` - SGD baseline (no UPGD)
- `slurm_incr_cifar_upgd_full.sh` - UPGD with full gating
- `slurm_incr_cifar_upgd_output_only.sh` - UPGD with output-only gating
- `slurm_incr_cifar_upgd_hidden_only.sh` - UPGD with hidden-only gating
- `slurm_incr_cifar_upgd_output_only_cbp.sh` - UPGD output-only + CBP
- `slurm_incr_cifar_upgd_hidden_only_cbp.sh` - UPGD hidden-only + CBP

### Array Job (Run All Variants)

- `slurm_incr_cifar_all_variants.sh` - SLURM array job running all 6 variants in parallel

## Usage

### Run Single Variant

```bash
sbatch slurm_incr_cifar_upgd_full.sh
```

### Run All Variants in Parallel

```bash
sbatch slurm_incr_cifar_all_variants.sh
```

This submits 6 jobs (array indices 0-5), one for each variant.

## Configuration Files

Each script uses a corresponding config file in `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/`:

- `upgd_baseline.json` → Full gating
- `upgd_output_only.json` → Output-only gating
- `upgd_hidden_only.json` → Hidden-only gating
- `upgd_output_only_cbp.json` → Output-only + CBP
- `upgd_hidden_only_cbp.json` → Hidden-only + CBP

## WandB Tracking

All experiments log to WandB:
- Project: `upgd-incremental-cifar`
- Entity: `shin283-purdue-university`
- Run names: `{job_id}_incr_cifar_{variant}_seed_{seed}`

## Cluster Configuration

- Partition: `ai`
- GPUs: 1 per job
- CPUs: 14 per job
- Time limit: 7 days
- Account: `jhaddock`

## Expected Runtimes

- Each experiment: ~4-6 days (4000 epochs on 20 tasks)
- Array job total: ~4-6 days (parallel execution)

## Logs

Logs are saved to `/scratch/gautschi/shin283/upgd/logs/`:
- `{job_id}_incr_cifar_{variant}.out` - Standard output
- `{job_id}_incr_cifar_{variant}.err` - Error output

## Results

Results are saved to `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/`

## Key Metrics to Compare

1. **Final Test Accuracy**: Performance after all 20 tasks
2. **Backward Transfer**: Accuracy on early tasks (forgetting)
3. **Forward Transfer**: Accuracy on new tasks (plasticity)
4. **Utility Distributions**: Per-layer utility histograms
5. **Gated vs Non-Gated Params**: Parameter counts per layer type
6. **Dead Neurons** (CBP variants): Fraction of inactive neurons

## Implementation Details

### ResNet18 Architecture

- Input: 32x32x3 CIFAR-100 images
- Layers: `conv1`, `layer1.*`, `layer2.*`, `layer3.*`, `layer4.*`, `fc`
- Output: 100 classes

### Gating Logic

```python
def _should_apply_gating(param_name, gating_mode):
    if gating_mode == 'full':
        return True  # Gate all layers
    elif gating_mode == 'output_only':
        return param_name.startswith('fc.')  # Gate only fc.weight, fc.bias
    elif gating_mode == 'hidden_only':
        return not param_name.startswith('fc.')  # Gate all except fc.*
```

### Non-Gated Layers

Non-gated layers use a fixed scaling factor (0.5) instead of utility-based gating:
```python
update = (gradient * 0.5) / sqrt(variance) + noise * 0.5
```

## Verification

Before running full experiments, verify setup:

```bash
# Check config files exist
ls -lh /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_*.json

# Check conda environment
conda activate /scratch/gautschi/shin283/conda_envs/lop
python -c "from lop.algos.upgd import UPGD; print('UPGD import successful')"

# Check GPU availability
nvidia-smi
```

## Contact

- Author: Shin Lee (shin283@purdue.edu)
- Cluster: Gautschi (Purdue University)
- Date: January 2026
