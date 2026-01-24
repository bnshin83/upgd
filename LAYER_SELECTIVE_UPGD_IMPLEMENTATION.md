# Layer-Selective UPGD Implementation for Incremental CIFAR-100

## Implementation Complete ✓

This document summarizes the implementation of layer-selective UPGD variants for the incremental CIFAR-100 continual learning benchmark.

## What Was Implemented

### 1. UPGD Optimizer with Layer-Selective Gating ✓

**File**: `/scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py`

**Changes**:
- Added `gating_mode` parameter: `'full'`, `'output_only'`, `'hidden_only'`
- Added `non_gated_scale` parameter: scaling factor for non-gated layers (default: 0.5)
- Implemented `_should_apply_gating(param_name, gating_mode)` method:
  - `full`: Gates all parameters (standard UPGD)
  - `output_only`: Gates only `fc.weight`, `fc.bias` (final classification layer)
  - `hidden_only`: Gates all parameters EXCEPT `fc.*` (all conv/bn layers)
- Updated `step()` method to apply conditional gating:
  - Gated layers: Use utility-based scaling with sigmoid
  - Non-gated layers: Use fixed scaling factor (non_gated_scale)
- Enhanced `get_gating_stats()` to track:
  - `gating_mode`: Current mode
  - `gated_params`: Count of gated parameters
  - `non_gated_params`: Count of non-gated parameters

### 2. Experiment Configuration Support ✓

**File**: `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/incremental_cifar_experiment.py`

**Changes**:
- Added `upgd_gating_mode` parameter (line 101)
- Added `upgd_non_gated_scale` parameter (line 102)
- Updated UPGD optimizer initialization to pass these parameters (lines 131-132)
- Added logging for gating mode and parameter counts (lines 322-325)

### 3. Configuration Files ✓

**Location**: `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/`

**Updated**:
- `upgd_baseline.json` - Added `upgd_gating_mode: "full"`, `upgd_non_gated_scale: 0.5`
- `upgd_with_cbp.json` - Added `upgd_gating_mode: "full"`, `upgd_non_gated_scale: 0.5`

**Created**:
- `upgd_output_only.json` - Output-only gating variant
- `upgd_hidden_only.json` - Hidden-only gating variant
- `upgd_output_only_cbp.json` - Output-only + CBP
- `upgd_hidden_only_cbp.json` - Hidden-only + CBP

### 4. SLURM Scripts ✓

**Location**: `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/`

**Created**:
- `README.md` - Comprehensive documentation
- `slurm_incr_cifar_sgd_baseline.sh` - SGD baseline
- `slurm_incr_cifar_upgd_full.sh` - Full gating
- `slurm_incr_cifar_upgd_output_only.sh` - Output-only gating
- `slurm_incr_cifar_upgd_hidden_only.sh` - Hidden-only gating
- `slurm_incr_cifar_upgd_output_only_cbp.sh` - Output-only + CBP
- `slurm_incr_cifar_upgd_hidden_only_cbp.sh` - Hidden-only + CBP
- `slurm_incr_cifar_all_variants.sh` - Array job for all variants

All scripts are executable and configured for the Gautschi cluster (ai partition, jhaddock account).

### 5. Test Script ✓

**File**: `/scratch/gautschi/shin283/upgd/test_layer_selective_gating.py`

**Tests**:
- UPGD initialization with different gating modes
- ResNet18 layer name detection
- Gating logic correctness
- Parameter counting (gated vs non-gated)
- Forward/backward pass with optimizer step

## How to Verify the Implementation

### Step 1: Activate Environment

```bash
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop
cd /scratch/gautschi/shin283/upgd
```

### Step 2: Run Test Script

```bash
python3.8 test_layer_selective_gating.py
```

**Expected Output**:
- All 5 tests should PASS
- Should show layer names, parameter counts, and gating statistics

### Step 3: Verify Config Files

```bash
ls -lh /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_*.json
```

**Expected**: 6 JSON files (baseline, with_cbp, output_only, hidden_only, output_only_cbp, hidden_only_cbp)

### Step 4: Verify SLURM Scripts

```bash
ls -lh /scratch/gautschi/shin283/upgd/upgd_aurel_scripts/
```

**Expected**: 8 files (README.md + 7 .sh scripts)

### Step 5: Test Single Variant (Dry Run)

```bash
# Don't actually submit, just check syntax
cd /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar
python3.8 incremental_cifar_experiment.py --config ./cfg/upgd_output_only.json --help
```

## How to Run Experiments

### Option 1: Run Single Variant

```bash
cd /scratch/gautschi/shin283/upgd/upgd_aurel_scripts
sbatch slurm_incr_cifar_upgd_output_only.sh
```

### Option 2: Run All Variants in Parallel

```bash
cd /scratch/gautschi/shin283/upgd/upgd_aurel_scripts
sbatch slurm_incr_cifar_all_variants.sh
```

This will submit 6 jobs (array indices 0-5), one for each variant:
- 0: SGD baseline
- 1: UPGD full gating
- 2: UPGD output-only
- 3: UPGD hidden-only
- 4: UPGD output-only + CBP
- 5: UPGD hidden-only + CBP

### Monitor Jobs

```bash
squeue -u shin283
```

### Check Logs

```bash
tail -f /scratch/gautschi/shin283/upgd/logs/<job_id>_*.out
```

## ResNet18 Layer Structure

### Final Layer (fc.*)
- `fc.weight`: (100, 512) - 51,200 parameters
- `fc.bias`: (100,) - 100 parameters
- **Total**: 51,300 parameters (~0.5% of model)

### Hidden Layers (conv/bn layers)
- `conv1.*`: Initial convolution
- `layer1.*`: ResNet blocks (64 filters)
- `layer2.*`: ResNet blocks (128 filters)
- `layer3.*`: ResNet blocks (256 filters)
- `layer4.*`: ResNet blocks (512 filters)
- **Total**: ~11.1M parameters (~99.5% of model)

## Gating Behavior by Mode

### Full Gating (Baseline)
- **Gated**: All 11.2M parameters
- **Non-gated**: 0 parameters
- **Behavior**: Standard UPGD, all layers adapt based on utility

### Output-Only Gating
- **Gated**: fc.* (51,300 parameters, 0.5%)
- **Non-gated**: All conv/bn layers (11.1M parameters, 99.5%)
- **Behavior**:
  - Output layer: Adaptive utility-based updates (high plasticity for new classes)
  - Hidden layers: Fixed scaling 0.5 (stable feature extraction)
- **Hypothesis**: Good for learning new classes while preserving features

### Hidden-Only Gating
- **Gated**: All conv/bn layers (11.1M parameters, 99.5%)
- **Non-gated**: fc.* (51,300 parameters, 0.5%)
- **Behavior**:
  - Hidden layers: Adaptive utility-based updates (plastic features)
  - Output layer: Fixed scaling 0.5 (stable class mapping)
- **Hypothesis**: Good for adapting features while maintaining output stability

## Expected Results

### Metrics to Track (WandB)

1. **Test Accuracy**: Final performance after 20 tasks
2. **Per-Task Accuracy**: Accuracy on each of the 20 tasks
3. **Backward Transfer**: Change in accuracy on old tasks (forgetting)
4. **Forward Transfer**: Initial performance on new tasks
5. **Utility Histograms**: Distribution of utilities (0-1 range)
6. **Gating Statistics**:
   - Mean gate value
   - Active fraction (gate > 0.5)
   - Gated vs non-gated parameter counts

### Hypothesis

| Variant | Final Acc | Forgetting | New Task Perf | Key Strength |
|---------|-----------|------------|---------------|--------------|
| SGD | Baseline | High | Medium | Simple baseline |
| UPGD Full | High | Low | High | Balanced plasticity |
| UPGD Output | Medium-High | Medium | Very High | New class learning |
| UPGD Hidden | Medium-High | Very Low | Medium | Feature preservation |
| UPGD Output + CBP | High | Low | Very High | New classes + recovery |
| UPGD Hidden + CBP | High | Very Low | High | Features + recovery |

## WandB Project

- **Project**: `upgd-incremental-cifar`
- **Entity**: `shin283-purdue-university`
- **Run Names**: `{job_id}_incr_cifar_{variant}_seed_{seed}`

## File Summary

### Modified Files (2)
1. `/scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py`
2. `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/incremental_cifar_experiment.py`

### Created Config Files (4 new + 2 updated)
1. `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_output_only.json`
2. `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_hidden_only.json`
3. `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_output_only_cbp.json`
4. `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_hidden_only_cbp.json`
5. `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_baseline.json` (updated)
6. `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_with_cbp.json` (updated)

### Created SLURM Scripts (8)
1. `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/README.md`
2. `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_sgd_baseline.sh`
3. `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_full.sh`
4. `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_output_only.sh`
5. `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_hidden_only.sh`
6. `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_output_only_cbp.sh`
7. `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_hidden_only_cbp.sh`
8. `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_all_variants.sh`

### Created Test Script (1)
1. `/scratch/gautschi/shin283/upgd/test_layer_selective_gating.py`

### Documentation (1)
1. `/scratch/gautschi/shin283/upgd/LAYER_SELECTIVE_UPGD_IMPLEMENTATION.md` (this file)

## Next Steps

1. **Verify Implementation**:
   ```bash
   source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
   conda activate /scratch/gautschi/shin283/conda_envs/lop
   cd /scratch/gautschi/shin283/upgd
   python3.8 test_layer_selective_gating.py
   ```

2. **Run Short Test** (optional, 1 epoch):
   - Modify config to set `num_epochs: 1`
   - Run single variant to verify no crashes
   - Check logs for gating statistics

3. **Submit Full Experiments**:
   ```bash
   cd /scratch/gautschi/shin283/upgd/upgd_aurel_scripts
   sbatch slurm_incr_cifar_all_variants.sh
   ```

4. **Monitor Progress**:
   - Check SLURM queue: `squeue -u shin283`
   - Monitor WandB: https://wandb.ai/shin283-purdue-university/upgd-incremental-cifar
   - Check logs: `tail -f /scratch/gautschi/shin283/upgd/logs/*.out`

5. **Analysis** (after 4-6 days):
   - Download results from WandB
   - Compare learning curves across variants
   - Analyze utility distributions by layer
   - Measure backward/forward transfer
   - Compare with RL results

## Implementation Notes

### Design Decisions

1. **Parameter Naming**: Used ResNet18 naming convention (`fc.*` for final layer)
2. **Non-Gated Scaling**: Fixed at 0.5 (matches RL implementation)
3. **Utility Tracking**: Still computed for all layers, only gating is selective
4. **Config Files**: Separate files for each variant for clarity
5. **SLURM Scripts**: Individual scripts + array job for flexibility

### Compatibility

- **RL Implementation**: Uses same gating logic, adapted for ResNet18 layer names
- **Existing Code**: Backward compatible (defaults to `gating_mode='full'`)
- **CBP Integration**: Works seamlessly with Continual Backpropagation

### Performance Considerations

- **Overhead**: Minimal (just conditional logic in update step)
- **Memory**: No additional memory required
- **Runtime**: Expected to be same as full UPGD

## Contact

- **Author**: Shin Lee (shin283@purdue.edu)
- **Institution**: Purdue University
- **Date**: January 2026
- **Repository**: `/scratch/gautschi/shin283/upgd/`

## References

- RL Implementation: `/scratch/gautschi/shin283/upgd/core/run/rl/rl_upgd_layerselective.py`
- Original UPGD: `/scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py`
- Incremental CIFAR: `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/`
