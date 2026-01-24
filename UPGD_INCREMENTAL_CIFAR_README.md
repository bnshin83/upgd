# UPGD for Incremental CIFAR-100

Implementation of UPGD (Utility-based Perturbed Gradient Descent) optimizer for the incremental CIFAR-100 continual learning benchmark.

## Implementation Summary

### Files Created

1. **UPGD Optimizer**
   - `lop/algos/upgd.py` - Custom PyTorch optimizer implementing UPGD
   - Features:
     - Utility tracking per parameter: `u_t = β_u * u_{t-1} + (1 - β_u) * (-∇L * θ)`
     - Sigmoid-based gating for selective updates
     - Adam-style adaptive moments (optional)
     - Noise perturbation for exploration
     - Comprehensive logging (utility histograms, norms, gating statistics)

2. **Configuration Files**
   - `lop/incremental_cifar/cfg/upgd_baseline.json` - UPGD without CBP
   - `lop/incremental_cifar/cfg/upgd_with_cbp.json` - UPGD with Continual Backpropagation

3. **SLURM Scripts** (in `slurm_runs/`)
   - `slurm_incremental_cifar_upgd.sh` - Run UPGD baseline
   - `slurm_incremental_cifar_sgd_baseline.sh` - Run SGD baseline for comparison
   - `slurm_incremental_cifar_upgd_with_cbp.sh` - Run UPGD + CBP
   - `slurm_incremental_cifar_upgd_variants.sh` - Hyperparameter sweep (array job)

4. **Testing**
   - `test_upgd_incremental_cifar.py` - Integration tests

### Files Modified

1. **Experiment File**
   - `lop/incremental_cifar/incremental_cifar_experiment.py`
   - Changes:
     - Added UPGD config parameters (lines 93-100)
     - Added conditional optimizer initialization (lines 116-134)
     - Added UPGD utility logging (lines 196-202, 296-320)

## UPGD Algorithm Details

### Utility Computation
```
u_t = β_u * u_{t-1} + (1 - β_u) * (-grad * param)
```

### Gating Mechanism
```
gate = 1 - sigmoid(u_t / max(u))
```
- `gate = 1`: Full update (low utility = needs plasticity)
- `gate = 0`: Preserve weights (high utility = important knowledge)

### Parameter Update
```
θ_{t+1} = θ_t * (1 - lr * weight_decay) - 2 * lr * update
update = (m_t / sqrt(v_t) + ε) * gate + noise * gate
```

Where:
- `m_t`: First moment (bias-corrected)
- `v_t`: Second moment (bias-corrected)
- `noise`: Gaussian perturbation

## Hyperparameters

### Default Values
- `lr`: 0.1 (same as SGD baseline)
- `beta_utility`: 0.999 (utility EMA decay)
- `sigma`: 0.001 (noise scale)
- `beta1`: 0.9 (first moment decay)
- `beta2`: 0.999 (second moment decay)
- `weight_decay`: 0.0005
- `momentum`: 0.9

### Hyperparameter Sweep
The variants script tests:
- **beta_utility**: 0.99, 0.999, 0.9999
- **sigma**: 0.0001, 0.001, 0.01
- **seeds**: 0, 1, 2

## Usage

### 1. Run Tests
```bash
cd /scratch/gautschi/shin283/upgd
python test_upgd_incremental_cifar.py
```

Expected output: All tests passed ✓

### 2. Submit SLURM Jobs

**Single UPGD run:**
```bash
cd /scratch/gautschi/shin283/upgd/slurm_runs
sbatch slurm_incremental_cifar_upgd.sh
```

**SGD baseline (for comparison):**
```bash
sbatch slurm_incremental_cifar_sgd_baseline.sh
```

**UPGD + CBP:**
```bash
sbatch slurm_incremental_cifar_upgd_with_cbp.sh
```

**Hyperparameter sweep:**
```bash
sbatch slurm_incremental_cifar_upgd_variants.sh
```

### 3. Monitor Jobs
```bash
squeue -u $USER
```

### 4. Check Logs
```bash
# SLURM logs
tail -f /scratch/gautschi/shin283/upgd/logs/<job_id>_incr_cifar_upgd.out

# WandB: https://wandb.ai/shin283-purdue-university/upgd-incremental-cifar
```

## Expected Results

### Training Metrics (logged per epoch)
- Train loss/accuracy
- Test loss/accuracy
- Validation loss/accuracy

### UPGD-Specific Metrics
- `upgd_global_max_utility`: Maximum utility across all parameters
- `upgd_mean_utility`: Average utility
- `upgd_utility_sparsity`: Fraction of near-zero utilities
- `upgd_mean_gate_value`: Average gating value (1 = update, 0 = preserve)
- `upgd_active_fraction`: Fraction of parameters with gate > 0.5

### Expected Behavior
1. **Utility values should be non-zero** after first epoch
2. **Gating should be active** (not all 0 or all 1)
3. **Mean gate value ~0.5** indicates balanced plasticity/stability
4. **No NaN/Inf values** in loss or utilities
5. **Learning curves should be stable** (no divergence)

## Comparison with Baselines

### SGD Baseline
- Fixed learning rate schedule (decay at epochs 60, 120, 160)
- No adaptive gating
- Expected to show catastrophic forgetting

### UPGD Advantages
- Selective parameter updates based on utility
- Maintains plasticity for low-utility parameters
- Preserves important knowledge in high-utility parameters
- Noise injection for exploration

### Expected Improvements
- Lower catastrophic forgetting on earlier tasks
- Better final test accuracy
- More stable learning curves
- Higher retention across all 20 tasks

## Troubleshooting

### Import Errors
```bash
# Ensure correct environment
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Check Python path
export PYTHONPATH=/scratch/gautschi/shin283/loss-of-plasticity:$PYTHONPATH
```

### CUDA Out of Memory
- Reduce batch size in config (default: 90)
- Use fewer workers (default: 12)

### NaN/Inf Values
- Reduce learning rate
- Reduce sigma (noise scale)
- Check weight_decay isn't too large

### Low Gating Activity
If `mean_gate_value` is always near 0 or 1:
- Adjust `beta_utility` (higher = slower utility decay)
- Adjust `sigma` (higher = more exploration)

## Next Steps

1. **Run initial experiments:**
   - UPGD baseline (seed 0)
   - SGD baseline (seed 0)
   - Compare learning curves

2. **Hyperparameter tuning:**
   - Run sweep for different beta_utility and sigma
   - Analyze utility distributions
   - Select best configuration

3. **Full evaluation:**
   - Run multiple seeds (0, 1, 2)
   - Compute average and std across seeds
   - Analyze forgetting metrics

4. **Ablation studies:**
   - UPGD vs UPGD+CBP
   - Adam moments vs SGD momentum
   - Different noise schedules

## Files Reference

```
/scratch/gautschi/shin283/
├── loss-of-plasticity/lop/
│   ├── algos/
│   │   └── upgd.py                    # UPGD optimizer implementation
│   └── incremental_cifar/
│       ├── cfg/
│       │   ├── upgd_baseline.json     # UPGD config
│       │   └── upgd_with_cbp.json     # UPGD + CBP config
│       └── incremental_cifar_experiment.py  # Modified experiment file
└── upgd/
    ├── slurm_runs/
    │   ├── slurm_incremental_cifar_upgd.sh           # UPGD baseline
    │   ├── slurm_incremental_cifar_sgd_baseline.sh   # SGD baseline
    │   ├── slurm_incremental_cifar_upgd_with_cbp.sh  # UPGD + CBP
    │   └── slurm_incremental_cifar_upgd_variants.sh  # Hyperparameter sweep
    ├── test_upgd_incremental_cifar.py  # Integration tests
    └── UPGD_INCREMENTAL_CIFAR_README.md  # This file
```

## References

- Original UPGD paper: [Add reference when available]
- Continual Backpropagation: [Dohare et al., 2021]
- Incremental CIFAR-100 benchmark: [Rebuffi et al., 2017]

## Contact

For questions or issues:
- Slack: @shin283
- Email: shin283@purdue.edu
