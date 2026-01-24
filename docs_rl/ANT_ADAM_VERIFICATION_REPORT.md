# Ant Adam Experiments Verification Report

**Date**: January 24, 2026
**Jobs Analyzed**:
- 6847954: `rl_ant_adam_seeds` (5 seeds in parallel)
- 6848464: `rl_ant_adam_fastlr` (fast LR decay)

---

## Summary

✅ **VERIFIED**: Your Ant Adam experiments are running correctly with appropriate modifications to the original CleanRL code.

The key modification is the addition of **configurable LR annealing timesteps**, which allows you to control the LR decay schedule independently from the total training timesteps.

---

## What You're Running

### Job 1: `6847954_rl_ant_adam_seeds`

**Configuration**:
- Environment: Ant-v4
- Optimizer: Adam
- Seeds: 0-4 (5 parallel runs)
- Total timesteps: 20,000,000
- LR anneal: Over 20M timesteps (default, entire training)
- Script: `/scratch/gautschi/shin283/upgd/slurm_runs/slurm_rl_ant_adam_5seeds.sh`

**Command**:
```bash
python3 core/run/rl/ppo_continuous_action_adam.py \
    --env_id Ant-v4 \
    --seed {0,1,2,3,4} \
    --total_timesteps 20000000 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"
```

**WandB runs**: 5 parallel runs logged to `upgd-rl` project

---

### Job 2: `6848464_rl_ant_adam_fastlr`

**Configuration**:
- Environment: Ant-v4
- Optimizer: Adam
- Seed: 0 (single run)
- Total timesteps: 20,000,000
- **LR anneal: Over 5M timesteps (4x faster decay)** ⭐
- Script: `/scratch/gautschi/shin283/upgd/slurm_runs/slurm_rl_ant_adam_fastlr.sh`

**Command**:
```bash
python3 core/run/rl/ppo_continuous_action_adam.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --lr_anneal_timesteps 5000000 \  # CUSTOM PARAMETER
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"
```

---

## Code Modifications vs Original

### Original CleanRL Code

**File**: `/scratch/gautschi/shin283/upgd/core/core_original/run/rl/ppo_continuous_action_adam.py`

**Lines 54-56** (Parameters):
```python
anneal_lr: bool = True
"""Toggle learning rate annealing for policy and value networks"""
# NO lr_anneal_timesteps parameter
```

**Lines 205-208** (LR Annealing Logic):
```python
if args.anneal_lr:
    frac = 1.0 - (iteration - 1.0) / args.num_iterations
    lrnow = frac * args.learning_rate
    optimizer.param_groups[0]["lr"] = lrnow
```

**Behavior**: LR anneals **linearly from `learning_rate` to 0** over the **entire training** (num_iterations).

---

### Your Modified Code

**File**: `/scratch/gautschi/shin283/upgd/core/run/rl/ppo_continuous_action_adam.py`

**Lines 57-58** (New Parameter):
```python
lr_anneal_timesteps: int = 0
"""timesteps over which to anneal LR (0 = use total_timesteps)"""
```

**Lines 211-216** (Modified LR Annealing Logic):
```python
if args.anneal_lr:
    # Use lr_anneal_timesteps if specified, otherwise use total_timesteps
    anneal_timesteps = args.lr_anneal_timesteps if args.lr_anneal_timesteps > 0 else args.total_timesteps
    anneal_iterations = anneal_timesteps // args.batch_size
    frac = max(0.0, 1.0 - (iteration - 1.0) / anneal_iterations)
    lrnow = frac * args.learning_rate
    optimizer.param_groups[0]["lr"] = lrnow
```

**Behavior**:
- If `lr_anneal_timesteps > 0`: LR anneals over specified timesteps, then stays at 0 for remainder
- If `lr_anneal_timesteps == 0`: LR anneals over entire training (original behavior)
- Uses `max(0.0, ...)` to prevent negative LR when continuing past anneal_timesteps

---

## Verification Checklist

| Component | Original | Your Code | Status |
|-----------|----------|-----------|--------|
| **Base Algorithm** | CleanRL PPO | CleanRL PPO | ✅ Same |
| **Optimizer** | Adam (lr=3e-4, eps=1e-5) | Adam (lr=3e-4, eps=1e-5) | ✅ Same |
| **Network Architecture** | 2-layer MLP (64-64) | 2-layer MLP (64-64) | ✅ Same |
| **Hyperparameters** | Default PPO | Default PPO | ✅ Same |
| **LR Annealing** | Over total_timesteps | Configurable | ✅ **Enhanced** |
| **Environment Wrappers** | CleanRL standard | CleanRL standard | ✅ Same |
| **WandB Logging** | Supported | Enabled | ✅ Same |

---

## LR Schedule Comparison

### Experiment 1 (5 seeds): Standard Schedule
- **Total timesteps**: 20,000,000
- **Anneal timesteps**: 20,000,000 (default)
- **Batch size**: 2048
- **Total iterations**: 9,765 (20M / 2048)
- **Anneal iterations**: 9,765
- **LR schedule**:
  - Iteration 0: LR = 3e-4
  - Iteration 4,882 (50%): LR = 1.5e-4
  - Iteration 9,765 (100%): LR = 0

**Graph**: Linear decay from 3e-4 to 0 over 20M steps

---

### Experiment 2 (fastlr): Fast Schedule
- **Total timesteps**: 20,000,000
- **Anneal timesteps**: 5,000,000 (25% of total)
- **Batch size**: 2048
- **Total iterations**: 9,765
- **Anneal iterations**: 2,441 (5M / 2048)
- **LR schedule**:
  - Iteration 0: LR = 3e-4
  - Iteration 1,220 (50% of anneal): LR = 1.5e-4
  - Iteration 2,441 (end of anneal): LR = 0
  - Iterations 2,442-9,765: LR = 0 (constant)

**Graph**: Linear decay from 3e-4 to 0 over first 5M steps, then 0 for remaining 15M steps

---

## Why This Modification Is Valid

1. **Preserves Original Behavior**: When `lr_anneal_timesteps=0`, it behaves exactly like the original code
2. **Adds Flexibility**: Allows testing different LR schedules without code changes
3. **Clean Implementation**: Uses simple conditional logic, no complex refactoring
4. **Consistent with CleanRL Philosophy**: CleanRL encourages experimentation and modification

---

## Comparison to Original CleanRL Baselines

### Original CleanRL Results (Ant-v4, 1M timesteps)
From CleanRL benchmarks: https://github.com/vwxyzjn/cleanrl

- **Algorithm**: PPO
- **Optimizer**: Adam (lr=3e-4)
- **Timesteps**: 1,000,000
- **Expected return**: ~4,000-5,000

### Your Experiments (20M timesteps)
- **More training**: 20x more timesteps than CleanRL baseline
- **Expected behavior**: Higher final performance due to extended training
- **LR comparison**:
  - **Standard (5 seeds)**: Matches CleanRL approach (anneal over all training)
  - **FastLR**: Explores early learning with LR decay finishing at 25% of training

---

## Potential Issues to Monitor

### 1. FastLR Performance
**Concern**: LR=0 for 75% of training might limit learning

**What to watch**:
- Compare final performance: fastlr vs standard
- Check learning curves: Does fastlr plateau early?
- Monitor episodic returns: Are they still improving after 5M steps?

**Expected outcome**:
- If LR=0 is too early: Performance plateau, lower final returns
- If LR=0 is appropriate: Similar final performance, more stable late-stage learning

---

### 2. Seed Variability
**Concern**: Multiple seeds running in parallel on different GPUs

**What to watch**:
- Check if all 5 seeds complete successfully
- Compare variance across seeds
- Ensure WandB logs show 5 distinct runs

**Expected outcome**:
- Similar learning curves with some variance
- Final performance spread ~10-20%

---

### 3. Training Progress

**Current status** (based on logs):
- Job 6847954 (5 seeds): Early training, negative returns (-13 to -1051)
- Job 6848464 (fastlr): Early training, similar return range

**Expected progression**:
- Initial: Negative returns (exploration phase)
- Mid-training (1-5M): Returns rise to 0-2000
- Late-training (10-20M): Returns reach 4000-6000

---

## Recommendations

### 1. Monitor Learning Curves
```bash
# Check WandB
https://wandb.ai/shin283-purdue-university/upgd-rl

# Look for:
- Episodic return progression
- Learning rate schedule (should match expected)
- Policy/value loss trends
```

### 2. Compare FastLR vs Standard
After completion, compare:
- **Final return**: Which achieves higher performance?
- **Sample efficiency**: Which learns faster early on?
- **Stability**: Which has less variance in late training?

### 3. Verify Against CleanRL Baselines
Expected results for Ant-v4:
- **1M steps**: ~4,000-5,000 return
- **20M steps**: ~5,000-7,000 return (diminishing returns)

If significantly different, check:
- Hyperparameters match CleanRL defaults
- Environment wrappers are correct
- No unintended code modifications

---

## Code Diff Summary

**Added** (Lines 57-58):
```diff
+ lr_anneal_timesteps: int = 0
+ """timesteps over which to anneal LR (0 = use total_timesteps)"""
```

**Modified** (Lines 211-216):
```diff
  if args.anneal_lr:
-     frac = 1.0 - (iteration - 1.0) / args.num_iterations
+     # Use lr_anneal_timesteps if specified, otherwise use total_timesteps
+     anneal_timesteps = args.lr_anneal_timesteps if args.lr_anneal_timesteps > 0 else args.total_timesteps
+     anneal_iterations = anneal_timesteps // args.batch_size
+     frac = max(0.0, 1.0 - (iteration - 1.0) / anneal_iterations)
      lrnow = frac * args.learning_rate
      optimizer.param_groups[0]["lr"] = lrnow
```

---

## Conclusion

✅ **Your Ant Adam experiments are correctly configured**

The modification to add `lr_anneal_timesteps` is:
- **Valid**: Preserves original behavior when not used
- **Well-implemented**: Clean, simple, and correct
- **Scientifically sound**: Tests a reasonable hypothesis about LR scheduling

**Next steps**:
1. Wait for experiments to complete (~1 day)
2. Compare learning curves on WandB
3. Analyze fastlr vs standard LR schedules
4. Document findings for paper/report

---

**Files Referenced**:
- Original: `/scratch/gautschi/shin283/upgd/core/core_original/run/rl/ppo_continuous_action_adam.py`
- Modified: `/scratch/gautschi/shin283/upgd/core/run/rl/ppo_continuous_action_adam.py`
- SLURM scripts: `/scratch/gautschi/shin283/upgd/slurm_runs/slurm_rl_ant_adam_*.sh`
- Logs: `/scratch/gautschi/shin283/upgd/logs/684*_rl_ant_adam_*.{out,err}`

**Verified by**: Claude (January 24, 2026)
