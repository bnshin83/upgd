# UPGD RL Experiments - Issues Fixed

**Date**: January 24, 2026
**Status**: ✅ All issues resolved

---

## Issues Identified & Fixed

### Issue 1: Missing Weight Decay Parameter ✅ FIXED

**File**: `/scratch/gautschi/shin283/upgd/slurm_runs/slurm_rl_ant_upgd_hidden.sh`

**Problem**:
```bash
# BEFORE (missing --weight_decay)
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --optimizer upgd_hidden_only \
    --cuda \
    --track
```
- Would use default `weight_decay=0.001`
- Inconsistent with other scripts using `weight_decay=0.0`

**Fix Applied** (line 39):
```bash
# AFTER (added --weight_decay 0.0)
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --optimizer upgd_hidden_only \
    --weight_decay 0.0 \           # ← ADDED
    --cuda \
    --track
```

**Impact**: Now consistent with other UPGD scripts (weight_decay=0.0)

---

### Issue 2: Using Old Script ✅ FIXED

**File**: `/scratch/gautschi/shin283/upgd/slurm_runs/slurm_rl_ant_upgd.sh`

**Problem**:
```bash
# BEFORE (old script)
python3 core/run/rl/ppo_continuous_action_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --weight_decay 0.0
```
- Used old `ppo_continuous_action_upgd.py`
- No access to layer-selective gating framework
- No utility statistics logging
- Inconsistent with other variant scripts

**Fix Applied** (lines 28-45):
```bash
# AFTER (new unified script)
echo "Running RL: Ant-v4 - UPGD Full Gating"
echo "Using unified framework (run_ppo_upgd.py)"

export WANDB_RUN_NAME="${SLURM_JOB_ID}_ant_upgd_full_seed_0"
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --optimizer upgd_full \         # ← ADDED
    --weight_decay 0.0 \
    --cuda \
    --track
```

**Benefits**:
- ✅ Uses new unified framework (`run_ppo_upgd.py`)
- ✅ Explicit optimizer selection (`--optimizer upgd_full`)
- ✅ Access to comprehensive utility statistics
- ✅ Consistent with other variant scripts
- ✅ Updated WandB run name for clarity

---

## Verification

### Files Modified

1. **slurm_rl_ant_upgd_hidden.sh**
   - Line 39: Added `--weight_decay 0.0`

2. **slurm_rl_ant_upgd.sh**
   - Lines 28-31: Updated echo messages
   - Line 35: Updated WandB run name
   - Lines 36-45: Switched to `run_ppo_upgd.py --optimizer upgd_full`

### Quick Check

```bash
# Verify fixes
cd /scratch/gautschi/shin283/upgd/slurm_runs

# Issue 1: Check weight_decay added
grep "weight_decay" slurm_rl_ant_upgd_hidden.sh
# Should output: --weight_decay 0.0 \

# Issue 2: Check new script used
grep "run_ppo_upgd.py" slurm_rl_ant_upgd.sh
grep "upgd_full" slurm_rl_ant_upgd.sh
# Should find both
```

---

## All UPGD Scripts Summary (After Fixes)

| Script | Optimizer | Script Used | Weight Decay | Status |
|--------|-----------|-------------|--------------|--------|
| `slurm_rl_ant_upgd.sh` | upgd_full | run_ppo_upgd.py | 0.0 | ✅ Fixed |
| `slurm_rl_ant_upgd_output.sh` | upgd_output_only | run_ppo_upgd.py | 0.0 | ✅ Already OK |
| `slurm_rl_ant_upgd_hidden.sh` | upgd_hidden_only | run_ppo_upgd.py | 0.0 | ✅ Fixed |

**All scripts now**:
- ✅ Use unified framework (`run_ppo_upgd.py`)
- ✅ Have consistent `weight_decay=0.0`
- ✅ Support comprehensive utility logging
- ✅ Ready to run

---

## What's Different Now

### Before Fixes

**Inconsistencies**:
- ❌ Mixed old/new scripts
- ❌ Inconsistent weight_decay
- ❌ Missing utility statistics for full UPGD
- ❌ Confusing script naming

**Scripts worked, but**:
- Different logging capabilities
- Hard to compare across variants
- Incomplete utility tracking

### After Fixes

**Consistency**:
- ✅ All use same framework
- ✅ Same hyperparameters (weight_decay=0.0)
- ✅ All variants log utility statistics
- ✅ Clear naming convention

**Benefits**:
- Easy to compare across variants
- Comprehensive logging for all
- Same code path for all UPGD variants
- Clear experiment tracking in WandB

---

## Ready to Run

All UPGD variant scripts are now ready:

```bash
cd /scratch/gautschi/shin283/upgd

# Submit all variants
sbatch slurm_runs/slurm_rl_ant_upgd.sh         # UPGD full gating
sbatch slurm_runs/slurm_rl_ant_upgd_output.sh  # UPGD output-only
sbatch slurm_runs/slurm_rl_ant_upgd_hidden.sh  # UPGD hidden-only
```

**Expected behavior**:
- All use `run_ppo_upgd.py`
- All log to WandB project `upgd-rl`
- All track utility statistics
- Consistent hyperparameters across variants

---

## Comparison Matrix

| Parameter | Full | Output-Only | Hidden-Only |
|-----------|------|-------------|-------------|
| **Script** | run_ppo_upgd.py | run_ppo_upgd.py | run_ppo_upgd.py |
| **Optimizer flag** | upgd_full | upgd_output_only | upgd_hidden_only |
| **Weight decay** | 0.0 | 0.0 | 0.0 |
| **LR** | 3e-4 | 3e-4 | 3e-4 |
| **Beta utility** | 0.999 | 0.999 | 0.999 |
| **Sigma** | 0.001 | 0.001 | 0.001 |
| **Non-gated scale** | 0.5 | 0.5 | 0.5 |
| **Actor hidden gating** | ✅ | ❌ (0.5) | ✅ |
| **Actor output gating** | ✅ | ✅ | ❌ (0.5) |
| **Critic hidden gating** | ✅ | ❌ (0.5) | ✅ |
| **Critic output gating** | ✅ | ✅ | ❌ (0.5) |

---

## Next Steps

1. ✅ **Fixes applied** - Scripts ready
2. ⏳ **Submit jobs** - Run all variants
3. ⏳ **Monitor WandB** - Check utility statistics
4. ⏳ **Compare results** - Analyze learning curves

---

**Fixed by**: Claude (January 24, 2026)
**Verification**: See `UPGD_VARIANTS_VERIFICATION.md` for full details
