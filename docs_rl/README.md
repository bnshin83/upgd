# UPGD RL Experiments - Documentation Hub

**Location**: `/scratch/gautschi/shin283/upgd/docs_rl/`
**Purpose**: Documentation and verification reports for UPGD RL experiments
**Date**: January 2026

---

## 📁 Documentation Files

### 1. **ANT_ADAM_VERIFICATION_REPORT.md** ⭐
Verification of Ant-v4 Adam baseline experiments

**What it covers**:
- ✅ Verification of Adam optimizer implementation vs CleanRL original
- ✅ Analysis of `lr_anneal_timesteps` modification
- ✅ Comparison: standard vs fast LR decay schedules
- ✅ Expected results and monitoring recommendations

**Jobs verified**:
- 6847954: Adam with 5 seeds (standard LR schedule)
- 6848464: Adam with fast LR decay

**Use this for**: Understanding baseline Adam experiments

---

### 2. **UPGD_VARIANTS_VERIFICATION.md** 📘
Comprehensive verification of UPGD layer-selective variants

**What it covers**:
- ✅ Layer-selective gating implementation (`rl_upgd_layerselective.py`)
- ✅ Unified PPO script (`run_ppo_upgd.py`)
- ✅ All optimizer variants: adam, upgd_full, upgd_output_only, upgd_hidden_only
- ✅ Actor-Critic architecture breakdown
- ✅ Gating behavior by variant
- ✅ SLURM scripts verification
- ✅ Issues found and recommendations
- ✅ Expected results and hypotheses

**Use this for**: Before running UPGD variant experiments

---

## 🎯 Quick Reference

### Optimizer Variants Available

| Variant | Gating Applied To | Non-Gated | Use Case |
|---------|-------------------|-----------|----------|
| **adam** | None | All | Baseline |
| **upgd_full** | All layers | None | Standard UPGD |
| **upgd_output_only** | actor_mean.4, critic.4 | Hidden layers | High output plasticity |
| **upgd_hidden_only** | actor_mean.0/2, critic.0/2 | Output layers | Adaptive features |
| **upgd_actor_output** | actor_mean.4 only | All others | Policy plasticity |
| **upgd_critic_output** | critic.4 only | All others | Value plasticity |

---

## 📊 Experiments Overview

### Adam Baseline
**Status**: ✅ Running (Jobs 6847954, 6848464)
- Standard schedule: LR decay over 20M steps (5 seeds)
- Fast schedule: LR decay over 5M steps (1 seed)

### UPGD Variants
**Status**: ⏳ Ready to run (pending verification review)
- Basic UPGD: `slurm_rl_ant_upgd.sh`
- Output-only: `slurm_rl_ant_upgd_output.sh`
- Hidden-only: `slurm_rl_ant_upgd_hidden.sh`

---

## 🔧 Key Files

### Implementation
```
/scratch/gautschi/shin283/upgd/core/run/rl/
├── adaupgd.py                      # Basic UPGD
├── rl_upgd_layerselective.py       # Layer-selective UPGD ⭐
├── ppo_continuous_action_adam.py   # PPO + Adam
├── ppo_continuous_action_upgd.py   # PPO + UPGD (old)
└── run_ppo_upgd.py                 # Unified script ⭐
```

### SLURM Scripts
```
/scratch/gautschi/shin283/upgd/slurm_runs/
├── slurm_rl_ant_adam_5seeds.sh     # Adam baseline (5 seeds)
├── slurm_rl_ant_adam_fastlr.sh     # Adam fast LR
├── slurm_rl_ant_upgd.sh            # UPGD basic
├── slurm_rl_ant_upgd_output.sh     # UPGD output-only
└── slurm_rl_ant_upgd_hidden.sh     # UPGD hidden-only
```

---

## ⚠️ Issues Found

### 1. Inconsistent Weight Decay
**File**: `slurm_rl_ant_upgd_hidden.sh`
**Issue**: Missing `--weight_decay 0.0` parameter
**Impact**: Will use default 0.001 instead of 0.0
**Fix**: Add `--weight_decay 0.0` to match other scripts

### 2. Old vs New Scripts
**File**: `slurm_rl_ant_upgd.sh`
**Issue**: Uses old `ppo_continuous_action_upgd.py` instead of new `run_ppo_upgd.py`
**Impact**: No access to layer-selective gating framework
**Recommendation**: Update to use `run_ppo_upgd.py --optimizer upgd_full`

---

## 🚀 How to Use This Documentation

### Before Running Adam Experiments
1. Read: `ANT_ADAM_VERIFICATION_REPORT.md`
2. Verify: LR schedules match your experimental design
3. Monitor: WandB for learning curves

### Before Running UPGD Experiments
1. Read: `UPGD_VARIANTS_VERIFICATION.md`
2. Fix: Issues listed in section 11
3. Verify: Layer gating logic matches your hypothesis
4. Submit: SLURM scripts
5. Monitor: WandB for utility statistics

---

## 📈 Expected Results

### Adam (Baseline)
- Final return (Ant-v4, 20M): ~5000-6000
- Learning curve: Steady improvement, plateaus around 10M steps

### UPGD Full
- Similar or better final performance
- Better sample efficiency early on
- More stable learning (utility gating prevents catastrophic updates)

### UPGD Output-Only
- Faster initial convergence (high output plasticity)
- Risk: Hidden features may become stale

### UPGD Hidden-Only
- Better feature adaptation
- May be slower to adapt to new reward structures

---

## 📊 Monitoring

### WandB Project
```
https://wandb.ai/shin283-purdue-university/upgd-rl
```

### Key Metrics
- `charts/episodic_return`: Learning progress
- `charts/learning_rate`: LR schedule verification
- `utility/global_max`: Maximum utility value
- `utility/hist_*_pct`: Utility distribution
- `layer/*/gating_applied`: Verify gating logic

### Logs
```bash
# Check job status
squeue -u shin283

# Monitor output
tail -f /scratch/gautschi/shin283/upgd/logs/<job_id>_*.out

# Check errors
tail -f /scratch/gautschi/shin283/upgd/logs/<job_id>_*.err
```

---

## 🔗 Related Documentation

- **Incremental CIFAR docs**: `/scratch/gautschi/shin283/upgd/docs/`
- **Original CleanRL**: https://github.com/vwxyzjn/cleanrl
- **UPGD Paper**: (add reference when available)

---

## 📞 Contact

- **Author**: Shin Lee (shin283@purdue.edu)
- **Institution**: Purdue University
- **Cluster**: Gautschi (ai partition)

---

**Last Updated**: January 24, 2026
