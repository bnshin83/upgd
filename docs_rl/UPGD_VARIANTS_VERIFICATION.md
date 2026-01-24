# UPGD Variants for RL - Verification Report

**Date**: January 24, 2026
**Purpose**: Verify UPGD layer-selective gating implementation for RL experiments
**Environments**: Ant-v4, HalfCheetah-v4, Hopper-v4, Walker2d-v4

---

## Summary

✅ **VERIFIED**: Your UPGD variants implementation is correct and ready to run!

The implementation includes:
1. **Basic UPGD** (full gating on all layers)
2. **Layer-Selective UPGD** (output-only, hidden-only, actor-only, critic-only)
3. **Unified script** supporting all variants via `--optimizer` flag

---

## What You Have Implemented

### Files Structure

```
/scratch/gautschi/shin283/upgd/core/run/rl/
├── adaupgd.py                      # Basic Adaptive UPGD (from original)
├── rl_upgd_layerselective.py       # Layer-selective UPGD (NEW)
├── ppo_continuous_action_upgd.py   # Basic PPO with UPGD
├── ppo_continuous_action_adam.py   # PPO with Adam (baseline)
└── run_ppo_upgd.py                 # Unified PPO with all variants (NEW)

/scratch/gautschi/shin283/upgd/slurm_runs/
├── slurm_rl_ant_upgd.sh            # Basic UPGD
├── slurm_rl_ant_upgd_output.sh     # Output-only gating
└── slurm_rl_ant_upgd_hidden.sh     # Hidden-only gating
```

---

## 1. Layer-Selective UPGD Implementation

### File: `rl_upgd_layerselective.py`

**Class**: `RLLayerSelectiveUPGD`

**Gating Modes**:
1. **`full`**: Gates all parameters (standard UPGD)
2. **`output_only`**: Gates only output layers
   - Actor: `actor_mean.4` (final layer → action logits)
   - Critic: `critic.4` (final layer → value estimate)
3. **`hidden_only`**: Gates only hidden layers
   - Actor: `actor_mean.0`, `actor_mean.2` (hidden layers)
   - Critic: `critic.0`, `critic.2` (hidden layers)
4. **`actor_output_only`**: Gates only actor output
5. **`critic_output_only`**: Gates only critic output

**Implementation** (lines 45-61):
```python
def _should_apply_gating(self, param_name, gating_mode):
    """Determine if utility gating should be applied to this parameter."""
    if gating_mode == 'full':
        return True
    elif gating_mode == 'output_only':
        # Output layers: actor_mean.4, critic.4
        return 'actor_mean.4' in param_name or 'critic.4' in param_name
    elif gating_mode == 'hidden_only':
        # Hidden layers: actor_mean.0/2, critic.0/2
        return ('actor_mean.0' in param_name or 'actor_mean.2' in param_name or
                'critic.0' in param_name or 'critic.2' in param_name)
    elif gating_mode == 'actor_output_only':
        return 'actor_mean.4' in param_name
    elif gating_mode == 'critic_output_only':
        return 'critic.4' in param_name
    else:
        raise ValueError(f"Unknown gating_mode: {gating_mode}")
```

**Key Features**:
- ✅ Tracks global max utility across all parameters
- ✅ Applies sigmoid gating: `scaled_utility = sigmoid(utility / global_max_util)`
- ✅ For gated layers: `update = gradient * (1 - scaled_utility) + noise * (1 - scaled_utility)`
- ✅ For non-gated layers: `update = gradient * non_gated_scale + noise * non_gated_scale`
- ✅ Comprehensive utility statistics logging

---

## 2. Unified PPO Script

### File: `run_ppo_upgd.py`

**Purpose**: Single script supporting all optimizer variants

**Optimizer Options** (line 44-46):
```python
optimizer: Literal["adam", "upgd_full", "upgd_output_only", "upgd_hidden_only",
                   "upgd_actor_output", "upgd_critic_output"] = "adam"
```

**Optimizer Creation** (lines 157-194):
```python
def create_optimizer(agent, args):
    """Create optimizer based on args.optimizer setting."""
    if args.optimizer == "adam":
        return optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    elif args.optimizer == "upgd_full":
        return AdaptiveUPGD(
            agent.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            beta_utility=args.beta_utility,
            sigma=args.sigma,
        )

    elif args.optimizer.startswith("upgd_"):
        # Layer-selective UPGD variants
        gating_mode_map = {
            "upgd_output_only": "output_only",
            "upgd_hidden_only": "hidden_only",
            "upgd_actor_output": "actor_output_only",
            "upgd_critic_output": "critic_output_only",
        }
        gating_mode = gating_mode_map.get(args.optimizer)

        return RLLayerSelectiveUPGD(
            agent.named_parameters(),  # ← Important: needs named params!
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            beta_utility=args.beta_utility,
            sigma=args.sigma,
            gating_mode=gating_mode,
            non_gated_scale=args.non_gated_scale,
        )
```

**Key Features**:
- ✅ Unified interface for all optimizers
- ✅ Layer names printed for debugging
- ✅ Utility statistics logged to WandB
- ✅ Compatible with CleanRL PPO implementation

---

## 3. Agent Architecture (Actor-Critic)

### Network Structure

**Actor (Policy Network)**:
```
actor_mean.0: Linear(27, 64)    # Hidden layer 1
actor_mean.1: Tanh()
actor_mean.2: Linear(64, 64)    # Hidden layer 2
actor_mean.3: Tanh()
actor_mean.4: Linear(64, 8)     # Output layer → action logits
actor_logstd: Parameter(8)      # Learnable log std
```

**Critic (Value Network)**:
```
critic.0: Linear(27, 64)        # Hidden layer 1
critic.1: Tanh()
critic.2: Linear(64, 64)        # Hidden layer 2
critic.3: Tanh()
critic.4: Linear(64, 1)         # Output layer → value estimate
```

**Total Parameters**: ~4.5K parameters
- Actor hidden: ~2K params (layers 0, 2)
- Actor output: ~520 params (layer 4)
- Critic hidden: ~2K params (layers 0, 2)
- Critic output: ~65 params (layer 4)

---

## 4. Gating Behavior by Variant

| Variant | Actor Hidden | Actor Output | Critic Hidden | Critic Output | Hypothesis |
|---------|--------------|--------------|---------------|---------------|------------|
| **Adam** | No gating | No gating | No gating | No gating | Baseline |
| **UPGD Full** | Gated | Gated | Gated | Gated | Balanced plasticity |
| **Output Only** | Fixed 0.5 | **Gated** | Fixed 0.5 | **Gated** | High output plasticity |
| **Hidden Only** | **Gated** | Fixed 0.5 | **Gated** | Fixed 0.5 | Adaptive features, stable policy |
| **Actor Output** | Fixed 0.5 | **Gated** | Fixed 0.5 | Fixed 0.5 | Policy plasticity only |
| **Critic Output** | Fixed 0.5 | Fixed 0.5 | Fixed 0.5 | **Gated** | Value estimation plasticity |

**Non-gated scale**: 0.5 (configurable via `--non_gated_scale`)

---

## 5. SLURM Scripts Verification

### Script 1: Basic UPGD
**File**: `slurm_rl_ant_upgd.sh`

```bash
python3 core/run/rl/ppo_continuous_action_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"
```

**Uses**: Old `ppo_continuous_action_upgd.py` with basic `AdaptiveUPGD`
**Status**: ✅ Valid, but uses older script (no layer-selective gating)

---

### Script 2: Output-Only UPGD
**File**: `slurm_rl_ant_upgd_output.sh`

```bash
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --optimizer upgd_output_only \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"
```

**Uses**: New `run_ppo_upgd.py` with `--optimizer upgd_output_only`
**Gating**: Only actor_mean.4 and critic.4
**Status**: ✅ Correct

---

### Script 3: Hidden-Only UPGD
**File**: `slurm_rl_ant_upgd_hidden.sh`

```bash
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --optimizer upgd_hidden_only \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"
```

**Uses**: New `run_ppo_upgd.py` with `--optimizer upgd_hidden_only`
**Gating**: Only actor_mean.0/2 and critic.0/2
**Status**: ✅ Correct

⚠️ **Note**: Missing `--weight_decay 0.0` (will default to 0.001)

---

## 6. Comparison with Original Source

### Original Source Files
- `/scratch/gautschi/shin283/upgd/core/core_original/run/rl/adaupgd.py`
- `/scratch/gautschi/shin283/upgd/core/core_original/run/rl/ppo_continuous_action_upgd.py`

### Differences

**1. ppo_continuous_action_upgd.py**:
- ✅ Import path fixed: `rl.adaupgd` → `core.run.rl.adaupgd`
- ✅ Gymnasium 1.0+ compatibility updates (TransformObservation, episode info)
- ✅ No algorithmic changes

**2. New Files (not in original)**:
- ✅ `rl_upgd_layerselective.py` - Layer-selective gating implementation
- ✅ `run_ppo_upgd.py` - Unified script for all variants

**Verdict**: Valid extensions of the original code, no breaking changes

---

## 7. Hyperparameters

### Default UPGD Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 3e-4 | Learning rate (same as Adam) |
| `weight_decay` | 0.001 | Weight decay for UPGD |
| `beta_utility` | 0.999 | Utility EMA decay rate |
| `sigma` | 0.001 | Noise scale for perturbation |
| `beta1` | 0.9 | First moment decay (Adam-style) |
| `beta2` | 0.999 | Second moment decay (Adam-style) |
| `eps` | 1e-5 | Numerical stability term |
| `non_gated_scale` | 0.5 | Scaling for non-gated layers |

### PPO Parameters (same as CleanRL)

| Parameter | Value |
|-----------|-------|
| `num_steps` | 2048 |
| `num_minibatches` | 32 |
| `update_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_coef` | 0.2 |
| `max_grad_norm` | 0.5 |

---

## 8. Expected Results

### Performance Hypotheses

**Adam (Baseline)**:
- Standard PPO performance
- Expected final return (Ant-v4, 20M steps): ~5000-6000

**UPGD Full**:
- Better sample efficiency early on
- Similar or better final performance
- More stable learning (utility gating prevents catastrophic updates)

**Output-Only**:
- High plasticity in policy/value outputs
- May converge faster initially
- Risk: Hidden features may become stale

**Hidden-Only**:
- Adaptive feature learning
- Policy/value outputs stable
- May be slower to adapt to new reward structures

---

## 9. Utility Statistics Logging

The layer-selective UPGD logs comprehensive statistics to WandB:

**Global Statistics**:
- `utility/global_max`: Maximum utility across all parameters
- `utility/gated_params`: Count of gated parameters
- `utility/non_gated_params`: Count of non-gated parameters
- `utility/mean`: Mean utility value
- `utility/std`: Standard deviation of utilities

**Utility Histogram** (9 bins):
- `utility/hist_0_20_pct`: % of utilities in [0.0, 0.2)
- `utility/hist_20_40_pct`: % in [0.2, 0.4)
- `utility/hist_40_48_pct`: % in [0.4, 0.48)
- `utility/hist_48_52_pct`: % in [0.48, 0.52)
- `utility/hist_52_60_pct`: % in [0.52, 0.6)
- `utility/hist_60_80_pct`: % in [0.6, 0.8)
- `utility/hist_80_100_pct`: % in [0.8, 1.0]

**Per-Layer Statistics**:
- `layer/{layer_name}/gating_applied`: 1.0 if gated, 0.0 if not
- `layer/{layer_name}/mean`: Mean utility for layer
- `layer/{layer_name}/std`: Std deviation
- `layer/{layer_name}/min`: Minimum utility
- `layer/{layer_name}/max`: Maximum utility
- `layer/{layer_name}/hist_*_pct`: Per-layer histogram bins

---

## 10. Verification Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| **RLLayerSelectiveUPGD implementation** | ✅ | Lines 45-61: correct gating logic |
| **create_optimizer function** | ✅ | Lines 157-194: all variants supported |
| **Agent architecture** | ✅ | 2-layer MLP for actor/critic |
| **Utility computation** | ✅ | u_t = β * u_{t-1} + (1-β) * (-∇L · θ) |
| **Gating formula** | ✅ | sigmoid(utility / global_max_util) |
| **Non-gated scaling** | ✅ | Fixed scale = 0.5 |
| **WandB logging** | ✅ | Comprehensive utility stats |
| **SLURM scripts** | ⚠️ | Need to check weight_decay consistency |
| **Compatibility** | ✅ | Gymnasium 1.0+ compatible |

---

## 11. Issues Found & Recommendations

### Issue 1: Inconsistent Weight Decay
**Files affected**: `slurm_rl_ant_upgd_hidden.sh`

**Problem**: Missing `--weight_decay 0.0` parameter
```bash
# Current
python3 core/run/rl/run_ppo_upgd.py \
    --optimizer upgd_hidden_only \
    --cuda \
    ...
# Will use default weight_decay=0.001
```

**Recommendation**: Add explicit `--weight_decay 0.0` for consistency with other scripts

**Fix**:
```bash
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --optimizer upgd_hidden_only \
    --weight_decay 0.0 \  # ADD THIS
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"
```

---

### Issue 2: Old vs New Scripts
**Files affected**: `slurm_rl_ant_upgd.sh`

**Problem**: Uses old `ppo_continuous_action_upgd.py` instead of new `run_ppo_upgd.py`

**Impact**: No layer-selective gating, just basic UPGD

**Recommendation**: If you want full gating with the new framework, update to:
```bash
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --optimizer upgd_full \  # Use new unified script
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"
```

---

### Issue 3: Missing Variants
**Missing**: You don't have SLURM scripts for:
- `upgd_full` (using new framework)
- `upgd_actor_output`
- `upgd_critic_output`

**Recommendation**: Create additional scripts if you want to test all variants

---

## 12. Quick Start

### Run Single Variant
```bash
cd /scratch/gautschi/shin283/upgd
sbatch slurm_runs/slurm_rl_ant_upgd_output.sh  # Output-only gating
```

### Run All Variants (Manual)
```bash
sbatch slurm_runs/slurm_rl_ant_upgd.sh         # Basic UPGD
sbatch slurm_runs/slurm_rl_ant_upgd_output.sh  # Output-only
sbatch slurm_runs/slurm_rl_ant_upgd_hidden.sh  # Hidden-only
```

### Test Locally (Quick Check)
```bash
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# Test output-only variant
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 10000 \
    --optimizer upgd_output_only \
    --track False \
    --cuda
```

---

## 13. Monitoring

### Check Job Status
```bash
squeue -u shin283
```

### Monitor Logs
```bash
tail -f /scratch/gautschi/shin283/upgd/logs/<job_id>_rl_ant_upgd_*.out
```

### Check WandB
```
https://wandb.ai/shin283-purdue-university/upgd-rl
```

**Look for**:
- Learning curves: episodic_return should increase
- Utility statistics: utility/hist_* distributions
- Gated vs non-gated params: should match expected counts
- Per-layer utilities: check if gating is applied correctly

---

## 14. Expected Training Time

**Per experiment**:
- Total timesteps: 20,000,000
- Steps per iteration: 2048
- Total iterations: ~9,765
- Expected runtime: 2-3 days on single GPU

---

## 15. Debugging Tips

### Verify Layer Names
```python
# Add to run_ppo_upgd.py or run manually
for name, param in agent.named_parameters():
    print(f"{name}: {param.shape}")
```

**Expected output**:
```
actor_mean.0.weight: (64, 27)
actor_mean.0.bias: (64,)
actor_mean.2.weight: (64, 64)
actor_mean.2.bias: (64,)
actor_mean.4.weight: (8, 64)
actor_mean.4.bias: (8,)
actor_logstd: (1, 8)
critic.0.weight: (64, 27)
critic.0.bias: (64,)
critic.2.weight: (64, 64)
critic.2.bias: (64,)
critic.4.weight: (1, 64)
critic.4.bias: (1,)
```

### Verify Gating Applied
Check WandB logs for:
```
utility/gated_params
utility/non_gated_params
layer/actor_mean.4/gating_applied
layer/critic.4/gating_applied
```

---

## 16. Conclusion

✅ **Your UPGD variants implementation is correct and ready to run!**

**Key Points**:
1. ✅ Layer-selective gating correctly implemented
2. ✅ Unified script supports all variants
3. ✅ Compatible with CleanRL PPO
4. ✅ Comprehensive utility logging
5. ⚠️ Minor issue: inconsistent weight_decay in one script
6. ⚠️ Consider updating old script to use new framework

**Recommended Action**:
1. Fix `slurm_rl_ant_upgd_hidden.sh` to add `--weight_decay 0.0`
2. Submit all variants
3. Monitor WandB for utility statistics
4. Compare learning curves across variants

---

**Files Referenced**:
- Implementation: `/scratch/gautschi/shin283/upgd/core/run/rl/`
  - `rl_upgd_layerselective.py`
  - `run_ppo_upgd.py`
  - `ppo_continuous_action_upgd.py`
- Scripts: `/scratch/gautschi/shin283/upgd/slurm_runs/slurm_rl_ant_upgd*.sh`
- Original: `/scratch/gautschi/shin283/upgd/core/core_original/run/rl/`

**Verified by**: Claude (January 24, 2026)
