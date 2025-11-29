# UPGD Learner Workflow Guidelines

## Overview
This document outlines how to run experiments with different learners in the UPGD framework, with proper WandB logging and JSON result storage. **Updated to include Charts tab fix and enhanced WandB visualization for all learners.**

## Available Learners

### Traditional Optimizers
- `sgd` - Stochastic Gradient Descent
- `adam` - Adam optimizer  
- `pgd` - Projected Gradient Descent

### Continual Learning Methods
- `ewc` - Elastic Weight Consolidation
- `mas` - Memory Aware Synapses
- `si` - Synaptic Intelligence
- `rwalk` - Random Walk

### UPGD Variants
- `upgd_fo_global` - First Order Global UPGD
- `upgd_so_global` - Second Order Global UPGD
- `upgd_fo_local` - First Order Local UPGD
- `upgd_so_local` - Second Order Local UPGD
- `upgd_nonprotecting_fo_global` - Non-protecting First Order Global
- `upgd_nonprotecting_so_global` - Non-protecting Second Order Global

### Feature UPGD Variants
- `feature_upgd_fo_global` - Feature First Order Global
- `feature_upgd_so_global` - Feature Second Order Global
- (+ local and non-protecting variants)

### Input-Aware UPGD (Special)
- `upgd_input_aware_fo_global` - Input-Aware First Order Global
- `upgd_input_aware_so_global` - Input-Aware Second Order Global

## Workflow Types

### 1. Enhanced Unified Workflow (Recommended for ALL Learners)
**Script**: Use `run_stats_with_curvature.py` 
**Features**: 
- **Full WandB integration with Charts tab** (9 tabs total)
- Enhanced frequent logging (every 5-10 steps)
- Incremental JSON updates every 1000 steps
- Early folder/file creation for progress monitoring
- **Consistent visualization across all learners**
- Moving average tracking (EMA accuracy/plasticity)
- Organized metric categories: `training/`, `plasticity/`, `network/`, `weights/`, `gradients/`, `curvature/`, `task_level/`, `summary/`

### 2. Legacy Standard Learners (Deprecated)
**Script**: Use `run_stats.py` 
**Features**: 
- Basic training metrics (loss, accuracy, plasticity)
- Network health metrics (dead units, weight/gradient norms)
- Standard JSON logging
- No WandB by default
- **Missing Charts tab** in WandB

**Enhanced Template Script Structure (Use for ALL learners)**:
```bash
#!/bin/bash
#SBATCH --job-name=learner_task_samples_seed
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_learner_task_samples_seed.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_learner_task_samples_seed.err

cd /scratch/gautschi/shin283/upgd
module load cuda python
source .upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# WandB Configuration (enables Charts tab)
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_LEARNER_TASK_params_samples_N_SAMPLES_seed_SEED"
export WANDB_MODE="online"

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task TASK_NAME \
    --learner LEARNER_NAME \
    --seed SEED \
    --lr LEARNING_RATE \
    [LEARNER_SPECIFIC_PARAMS] \
    --network NETWORK_NAME \
    --n_samples N_SAMPLES \
    --compute_curvature_every N_SAMPLES \
    --save_path logs
```

**Legacy Script Structure (Not recommended)**:
```bash
# Same SLURM headers as above, but using:
python3 core/run/run_stats.py \
    --task TASK_NAME \
    --learner LEARNER_NAME \
    --seed SEED \
    --lr LEARNING_RATE \
    [LEARNER_SPECIFIC_PARAMS] \
    --network NETWORK_NAME \
    --n_samples N_SAMPLES
```

### 3. Input-Aware Learners (Special Parameters)
**Script**: Use `run_stats_with_curvature.py` (same as enhanced unified workflow)
**Features**: 
- All enhanced features PLUS real curvature tracking
- **Identical WandB Charts tab structure** as standard learners
- Additional curvature-specific parameters required
- Real curvature computation (vs placeholder values for standard learners)

**Template Script Structure**:
```bash
#!/bin/bash
#SBATCH --job-name=learner_task_curv_threshold_samples_seed
[... standard SLURM headers ...]

# WandB Configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_learner_task_params"
export WANDB_MODE="online"

python3 core/run/run_stats_with_curvature.py \
    --task TASK_NAME \
    --learner upgd_input_aware_fo_global \
    --seed SEED \
    --lr LEARNING_RATE \
    --sigma SIGMA \
    --network NETWORK_NAME \
    --n_samples N_SAMPLES \
    --curvature_threshold THRESHOLD \
    --lambda_max LAMBDA_MAX \
    --hutchinson_samples HUTCHINSON_SAMPLES \
    --compute_curvature_every FREQUENCY \
    --save_path logs
```

## WandB Charts Tab Fix Applied

**The Charts tab issue has been resolved!** All learners now have consistent WandB visualization when using the enhanced workflow.

### What was fixed:
- **Enhanced frequent logging**: Standard learners now log metrics every 5 steps (matching input-aware frequency)
- **Structured metric categories**: All learners use organized prefixes (`training/`, `plasticity/`, `network/`, `weights/`, `gradients/`, `curvature/`)
- **Moving averages**: EMA accuracy and plasticity tracking for all learners
- **REAL curvature computation**: ALL learners now compute actual input curvature for analysis (not just placeholders)
- **Curvature usage distinction**: Input-aware learners use curvature for updates, standard learners compute it only for analysis

### Charts Tab Structure (All Learners):
1. **Overview** - Basic run information
2. **Training** - Loss, accuracy, progress metrics
3. **Plasticity** - Current and averaged plasticity tracking
4. **Network** - Dead units ratio, network health
5. **Weights** - L1/L2 norms, rank ratios
6. **Gradients** - L1/L2 norms, sparsity ratios
7. **Curvature** - Input curvature values (REAL for ALL learners - used for updates by input-aware, analysis-only for standard)
8. **Task Level** - Task-by-task summaries
9. **Summary** - Final experiment statistics

### Legacy Manual WandB Setup (Not recommended):
For `run_stats.py` users who want to add WandB manually:

```python
# Add after imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add in __init__
if WANDB_AVAILABLE and os.environ.get('WANDB_PROJECT'):
    wandb.init(project=os.environ.get('WANDB_PROJECT'), ...)
    self.wandb_enabled = True

# Add in training loop (minimal logging - missing Charts tab)
if self.wandb_enabled and i % 10 == 0:
    wandb.log({'loss': loss.item(), 'step': i}, step=i)
```

## Learner-Specific Parameters

### SGD
- `lr` (learning rate)
- `weight_decay` (optional)

### Adam  
- `lr` (learning rate)
- `weight_decay`, `beta1`, `beta2`, `eps` (optional)

### UPGD Variants
- `lr` (learning rate)
- `sigma` (noise parameter)

### EWC/MAS/SI
- `lr` (learning rate)
- `beta_weight`, `beta_fisher`, `lamda` (regularization params)

### Input-Aware UPGD
- `lr`, `sigma` (standard UPGD params)
- `curvature_threshold` (curvature threshold for updates)
- `lambda_max` (maximum lambda value)
- `hutchinson_samples` (samples for curvature estimation)
- `compute_curvature_every` (frequency of curvature computation)

## JSON Output Structure

All experiments save results to:
```
logs/TASK/LEARNER/NETWORK/HYPERPARAMS/SEED.json
```

With n_samples modification (after updating Logger):
```
logs/TASK/LEARNER/NETWORK/HYPERPARAMS_n_samples_N/SEED.json
```

### Standard JSON Contents:
- `losses` (per task)
- `plasticity_per_task`
- `n_dead_units_per_task`
- `weight_rank_per_task`, `weight_l2_per_task`, `weight_l1_per_task`
- `grad_l2_per_task`, `grad_l1_per_task`, `grad_l0_per_task`
- `accuracies` (for classification tasks)
- Metadata: `task`, `learner`, `network`, `optimizer_hps`, `n_samples`, `seed`

### Input-Aware Additional Contents:
- `input_curvature_per_task`, `lambda_values_per_task`
- `avg_curvature_per_task`, `curvature_max/min/std_per_task`
- `input_curvature_per_step`, `lambda_values_per_step` (full step data)
- `compute_curvature_every` (configuration)
- `status` ("in_progress" or "completed")
- `current_step`, `progress_percent` (for incremental updates)

## Example Scripts for Different Learners (Enhanced Workflow)

### Adam with Full WandB Integration
```bash
#!/bin/bash
#SBATCH --job-name=adam_input_mnist_samples_1000000_seed_0
# ... standard SLURM headers ...

export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_adam_input_mnist_lr_0.001_samples_1000000_seed_0"
export WANDB_MODE="online"

python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner adam \
    --seed 0 \
    --lr 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs
```

### UPGD FO Global with Charts Tab
```bash
#!/bin/bash
#SBATCH --job-name=upgd_fo_global_input_mnist_samples_1000000_seed_0
# ... standard SLURM headers ...

export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_fo_global_input_mnist_lr_0.01_sigma_0.001_samples_1000000_seed_0"
export WANDB_MODE="online"

python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs
```

### EWC with Full WandB Visualization
```bash
#!/bin/bash
#SBATCH --job-name=ewc_emnist_samples_1000000_seed_0
# ... standard SLURM headers ...

export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_ewc_emnist_lambda_1.0_samples_1000000_seed_0"
export WANDB_MODE="online"

python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_emnist_stats \
    --learner ewc \
    --seed 0 \
    --lr 0.01 \
    --beta_weight 0.9999 \
    --beta_fisher 0.9999 \
    --lamda 1.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs
```

### Input-Aware UPGD (Real Curvature Computation)
```bash
#!/bin/bash
#SBATCH --job-name=upgd_input_aware_input_mnist_curv_0.05_samples_800_seed_0
# ... standard SLURM headers ...

export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_input_aware_input_mnist_curv_0.05_samples_800_seed_0"
export WANDB_MODE="online"

python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_input_aware_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 800 \
    --curvature_threshold 0.05 \
    --lambda_max 1.0 \
    --hutchinson_samples 5 \
    --compute_curvature_every 1 \
    --save_path logs
```

## Key Differences Summary (Updated)

| Feature | Legacy Workflow | Enhanced Unified Workflow |
|---------|-----------------|---------------------------|
| Script | `run_stats.py` | `run_stats_with_curvature.py` |
| WandB Charts Tab | **Missing** | **9 tabs with full visualization** |
| Logging Frequency | End-of-experiment only | Every 5-10 steps + every 1000 steps + end |
| Curvature Data | No | Yes (REAL curvature computed for ALL learners) |
| Folder Creation | End-of-experiment | Immediate |
| Progress Monitoring | Limited | Real-time via JSON/WandB |
| Moving Averages | No | EMA accuracy/plasticity tracking |
| Metric Categories | Basic | Organized (`training/`, `plasticity/`, etc.) |

## Updated Recommendations

1. **✅ Use Enhanced Workflow for ALL learners**: `run_stats_with_curvature.py` provides Charts tab and better monitoring
2. **✅ Set `--compute_curvature_every N_SAMPLES`**: For standard learners, set this to total samples to avoid computation overhead
3. **✅ Use consistent WandB project**: `"upgd-input-aware"` for easy comparison across all learners
4. **✅ Include n_samples in job names**: `{learner}_{task}_samples_{n_samples}_seed_{seed}`
5. **✅ Monitor via WandB Charts tab**: All 9 tabs now available for every learner type
6. **⚠️ Legacy workflow deprecated**: Only use `run_stats.py` for minimal quick tests

## Migration Guide

To upgrade existing scripts from legacy to enhanced workflow:

```bash
# Replace this:
python3 core/run/run_stats.py \
    --task TASK --learner LEARNER --seed SEED \
    --lr LR [PARAMS] --network NETWORK --n_samples N

# With this:
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_LEARNER_TASK_params_samples_N_seed_SEED"
export WANDB_MODE="online"

python3 core/run/run_stats_with_curvature.py \
    --task TASK --learner LEARNER --seed SEED \
    --lr LR [PARAMS] --network NETWORK --n_samples N \
    --compute_curvature_every N --save_path logs
```

This provides **identical Charts tab structure** for all learners with enhanced monitoring capabilities.

## Input Curvature Analysis for All Learners

### Universal Curvature Computation
**NEW FEATURE**: All learners now compute real input curvature for analysis, regardless of whether they use it for parameter updates.

### How it works:
1. **Input-Aware Learners** (`upgd_input_aware_fo_global`):
   - Compute curvature using learner-specific method (Hutchinson trace estimation)
   - **Use curvature for parameter updates** (affects optimization)
   - Log curvature with corresponding lambda values
   - Example: `input_curvature: 0.000843`, `lambda_value: 0.379612`

2. **Standard Learners** (SGD, Adam, UPGD variants, EWC, etc.):
   - Compute curvature using universal finite differences method
   - **Do NOT use curvature for updates** (analysis only)
   - Lambda values remain 0.0 (not applicable)  
   - Example: `input_curvature: 0.002156`, `lambda_value: 0.0`

### Benefits:
- **Cross-learner comparison**: Compare how different optimizers navigate the loss landscape
- **Curvature analysis**: Understand input sensitivity across all optimization methods  
- **Research insights**: Analyze whether high-curvature regions correlate with learning difficulties
- **Consistent visualization**: All learners show meaningful curvature charts

### Configuration:
```bash
# Compute curvature every 10 steps for analysis
--compute_curvature_every 10

# Compute curvature every 1000 steps (less overhead)  
--compute_curvature_every 1000

# Only at the end (minimal overhead)
--compute_curvature_every N_SAMPLES
```

### Performance Impact:
- **Minimal overhead**: Universal method uses efficient Hutchinson estimation (5 random vectors)
- **Configurable frequency**: Adjust `compute_curvature_every` to balance analysis vs speed
- **No optimizer changes**: Standard learners maintain their original update rules