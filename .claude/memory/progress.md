# Progress Tracking

## Completed ✓

### Supervised Learning Experiments
- ✓ Input-permuted MNIST (all methods)
- ✓ Label-permuted EMNIST (all methods)
- ✓ CIFAR-10 experiments
- ✓ Mini-ImageNet experiments
- ✓ Layer-selective UPGD variants comparison

### RL Experiments
- ✓ Ant-v4 20M timesteps (all methods, 20 seeds)
  - Results: Hidden-only (4843) >> Output-only (3229) - confirms input-shift regime
- ✓ Humanoid-v4 test runs (200K timesteps) - logger fix validated

### Infrastructure
- ✓ Fixed logger race condition (exist_ok=True)
- ✓ Set up localcontrol workflow
- ✓ Created dual-cluster monitoring
- ✓ Committed configs to git for cross-machine work

## In Progress ⏳

### Humanoid-v4 Full Experiment (20M timesteps, 20 seeds per method)
- ⏳ Gautschi: upgd_full, upgd_output_only, upgd_hidden_only (60 tasks)
  - Job 7609377: 8/60 running as of 2026-02-09 9:15 AM
- ⏳ Gilbreth: adam baseline (20 tasks, 2 seeds parallel)
  - Job 10269468: Queued, starts ~12:30 PM EST
- **Expected completion:** ~3.75 days (2026-02-12/13)

## Planned 📋

### After Humanoid-v4 Completion
- 📋 Extract final episodic returns from WandB (80 runs)
- 📋 Statistical analysis (means, stdev, t-tests)
- 📋 Compare with Ant-v4 results
- 📋 Validate regime hypothesis on Humanoid
- 📋 Generate learning curves and comparison plots
- 📋 Update paper with findings

### Future Work (Not Critical)
- 📋 Walker2d-v4 experiments
- 📋 HumanoidStandup-v4 experiments
- 📋 Additional environments for generalization

## Blocked/Issues ⚠️
- None currently
