# Progress Tracking

## Completed

### Supervised Learning Experiments
- Done: Input-permuted MNIST, Label-permuted EMNIST, CIFAR-10, Mini-ImageNet (all methods)
- Done: Layer-selective UPGD variants comparison
- Done: A0 seed gap fix — all 3 UPGD methods × 2 datasets × 9 seeds complete
- Done: WandB config filtering in plot_paper_figures.py

### RL Experiments
- Done: Ant-v4 20M timesteps (all methods, 20+ seeds)
  - Results: Hidden-only (4843) >> Output-only (3229) — confirms input-shift regime
- Done: Humanoid-v4 test runs, Adam baseline (20 seeds), upgd_full (20 seeds)
- Done: Grid-world v2 (360/360 tasks)
- Done: Grid-world v4 50K comparison (dynamics/reward/joint × 4 methods × 10 seeds)
- Done: Grid-world v4 100K layer gradient (7 methods × 10 seeds, reward_shift)
- Done: Grid-world v4 layer gradient fix (L1/L2/L2L3 rerun)
- Done: Ant-v4 phase-adaptive batch 1 (tasks 0-23, schedules hhh/fff/ooo/adam + partial hhf)

### Infrastructure
- Done: Fixed logger race condition, localcontrol workflow, cross-machine sync
- Done: Local gridworld venv on Studio
- Done: Fixed hardcoded log path for cross-cluster compatibility
- Done: Added `.upgd` to rsync exclude to prevent cluster venv corruption

## In Progress

### Ant-v4 Phase-Adaptive Gating (40M steps)
- **Batch 1** (7856803): COMPLETED — 24/24 tasks (schedules 1-4 + partial 5)
- **Batch 2** (7883799): 24/26 running on Gautschi — ETA Wed Feb 19 morning
  - Covers schedules 5-10: hhf(seed5), hff, hfh, fhh, hho, ffh
- Code: `set_gating_mode()` in optimizer, `gating_schedule` in PPO runner
- Script: `.localcontrol/experiments/rl/ant_40m_phase_adaptive.sh`

### Humanoid-v4 Remaining (on Gilbreth)
- Job 10284522: 6 running (tasks 36-41), 18 pending (42-59)
- Methods: upgd_output_only + upgd_hidden_only
- ETA: ~Fri Feb 21

## Planned

### After Phase-Adaptive Completion (~Wed)
- Analyze 50 runs on WandB: compare adaptive schedules vs uniform baselines
- Key question: Does hhf or hff beat hhh at 40M?
- If promising: extend top-3 schedules to 10 seeds

### After Humanoid Completion (~Fri)
- B1: Fill RL results into paper
- Generate learning curves and comparison plots

### SlipperyAnt (deferred)
- Phase 1: Fix UPGD (sigma=0.001, remove 1e-8 clamp, port optimizer)
- Phase 2: Phase-adaptive gating (after Phase 1 succeeds)

## Cancelled/Resolved
- Grid-world Tier 0 on Gautschi (7741158) — cancelled, ran locally
- ant_40m_phase_adaptive_resub first attempt (7883597) — cancelled, wrong array range (CUSTOM_ARRAY vs ARRAY_RANGE)
- ant_40m_phase batch 2 original (7856803 tasks 24-49) — crashed in 6s due to torch_shm_manager missing
