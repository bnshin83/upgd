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
- Done: Humanoid-v4 20M (adam, upgd_full, hidden_only, output_only × 20 seeds)
- Done: Walker2d-v4 20M (adam, upgd_full, hidden_only, output_only × 20 seeds)
- Done: HumanoidStandup-v4 20M (adam, upgd_full, hidden_only, output_only × 20 seeds)
- Done: Grid-world v2 (360/360 tasks)
- Done: Grid-world v4 50K comparison (dynamics/reward/joint × 4 methods × 10 seeds)
- Done: Grid-world v4 100K layer gradient (7 methods × 10 seeds, reward_shift)
- Done: Grid-world v4 layer gradient fix (L1/L2/L2L3 rerun)
- Done: Ant-v4 phase-adaptive batch 1+2 (all schedules complete)
- Done: SlipperyAnt full ablation (std, l2, cbp, cbp_h1/h2/no_gnt, upgd_full, reset/shrink_head, combos)

### Infrastructure
- Done: Fixed logger race condition, localcontrol workflow, cross-machine sync
- Done: Local gridworld venv on Studio
- Done: Fixed hardcoded log path for cross-cluster compatibility
- Done: Added `.upgd` to rsync exclude to prevent cluster venv corruption
- Done: Study project setup at ~/projects/study/ with 18-lesson teaching plan (Feb 14)

## In Progress

### Results Collection & Analysis
- Need to pull all Walker2d + HumanoidStandup 20M results from WandB
- Need final 4-method plots for both environments
- 40M rerun decision pending (plan at `plan/checkpoint_resume_40m.md`)

### UPGD Study/Teaching Plan
- 18 lessons written at ~/projects/study/plan/lessons/
- Lesson 01 (iterator protocol) started interactively

## Planned

### 40M Reruns (if needed)
- Implement checkpoint save/resume
- Rerun all 4 methods × 20 seeds at 40M for Walker2d + HumanoidStandup

### Paper
- B1: Fill RL results into paper with final plots
- Generate learning curves and comparison plots
- Show advisor — decision point on whether more RL needed

## Cancelled/Resolved
- Grid-world Tier 0 on Gautschi (7741158) — cancelled, ran locally
- ant_40m_phase_adaptive_resub first attempt (7883597) — cancelled, wrong array range
- ant_40m_phase batch 2 original (7856803 tasks 24-49) — crashed, torch_shm_manager missing
