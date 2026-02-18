# Progress Tracking

## Completed

### Supervised Learning Experiments
- Done: Input-permuted MNIST, Label-permuted EMNIST, CIFAR-10, Mini-ImageNet (all methods)
- Done: Layer-selective UPGD variants comparison
- Done: WandB config filtering in plot_paper_figures.py (Feb 13)
- Done: Added all 7 baselines to Table 1 via WandB config-filtered fetch (Feb 13)
- Done: A0 UPGD full — 18/18 tasks on Gilbreth (10277360) — **Feb 14**

### RL Experiments
- Done: Ant-v4 20M timesteps (all methods, 20 seeds)
  - Results: Hidden-only (4843) >> Output-only (3229) — confirms input-shift regime
- Done: Humanoid-v4 test runs (200K) — logger fix validated
- Done: Humanoid Adam baseline on Gilbreth (job 10271209, 20 seeds)
- Done: Humanoid upgd_full on Gautschi (tasks 0-19)
- Done: Grid-world v2 (360/360 tasks on Gautschi, job 7688863)

### Infrastructure
- Done: Fixed logger race condition (exist_ok=True)
- Done: localcontrol workflow + dual-cluster monitoring
- Done: Cross-machine sync (Studio ↔ Pro via Tailscale)
- Done: Fixed hardcoded log path for cross-cluster compatibility
- Done: Local gridworld venv on Studio (`~/venvs/gridworld/`)
- Done: Study project setup at ~/projects/study/ with 18-lesson teaching plan (Feb 14)

## In Progress

### A0: UPGD Seed Gap Fix (CRITICAL for paper)
- **Gilbreth: a0_upgd_full (10277360) — COMPLETED**
- Gautschi: a0_upgd_output_only (7739971) — tasks 16-17 running, **finishing tonight**
- Gautschi: a0_upgd_hidden_only (7739973) — tasks 16-17 running, **finishing tonight**
- **Status:** ~90% complete, all done by Feb 15 morning

### Gridworld Tier 1C (on Gautschi)
- Job 7780865: 25 tasks running, more pending
- WITH WandB tracking (upgd-gridworld project)
- Tier 0 gate presumably passed (job was submitted)

### Humanoid-v4 Remaining (on Gilbreth)
- Job 10284522: tasks 0-5 running (started after A0 freed GPUs)
- 36 total tasks (upgd_output_only + upgd_hidden_only seeds)
- Gautschi job 7609377 still HELD (backup)

## Planned

### After A0 Completion (tonight/tomorrow)
- Re-run plot_paper_figures.py --cache to regenerate figures
- Update paper_body_v2.md table + prose if numbers shift
- Verify 10 seeds per UPGD method × dataset in WandB

### After Humanoid Completion
- B1: Fill existing Ant/Humanoid RL results into paper
- Generate learning curves and comparison plots
- Show advisor — decision point on whether more RL needed

### After Gridworld Tier 1C
- If results show clean regime separation → strongest RL contribution
- If not → pivot to MuJoCo-only RL story

## Cancelled/Resolved
- 7 rerun configs (rerun_sgd/adam/si/snp/upgd_*) — cancelled, CSV params wrong for baselines
- Grid-world Tier 0 on Gautschi (7741158) — cancelled, running locally instead
- Humanoid tasks 24-31 on Gautschi — cancelled (accidentally released)
