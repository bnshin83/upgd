# Cluster Job Tracking

## Active Jobs

### Gautschi — A0 UPGD Output-Only
- **Job ID:** 7739971
- **Submitted:** 2026-02-14
- **Config:** `.localcontrol/experiments/supervised/a0_upgd_output_only.sh`
- **Tasks:** 18 (9 seeds × 2 datasets), 8 concurrent
- **Seeds:** 0,3,5,7,8,11,14,16,19 (via PLAN_SEEDS remapping)
- **Datasets:** CIFAR-10 (lr=0.01, σ=0.001, β=0.999, wd=0.0), Input-MNIST (lr=0.01, σ=0.1, β=0.9999, wd=0.01)
- **Runtime:** ~9h per task
- **Status:** Tasks 16-17 running (~4h in as of evening Feb 14), **finishing tonight**
- **WandB:** shin283-purdue-university/upgd

### Gautschi — A0 UPGD Hidden-Only
- **Job ID:** 7739973
- **Config:** `.localcontrol/experiments/supervised/a0_upgd_hidden_only.sh`
- **Tasks:** 18, 8 concurrent
- **Same seeds/datasets as above**
- **Status:** Tasks 16-17 running, **finishing tonight**
- **WandB:** shin283-purdue-university/upgd

### Gautschi — Gridworld Tier 1C (NEW)
- **Job ID:** 7780865
- **Status:** 25 tasks running (20-39), task 4+ pending (MaxCpuPerAccount)
- **Submitted:** Between sessions (not by this Claude instance)
- **WandB:** shin283-purdue-university/upgd-gridworld

### Gautschi — Humanoid UPGD (HELD)
- **Job ID:** 7609377
- **Status:** HELD (JobHeldUser)
- **Tasks 0-23:** Completed (upgd_full done, output_only tasks 20-23 done)
- **Tasks 24-31:** Cancelled (accidentally released)
- **Tasks 32-59:** Held (not started)
- **Note:** Remaining tasks covered by Gilbreth job 10284522 instead

### Gilbreth — Humanoid Remaining
- **Job ID:** 10284522
- **Config:** `.localcontrol/experiments/rl/humanoid_gilbreth_remaining.sh`
- **Tasks:** 36 (array 24-59, %6 concurrent)
- **Methods:** upgd_full (unused, already done), upgd_output_only (24-39), upgd_hidden_only (40-59)
- **Status:** NOW RUNNING — tasks 0-5 active (5-6h in as of evening Feb 14)
- **WandB:** shin283-purdue-university/upgd-rl

## Completed Jobs
| Job | Cluster | Job ID | Result |
|-----|---------|--------|--------|
| A0 UPGD full | Gilbreth | 10277360 | **18/18 tasks complete (Feb 14)** |
| Ant-v4 full | Gautschi | — | Hidden-only (4843) >> Output-only (3229) |
| Humanoid Adam | Gilbreth | 10271209 | 20 seeds complete |
| Humanoid upgd_full | Gautschi | 7609377 (0-19) | 20 seeds complete |
| Grid-world v2 | Gautschi | 7688863 | 360/360 tasks complete |
| Humanoid test | Gautschi | 7608913 | Logger fix validated |

## Cancelled Jobs
| Job ID | Cluster | Reason |
|--------|---------|--------|
| 10276337 | Gilbreth | rerun_sgd — CSV params don't match WandB |
| 10276338 | Gilbreth | rerun_adam — same |
| 10276339 | Gilbreth | rerun_si — same |
| 10276341 | Gilbreth | rerun_snp — same |
| 10276342 | Gilbreth | rerun_upgd_full — covered by A0 |
| 10276343 | Gilbreth | rerun_upgd_output_only — covered by A0 |
| 10276344 | Gilbreth | rerun_upgd_hidden_only — covered by A0 |
| 7741158 | Gautschi | gridworld_tier0 — running locally instead |

## Monitoring Commands
```bash
export PATH="$HOME/projects/localcontrol/bin:$PATH"
lc-status gautschi
lc-status gilbreth
lc-logs gautschi 7739971   # A0 output_only
lc-logs gautschi 7739973   # A0 hidden_only
lc-logs gautschi 7780865   # Gridworld Tier 1C
lc-logs gilbreth 10284522  # Humanoid remaining
```
