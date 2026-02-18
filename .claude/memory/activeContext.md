# Active Context

**Last Updated:** 2026-02-14 (evening session)

## Current Focus
- **A0 supervised seed gap:** Last 2 tasks each for output_only + hidden_only on Gautschi (~5h remaining)
- **A0 UPGD full:** COMPLETED on Gilbreth (10277360 finished)
- **Gridworld Tier 1C:** NOW RUNNING on Gautschi (7780865) — 25 tasks active
- **Humanoid remaining:** NOW RUNNING on Gilbreth (10284522) — 6 tasks active

## Three-Machine Setup
| Machine | Role | Current Jobs |
|---------|------|-------------|
| **Gautschi** (H100) | A0 output/hidden (last 2 each) + Gridworld Tier 1C | 7739971, 7739973, 7780865 |
| **Gilbreth** (A100) | Humanoid remaining (started!) | 10284522 |
| **Studio** (M3 Ultra) | Gridworld Tier 0 (local, completed or near) | — |

## Running Jobs

### Gautschi
- **a0_upgd_output_only** (7739971): tasks 16-17 running (~4h in), **last 2 tasks**
- **a0_upgd_hidden_only** (7739973): tasks 16-17 running (~4h in), **last 2 tasks**
- **gw_tier1c** (7780865): **NEW** — 25 tasks running (20-39), task 4+ pending (MaxCpuPerAccount)
- **humanoid** (7609377): HELD (unchanged)

### Gilbreth
- **humanoid_remaining** (10284522): tasks 0-5 running (5-6h in), rest pending — **now active** (A0 freed GPUs)

## Key Updates This Session
- Discovered A0 full (10277360) completed on Gilbreth
- Gridworld Tier 1C submitted to Gautschi (not in previous memory — submitted between sessions)
- Humanoid remaining now running (was queued behind A0)
- Created study project at ~/projects/study/ with 18-lesson UPGD teaching plan

## A0 Seed Details
- Seeds: 0, 3, 5, 7, 8, 11, 14, 16, 19 (+ existing seed 2 = 10 total per method×dataset)
- Datasets: label_permuted_cifar10_stats, input_permuted_mnist_stats
- Per-dataset hyperparams from optimizer_best_sets.csv (verified correct)

## Next Actions
- A0 output/hidden finish tonight → ALL A0 supervised data ready
- Pull A0 full results from WandB → check if numbers look right
- After A0 complete: re-plot paper figures, update tables
- Monitor Gridworld Tier 1C progress on Gautschi
- Monitor Humanoid remaining on Gilbreth
- After humanoid completes: fill RL section (B1)
