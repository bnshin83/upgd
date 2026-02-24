# Active Context

**Last Updated:** 2026-02-23 (Tue 1am)

## Current Focus
- **20-seed RL production runs:** Walker2d nearly done, HumanoidStandup split across both clusters
- **40M rerun planned:** 20M insufficient — plan at `plan/checkpoint_resume_40m.md`
- **SlipperyAnt ablation:** All combo methods completed

## Active Jobs

### Gautschi (H100)
| Job | Env | Tasks | Method | Status | ETA |
|-----|-----|-------|--------|--------|-----|
| 8103978 | Walker2d | 64-79 running | output_only | Running (~4h in) | Tue ~6-7am |
| 8107499 | HumanoidStandup | 42-49 running, 50-59 pending | hidden_only | Running | Tue evening |

### Gilbreth (A100)
| Job | Env | Tasks | Method | Status | ETA |
|-----|-----|-------|--------|--------|-----|
| 10330476 | HumanoidStandup | 60-65 running, 66-79 pending | output_only | Running (6 concurrent) | Wed morning |

*Walker2d hidden_only: 18/20 completed, 2 finishing. Output_only: all 20 running.*
*HumanoidStandup output_only moved from Gautschi → Gilbreth to parallelize.*
*Gilbreth 10320607 (upgd_full) + 10329294 (hidden_only 0-1): ALL COMPLETED.*

## Completed Runs

### Walker2d-v4 (Gautschi) — 20 seeds each
| Method | Seeds | Status | Notes |
|--------|-------|--------|-------|
| adam | 20 | Done | Full 20M |
| upgd_full | 20 | Done | Full 20M |
| upgd_hidden_only | 18 done, 2 finishing | ~Done | 19.9M plotted |
| upgd_output_only | 0 done, 20 running | Running | 8M plotted, ETA Tue ~6-7am |

**Plot:** `results/walker2d_4methods_partial.png` — hidden_only ≈ full > Adam > output_only (early)

### HumanoidStandup-v4 — 20 seeds each
| Method | Seeds | Status | Notes |
|--------|-------|--------|-------|
| adam | 20 | Done | Full 20M, on Gilbreth |
| upgd_full | 20 | Done | Full 20M (16 complete + 4 just finished on Gilbreth) |
| upgd_hidden_only | 2 done (Gilbreth) + 8 running (Gautschi) | Running | 10 seeds at 3M plotted, ETA Tue evening |
| upgd_output_only | 0 done, 6 running + 14 pending (Gilbreth) | Running | Just started, ETA Wed morning |

**Plot:** `results/humanoidstandup_4methods_partial.png` — hidden_only tracking near full (early), output_only not plotted yet

### SlipperyAnt Ablation (all completed)
- Ranking: cbp ≈ shrink_head > cbp_h1 > cbp_h2 > cbp_no_gnt > upgd_full > std
- Combo methods done: cbp_shrink, cbp_fast, cbp_h1_shrink, cbp_h1_full

### HalfCheetah-v4 (Gilbreth) — status unknown, not checked recently

## Key Findings
- Walker2d: hidden_only ≈ full >> Adam; output_only below Adam at 8M (still early)
- HumanoidStandup: full > Adam clearly; hidden_only tracking near full early on
- 20M steps likely insufficient — planning 40M rerun with checkpoint saving

## Next Actions
- Implement checkpoint save/resume (`plan/checkpoint_resume_40m.md`) after current runs finish
- Rerun all 4 methods × 20 seeds at 40M for Walker2d + HumanoidStandup
- Check HalfCheetah status on Gilbreth
- Final 4-method plots when all 20M runs complete (Wed)
