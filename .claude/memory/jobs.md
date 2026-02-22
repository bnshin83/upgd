# Cluster Job Tracking

## Active Jobs

### Gautschi — SlipperyAnt s4-10 (partial)
- **Job ID:** 8073259
- **Config:** `.localcontrol/experiments/rl/sant_val_s4_10.sh`
- **Tasks:** 6 methods × 7 seeds = 42
- **Status:**
  - Tasks 0-5 (std s4-10): completed earlier
  - Tasks 6-13 (std s8-10 + l2 s4-10): **CANCELED** (freed slots)
  - Tasks 14-17 (cbp s4-7): **RUNNING**
  - Tasks 18-41: **HELD** (cbp s8-10 + all UPGD variants)

### Gautschi — CBP Layer-Selective Ablation
- **Job ID:** 8081884
- **Submitted:** 2026-02-22
- **Config:** `.localcontrol/experiments/rl/sant_val_cbp_layers.sh`
- **Tasks:** cbp_h1, cbp_h2, cbp_no_gnt × 3 seeds = 9
- **Status:** 9/9 RUNNING
- **WandB:** sant__cbp_h1, sant__cbp_h2, sant__cbp_no_gnt

### Gautschi — UPGD Full (wd=1e-4)
- **Job ID:** 8081956
- **Submitted:** 2026-02-22
- **Config:** `.localcontrol/experiments/rl/sant_val_upgd_wd4.sh`
- **Tasks:** upgd_full (wd=1e-4) × 3 seeds = 3
- **Status:** 3/3 RUNNING
- **WandB:** sant__upgd_full (tag: wd1e-4)

### Gautschi — Head Intervention
- **Job ID:** 8082254
- **Submitted:** 2026-02-22
- **Config:** `.localcontrol/experiments/rl/sant_val_reset_head.sh`
- **Tasks:** reset_head + shrink_head × 3 seeds = 6
- **Status:** RUNNING/PENDING
- **WandB:** sant__reset_head, sant__shrink_head

## Completed Jobs

| Job | Cluster | Job ID | Result |
|-----|---------|--------|--------|
| Ant-v4 phase-adaptive batch 1 | Gautschi | 7856803 (0-23) | 24/24 complete, 20-25h each |
| Ant-v4 phase-adaptive batch 2 | Gautschi | 7883799 (24-49) | 26/26 complete |
| A0 UPGD full | Gilbreth | 10277360 | 18/18 complete |
| A0 UPGD output_only | Gautschi | 7739971 | 18/18 complete |
| A0 UPGD hidden_only | Gautschi | 7739973 | 18/18 complete |
| Ant-v4 20M | Gautschi | — | Hidden (4843) >> Output (3229) |
| Humanoid Adam | Gilbreth | 10271209 | 20 seeds complete |
| Humanoid upgd_full | Gautschi | 7609377 (0-19) | 20 seeds complete |
| Humanoid remaining | Gilbreth | 10284522 | Complete |
| Grid-world v2 | Gautschi | 7688863 | 360/360 complete |
| Grid-world v4 50K | Gautschi | 7791533-38 | 240 runs complete |
| Grid-world v4 layer gradient | Gautschi | 7801663 | 70 runs complete |
| Grid-world v4 layer gradient fix | Gautschi | 7835134 | 30 runs complete |
| SlipperyAnt validate s1-2 | Gautschi | — | cbp > l2 > upgd_full > std > hidden ≈ output |

## Failed/Canceled

| Job ID | Issue | Fix |
|--------|-------|-----|
| 8081852 | Canceled to add cbp_no_gnt | Resubmitted as 8081884; deleted 6 orphaned WandB runs |
| 8082032 | reset_head with full reinit collapsed at 2M | Changed to optimizer-state reset; resubmitted as 8082254 |

## Lessons Learned
- `CUSTOM_ARRAY` only works for `EXP_TYPE="custom"` in localcontrol; use `ARRAY_RANGE` for RL
- Add venv dirs (.upgd) to RSYNC_EXCLUDE to prevent accidental corruption
- Full output layer reinit in PPO → ratio explosion → collapse. Use shrink-and-perturb instead.
- Always clean up orphaned WandB runs after canceling jobs (same group/tag pollutes data)
