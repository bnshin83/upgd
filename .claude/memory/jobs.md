# Cluster Job Tracking

## Active Jobs

**None — both clusters idle as of 2026-02-26.**

## Completed Jobs

| Job | Cluster | Job ID | Result |
|-----|---------|--------|--------|
| Walker2d adam/full/hidden/output × 20 seeds | Gautschi | 8103978 etc. | 20M complete, hidden ≈ full >> Adam |
| HumanoidStandup adam × 20 seeds | Gilbreth | — | 20M complete |
| HumanoidStandup upgd_full × 20 seeds | Gilbreth | 10320607 | 20M complete |
| HumanoidStandup hidden_only × 20 seeds | Gautschi+Gilbreth | 8107499, 10329294 | 20M complete |
| HumanoidStandup output_only × 20 seeds | Gilbreth | 10330476 | 20M complete |
| Ant-v4 phase-adaptive batch 1 | Gautschi | 7856803 (0-23) | 24/24 complete |
| Ant-v4 phase-adaptive batch 2 | Gautschi | 7883799 (24-49) | 26/26 complete |
| A0 UPGD full | Gilbreth | 10277360 | 18/18 complete |
| A0 UPGD output_only | Gautschi | 7739971 | 18/18 complete |
| A0 UPGD hidden_only | Gautschi | 7739973 | 18/18 complete |
| Ant-v4 20M | Gautschi | — | Hidden (4843) >> Output (3229) |
| Humanoid-v4 Adam + upgd_full | Mixed | 10271209, 7609377 | 20 seeds each |
| Humanoid-v4 remaining | Gilbreth | 10284522 | Complete |
| Grid-world v2 | Gautschi | 7688863 | 360/360 complete |
| Grid-world v4 50K | Gautschi | 7791533-38 | 240 runs complete |
| Grid-world v4 layer gradient | Gautschi | 7801663 | 70 runs complete |
| Grid-world v4 layer gradient fix | Gautschi | 7835134 | 30 runs complete |
| SlipperyAnt validate s1-2 | Gautschi | — | cbp > l2 > upgd_full > std |
| SlipperyAnt ablation (all methods) | Gautschi | 8073259, 8081884, 8081956, 8082254 | All complete |
| SlipperyAnt combos (cbp_shrink, cbp_fast, etc.) | Gautschi | 8088188, 8090346, 8090377, 8092578 | All complete |

## Failed/Canceled

| Job ID | Issue | Fix |
|--------|-------|-----|
| 8081852 | Canceled to add cbp_no_gnt | Resubmitted as 8081884 |
| 8082032 | reset_head with full reinit collapsed at 2M | Changed to optimizer-state reset; resubmitted as 8082254 |
| 7883597 | Wrong array range (CUSTOM_ARRAY vs ARRAY_RANGE) | Resubmitted as 7883799 |

## Lessons Learned
- `CUSTOM_ARRAY` only works for `EXP_TYPE="custom"` in localcontrol; use `ARRAY_RANGE` for RL
- Add venv dirs (.upgd) to RSYNC_EXCLUDE to prevent accidental corruption
- Full output layer reinit in PPO → ratio explosion → collapse. Use shrink-and-perturb instead.
- Always clean up orphaned WandB runs after canceling jobs (same group/tag pollutes data)
