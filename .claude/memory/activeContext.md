# Active Context

**Last Updated:** 2026-02-22 (Sat evening)

## Current Focus
- **SlipperyAnt ablation study:** Running combo methods (cbp_shrink, cbp_fast, cbp_h1_shrink, cbp_h1_full)
- All new runs include per-layer drift logging for theory validation

## Active Gautschi Jobs

| Job | Method | Seeds | Status |
|-----|--------|-------|--------|
| 8073259 (18-41) | cbp+UPGD s4-10 | 4-10 | HELD |
| 8088188 | shrink_head | 1-3 | Running |
| 8090346 (3-5) | cbp_fast | 1-3 | Running |
| 8090377 | cbp_h1_shrink + cbp_h1_full | 1-3 | Pending |
| 8092578 | cbp_shrink (fixed) | 1-3 | Pending/Starting |

## Completed Ablation Results

| Method | Seeds | Result |
|--------|-------|--------|
| std | 10 (9 ok, 1 crash) | Weak baseline |
| l2 | 10 (3 ok, 7 crash) | Crashes need investigation |
| cbp | 7 | Strong — best so far |
| cbp_h1 | 3 | Good, beats cbp at 10-12M, unstable later |
| cbp_h2 | 3 | Weakest CBP variant |
| cbp_no_gnt | 3 | Bad — confirms GnT matters |
| reset_head | 3 | Mild effect, s3 bad friction (0.022) |
| upgd_full | 6 (3 wd=1e-3 + 3 wd=1e-4) | Middle of pack |
| upgd_output_only | 3 | Weak |
| upgd_hidden_only | 3 | Weak |

## Next Actions
- Monitor combo method results (~24h each)
- Key question: does cbp_shrink or cbp_h1_shrink beat cbp?
- Check drift logs in WandB for P_T^φ vs P_T^head validation
- Investigate l2 crash issue (7/10 crashed at 5.7-9.1M steps)
- Release held tasks (18-41) when slots free up
