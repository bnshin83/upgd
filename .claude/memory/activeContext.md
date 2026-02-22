# Active Context

**Last Updated:** 2026-02-22 (Sat)

## Current Focus
- **SlipperyAnt ablation study:** Theory-motivated CBP dissection + head intervention experiments
- Friction change every 2M steps = dynamics shift (output-shift dominant)

## Cluster Status

| Machine | Role | Current Jobs |
|---------|------|-------------|
| **Gautschi** (H100) | SlipperyAnt ablations | 8073259, 8081884, 8081956, 8082254 |
| **Gilbreth** (A100) | Idle | — |

## Running Jobs

### Gautschi
- **sant_val_s4_10** (8073259): seeds 4-10, 6 methods
  - Tasks 6-13 (std+l2) **CANCELED** — not needed
  - Tasks 14-17 (cbp seeds 4-7) **RUNNING** (~3h in)
  - Tasks 18-41 **HELD** (cbp remaining + all UPGD variants)
- **sant_val_cbp_layers** (8081884): cbp_h1, cbp_h2, cbp_no_gnt × 3 seeds = 9 tasks **RUNNING**
- **sant_val_upgd_wd4** (8081956): upgd_full (wd=1e-4) × 3 seeds = 3 tasks **RUNNING**
- **sant_val_reset_head** (8082254): reset_head + shrink_head × 3 seeds = 6 tasks **RUNNING/PENDING**

### Key WandB Groups (upgd-rl project)
- `sant__cbp_h1`, `sant__cbp_h2`, `sant__cbp_no_gnt` — CBP dissection
- `sant__upgd_full` (tag `wd1e-4`) — UPGD with matched weight decay
- `sant__reset_head`, `sant__shrink_head` — head intervention at friction change

## Experiment Design Rationale
1. **cbp_no_gnt vs cbp**: Does GnT (neuron regeneration) help or is it just Adam(betas=0.99, wd=1e-4)?
2. **cbp_h1 vs cbp_h2**: Where within hidden layers does regeneration matter?
3. **reset_head**: Clear Adam state for head at friction change (let head re-adapt faster)
4. **shrink_head**: Blend head 50% toward init + clear Adam state (soft reset)
5. **upgd_full (wd=1e-4)**: Fair comparison with CBP's weight decay

## Key Lesson This Session
- NEVER fully reinit output layer in PPO — ratio explodes → collapse
- Use shrink-and-perturb or optimizer-state-only reset instead

## Next Actions
- Monitor ablation runs (~24h each)
- Analyze: cbp_no_gnt vs cbp tells us if regeneration matters
- If reset_head or shrink_head beats cbp → validates φ-vs-head theory
- Release held tasks (18-41) after ablation slots free up
