# Active Context

**Last Updated:** 2026-02-09

## Current Focus
- **RL experiments:** Humanoid-v4 running on dual clusters (Gautschi + Gilbreth)
- **Cross-machine workflow:** Established sync system between Studio and Pro via Tailscale

## Running Jobs

### Gautschi (H100)
- **Job ID:** 7609377
- **Started:** 2026-02-09 ~9:15 AM EST
- **Tasks:** 60 (8 concurrent)
- **Methods:** upgd_full (0-19), upgd_output_only (20-39), upgd_hidden_only (40-59)
- **Timeline:** ~3.75 days
- **Status:** Running (8/60 active, tasks 0-7 upgd_full, ~2h elapsed as of 11:15 AM)

### Gilbreth (A100)
- **Job ID:** 10271209 (replaces failed 10269468)
- **Submitted:** 2026-02-09 ~4:00 PM EST
- **Tasks:** 10 array jobs (2 seeds parallel each = 20 total seeds)
- **Method:** adam (baseline)
- **Timeline:** ~1.67 days
- **Status:** Pending (Resources)
- **Note:** Previous job 10269468 failed — hardcoded `/scratch/gautschi` log path. Fixed to use relative `logs/`.

## Recent Changes
- **2026-02-09 PM:** Cross-machine sync system established (Studio ↔ Pro via Tailscale)
  - Created `~/sync-repo.sh` (bash 3.2 compatible) for complete bidirectional sync
  - Updated `~/.claude/CLAUDE.md` so Claude handles "sync upgd" commands automatically
  - Tested: synced 732 files from Studio to Pro successfully
  - Documentation: `~/.claude/MAC_SYNC_SETUP_HISTORY.md`
- **2026-02-09 AM:** Fixed logger race condition (exist_ok=True in os.makedirs)
- **2026-02-09 AM:** Created dual-cluster experiment split (Gautschi: UPGD methods, Gilbreth: adam)
- **2026-02-09 AM:** Added monitoring scripts (monitor_humanoid_dual.sh)
- **2026-02-09 AM:** Committed .localcontrol/ configs to git for cross-machine work

## Next Actions
- Monitor Humanoid-v4 job completion (~3-4 days)
- Verify all 80 runs complete successfully
- Extract results from WandB
- Compare Humanoid-v4 results with Ant-v4 (validate regime hypothesis)
- Update paper with findings
- Use "sync upgd" to keep Pro/Studio synchronized during analysis phase
