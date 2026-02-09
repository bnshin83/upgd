# Active Context

**Last Updated:** 2026-02-09

## Current Focus
Running Humanoid-v4 RL experiments across two clusters to validate regime theory on complex continuous control tasks.

## Running Jobs

### Gautschi (H100)
- **Job ID:** 7609377
- **Started:** 2026-02-09 ~9:15 AM EST
- **Tasks:** 60 (8 concurrent)
- **Methods:** upgd_full (0-19), upgd_output_only (20-39), upgd_hidden_only (40-59)
- **Timeline:** ~3.75 days
- **Status:** Running (8/60 active as of 9:15 AM)

### Gilbreth (A100)
- **Job ID:** 10269468
- **Started:** Queued (waiting for SGD jobs to complete)
- **Tasks:** 10 array jobs (2 seeds parallel each = 20 total seeds)
- **Method:** adam (baseline)
- **Timeline:** ~1.67 days (will start ~12:30 PM EST)
- **Status:** Pending (Resources)

## Recent Changes
- **2026-02-09:** Fixed logger race condition (exist_ok=True in os.makedirs)
- **2026-02-09:** Created dual-cluster experiment split (Gautschi: UPGD methods, Gilbreth: adam)
- **2026-02-09:** Added monitoring scripts (monitor_humanoid_dual.sh)
- **2026-02-09:** Committed .localcontrol/ configs to git for cross-machine work

## Next Actions
- Monitor job completion (~3-4 days)
- Verify all 80 runs complete successfully
- Extract results from WandB
- Compare Humanoid-v4 results with Ant-v4 (validate regime hypothesis)
- Update paper with findings
