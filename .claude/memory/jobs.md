# Cluster Job Tracking

## Active Jobs

### Gautschi - Humanoid-v4 UPGD Methods
- **Job ID:** 7609377
- **Submitted:** 2026-02-09 ~9:15 AM EST
- **Cluster:** Gautschi (H100)
- **Config:** `.localcontrol/experiments/rl/humanoid_gautschi.sh`
- **Tasks:** 60 (array 0-59%8)
- **Methods:**
  - upgd_full (tasks 0-19, seeds 0-19)
  - upgd_output_only (tasks 20-39, seeds 0-19)
  - upgd_hidden_only (tasks 40-59, seeds 0-19)
- **Resources:** 1 GPU per task, 14 CPUs, 16h time limit
- **Timeline:** ~3.75 days (60 tasks / 8 concurrent)
- **Status:** Running (8/60 active as of 9:15 AM)
- **WandB:** shin283-purdue-university/upgd-rl (filter: *hum_20m*)

### Gilbreth - Humanoid-v4 Adam Baseline
- **Job ID:** 10271209 (replaces failed 10269468)
- **Submitted:** 2026-02-09 ~4:00 PM EST
- **Cluster:** Gilbreth (A100-80GB)
- **Config:** `.localcontrol/experiments/rl/humanoid_gilbreth.sh`
- **Tasks:** 10 array jobs (0-9%3)
- **Method:** adam (seeds 0-19, 2 seeds per array task in parallel)
  - Array task 0: seeds 0,1
  - Array task 1: seeds 2,3
  - ...
  - Array task 9: seeds 18,19
- **Resources:** 2 GPUs per task (full node), 14 CPUs, 20h time limit
- **Timeline:** ~1.67 days (10 tasks / 3 concurrent)
- **Status:** Pending (Resources)
- **Fix:** Changed hardcoded log path `/scratch/gautschi/...` → relative `logs/` in run_ppo_upgd.py:271
- **WandB:** shin283-purdue-university/upgd-rl (filter: *hum_20m_adam*)

## Monitoring

### Commands
```bash
# Dual-cluster monitoring
cd /Users/boonam/projects/upgd
./monitor_humanoid_dual.sh 7609377 10269468

# Individual cluster status
export PATH="$HOME/projects/localcontrol/bin:$PATH"
lc-status gautschi
lc-status gilbreth

# View logs
lc-logs gautschi 7609377
lc-logs gilbreth 10269468
```

### Expected Completion
- **Gilbreth:** ~2026-02-10 evening (1.67 days from 12:30 PM start)
- **Gautschi:** ~2026-02-12 evening (3.75 days from 9:15 AM start)
- **Combined:** All 80 runs by ~2026-02-12/13

## Failed Jobs

### Gilbreth - Humanoid-v4 Adam (FAILED)
- **Job ID:** 10269468
- **Failed:** 2026-02-09 ~9:43-9:53 AM EST
- **Cause:** Hardcoded log path `/scratch/gautschi/shin283/upgd/logs` in `run_ppo_upgd.py:271` — doesn't exist on Gilbreth
- **Error:** `PermissionError: [Errno 13] Permission denied: '/scratch/gautschi'`
- **Fix:** Changed to relative path `logs/`, resubmitted as job 10271209

## Completed Jobs

### Gautschi - Humanoid-v4 Test Run
- **Job ID:** 7608913
- **Completed:** 2026-02-09 ~9:00 AM EST
- **Tasks:** 2 (upgd_full seeds 0-1)
- **Purpose:** Verify logger race condition fix
- **Result:** ✓ Both tasks completed successfully (exit 0:0)
- **Runtime:** ~8 minutes each (200K timesteps)

### Ant-v4 Full Experiment (Previous)
- **Completed:** Before 2026-02-09
- **Tasks:** 100 (5 methods × 20 seeds)
- **Key Results:**
  - upgd_hidden_only: 4843 ± 510 (best)
  - upgd_output_only: 3229 ± 612 (worst)
  - upgd_full: 4570 ± 486
- **Conclusion:** Input-shift regime dominance (hidden >> output)

## Job History Notes
- All previous supervised learning jobs completed
- Gilbreth SGD job 10260102: 16/20 tasks complete, 4 running (finishes ~12:30 PM)
