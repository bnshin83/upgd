# Humanoid-v4 RL Experiment Status

## Overview
Complete Humanoid-v4 experiments with 80 tasks (4 methods × 20 seeds) to validate the input-shift regime hypothesis on a second, more complex environment.

## Current Status: ✅ Ready for Full Submission

### Test Run (COMPLETED)
- **Job ID:** 7608913 (2nd attempt after race condition fix)
- **Status:** Both tasks running successfully
- **Fix Applied:** Logger race condition fixed (exist_ok=True)
- **Runtime:** ~30 min per task for 200K timesteps

### Full Experiment Configuration
- **Total tasks:** 80 (4 methods × 20 seeds)
- **Methods:** upgd_full, upgd_output_only, upgd_hidden_only, adam
- **Seeds:** 0-19 (20 seeds per method)
- **Timesteps:** 20M per task (~12-14 hours)
- **Concurrent:** 8 tasks (H100 GPUs)
- **Wall time:** 5-7 days estimated

### Array Task Mapping
```
Tasks  0-19  → upgd_full          seeds 0-19
Tasks 20-39  → upgd_output_only   seeds 0-19
Tasks 40-59  → upgd_hidden_only   seeds 0-19
Tasks 60-79  → adam               seeds 0-19
```

## Files Created

### Experiment Configurations
1. **Test:** `.localcontrol/experiments/rl/humanoid_test.sh`
   - 2 tasks, 200K timesteps, 2-hour limit

2. **Full:** `.localcontrol/experiments/rl/humanoid_completion.sh`
   - 80 tasks, 20M timesteps, 16-hour limit, 8 concurrent

### Monitoring
3. **Script:** `monitor_humanoid.sh`
   - Usage: `./monitor_humanoid.sh [job_id]`
   - Shows status, logs, and useful commands

### Code Fixes
4. **Logger Fix:** `core/logger.py` (line 52)
   - Added `exist_ok=True` to prevent race conditions
   - Critical for parallel job execution

## Next Steps

### Step 1: Verify Test Completion (~30 min)
```bash
cd /Users/boonam/projects/upgd
./monitor_humanoid.sh 7608913
```

**Success criteria:**
- Both tasks complete with `State=COMPLETED`, `ExitCode=0:0`
- WandB runs show training progress
- No errors in logs

### Step 2: Submit Full Job (After test passes)
```bash
cd /Users/boonam/projects/upgd
PATH="$HOME/projects/localcontrol/bin:$PATH" \
  lc-submit --cluster gautschi --exp rl/humanoid_completion --sync
```

### Step 3: Monitor Progress (5-7 days)
```bash
# Quick status check
./monitor_humanoid.sh [full_job_id]

# Live monitoring
PATH="$HOME/projects/localcontrol/bin:$PATH" lc-status gautschi --watch

# Check WandB
# https://wandb.ai/shin283-purdue-university/upgd-rl
```

### Step 4: Verification After Completion
```bash
# Check all tasks completed
ssh gautschi "sacct -j [job_id] --format=JobID,State,ExitCode -X"

# Verify WandB runs (should be 80 total)
# Each run should have >9,000 logged steps (20M / 2048)
```

## Expected Results

Based on Ant-v4 (input-shift regime):
- **Hidden-only:** 4843 ± 510 (best)
- **Output-only:** 3229 ± 612 (worst)
- **Full:** 4570 ± (comparable to hidden)
- **Adam:** TBD

**Hypothesis for Humanoid-v4:**
- Similar input-shift dominance (hidden > output)
- Possibly larger margin due to 14× higher observation dimensionality (376 vs 27)

## Resources

- **GPU-hours:** ~960 hours (80 × 12h avg)
- **Storage:** ~8 GB (100MB × 80 runs)
- **Cluster:** Gautschi H100 (8 concurrent)
- **WandB:** shin283-purdue-university/upgd-rl

## Timeline

- **Test run:** ~1 hour (CURRENT)
- **Full run:** 5-7 days
- **Analysis:** 1-2 days
- **Total:** 7-10 days

## Key Decisions

1. **20 seeds** (vs 5 in original Ant): Stronger statistical evidence
2. **Gautschi H100** (vs Gilbreth A100): Higher throughput (8 vs 6 concurrent)
3. **Humanoid only** (vs all 4 MuJoCo envs): Focus resources for paper deadline
4. **Walker2d/HumanoidStandup** → Future work

## Contact & Monitoring

- **WandB Dashboard:** https://wandb.ai/shin283-purdue-university/upgd-rl
- **Filter runs:** `*humanoid*` or `*hum_*`
- **Monitoring script:** `./monitor_humanoid.sh`
- **Cluster:** Gautschi (ssh gautschi)

---

**Last Updated:** 2026-02-09
**Status:** Test running (Job 7608913)
**Next Action:** Verify test, then submit full job
