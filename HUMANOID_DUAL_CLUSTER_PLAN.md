# Humanoid-v4 Dual-Cluster Execution Plan

## Distribution Strategy

**Gautschi (H100, 8 concurrent GPUs):**
- **Methods:** upgd_full, upgd_output_only, upgd_hidden_only
- **Tasks:** 60 (3 methods × 20 seeds)
- **Timeline:** 60 ÷ 8 = 7.5 batches × 12h = **~90h = 3.75 days**

**Gilbreth (A100, 3 nodes × 2 GPUs = 6 GPUs):**
- **Method:** adam (baseline)
- **Tasks:** 10 array jobs (each runs 2 seeds in parallel)
- **Timeline:** 10 ÷ 3 = 3.34 batches × 12h = **~40h = 1.67 days**

**Total completion time: ~3.75 days** (Gautschi finishes last)

---

## Task Mapping

### Gautschi (60 tasks, 1 GPU per task)
```
Task  0-19: upgd_full         seeds 0-19
Task 20-39: upgd_output_only  seeds 0-19
Task 40-59: upgd_hidden_only  seeds 0-19
```

### Gilbreth (10 array tasks, 2 GPUs per task)
```
Array Task 0: adam seeds 0,1   (parallel on GPU 0,1)
Array Task 1: adam seeds 2,3   (parallel on GPU 0,1)
Array Task 2: adam seeds 4,5   (parallel on GPU 0,1)
...
Array Task 9: adam seeds 18,19 (parallel on GPU 0,1)
```

---

## Configuration Files

### 1. Gautschi Experiment
**File:** `.localcontrol/experiments/rl/humanoid_gautschi.sh`
- Standard RL experiment type
- 60 tasks with ARRAY_RANGE="0-59%8"
- 16-hour time limit
- 1 GPU per task (standard)

### 2. Gilbreth Experiment
**File:** `.localcontrol/experiments/rl/humanoid_gilbreth.sh`
- Custom experiment type (runs 2 seeds in parallel)
- 10 array tasks with CUSTOM_ARRAY="0-9%3"
- 20-hour time limit (for 2 parallel runs)
- 2 GPUs per task (full node utilization)

---

## Prerequisites

### Gautschi Test Job
- **Status:** Running (Job 7608913)
- **Completion:** ~30 minutes from submission
- **Verify:** Both tasks complete successfully

### Gilbreth SGD Jobs
- **Status:** 4 tasks running (Job 10260102)
- **Completion:** ~12:30 PM EST today
- **Action:** Wait for completion before submitting Humanoid

---

## Submission Commands

### 1. Submit to Gautschi (After test passes)
```bash
cd /Users/boonam/projects/upgd
PATH="$HOME/projects/localcontrol/bin:$PATH" \
  lc-submit --cluster gautschi --exp rl/humanoid_gautschi --sync
```

### 2. Submit to Gilbreth (After SGD jobs finish ~12:30 PM)
```bash
cd /Users/boonam/projects/upgd
PATH="$HOME/projects/localcontrol/bin:$PATH" \
  lc-submit --cluster gilbreth --exp rl/humanoid_gilbreth --sync
```

---

## Monitoring

### Dual-Cluster Monitor
```bash
cd /Users/boonam/projects/upgd
./monitor_humanoid_dual.sh [gautschi_job_id] [gilbreth_job_id]
```

### Individual Cluster Status
```bash
# Gautschi
PATH="$HOME/projects/localcontrol/bin:$PATH" lc-status gautschi

# Gilbreth
PATH="$HOME/projects/localcontrol/bin:$PATH" lc-status gilbreth
```

### View Logs
```bash
# Gautschi logs
PATH="$HOME/projects/localcontrol/bin:$PATH" lc-logs gautschi [job_id]

# Gilbreth logs
PATH="$HOME/projects/localcontrol/bin:$PATH" lc-logs gilbreth [job_id]
```

### WandB Dashboard
https://wandb.ai/shin283-purdue-university/upgd-rl
- Filter: `*humanoid*` or `*hum_20m*`
- Expected: 80 total runs (60 from Gautschi + 20 from Gilbreth)

---

## Timeline

### Day 0 (Today)
- ✅ Test job submitted to Gautschi (7608913)
- ⏳ Test completion (~30 min)
- ⏳ SGD jobs finish on Gilbreth (~12:30 PM)
- 🎯 Submit both Humanoid jobs

### Day 1-2
- Gilbreth completes (~40h = 1.67 days)
- Gautschi continues running

### Day 3-4
- Gautschi completes (~90h = 3.75 days)
- All 80 tasks done

### Day 4-5
- Analyze results
- Generate figures
- Update paper

---

## Advantages of This Split

1. **Conceptually clean:** All UPGD variants together, Adam baseline separate
2. **Efficient resource use:** Gilbreth uses full nodes (2 GPUs), no waste
3. **Faster than single cluster:** 3.75 days vs 5-7 days
4. **Proven config:** Gilbreth approach matches old working scripts
5. **Easy to interpret:** Methods grouped logically

---

## Verification After Completion

### Check All Tasks Completed
```bash
# Gautschi (should have 60 completed)
ssh gautschi "sacct -j [job_id] --format=State -X | grep COMPLETED | wc -l"

# Gilbreth (should have 10 completed)
ssh gilbreth "sacct -j [job_id] --format=State -X | grep COMPLETED | wc -l"
```

### Verify WandB Runs
- Expected: 80 total runs
- Each run should have >9,000 logged steps (20M / 2048)
- Check for any failed/incomplete runs

### Extract Results
- Gautschi: 60 runs (upgd_full, upgd_output_only, upgd_hidden_only)
- Gilbreth: 20 runs (adam)
- Compute means and stdev for each method
- Compare with Ant-v4 results

---

**Last Updated:** 2026-02-09 09:15 AM EST
**Status:** Test job running, ready for full submission
