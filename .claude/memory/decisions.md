# Key Decisions

## Humanoid-v4 Experiment Design (2026-02-09)

### Decision: Dual-Cluster Split by Method Type
**Chosen:** Gautschi (UPGD methods) + Gilbreth (adam baseline)

**Rationale:**
- Conceptually cleaner: all UPGD variants together, baseline separate
- Efficient resource use: Gilbreth uses full nodes (2 GPUs per task)
- Faster completion: 3.75 days vs 5-7 days on single cluster
- Doesn't need perfect load balance (user's preference)

**Alternatives considered:**
- Even 40/40 split: More balanced but splits method families
- Single cluster: Simpler but slower (5-7 days)

### Decision: 20 Seeds per Method
**Chosen:** 20 seeds for all methods

**Rationale:**
- Strong statistical evidence (matches Ant-v4)
- Required for robust mean/stdev calculations
- Enables proper significance testing
- Industry standard for RL experiments

### Decision: Gilbreth Configuration - 2 Seeds Parallel per Node
**Chosen:** 10 array tasks, each runs 2 seeds in parallel on 2 GPUs

**Rationale:**
- Matches old proven Gilbreth scripts
- Uses full node resources (no GPU waste)
- Better cluster citizenship (full node allocation)
- Same completion time as 1-GPU-per-task (both ~3.5 days)

### Decision: Humanoid-v4 Only (Skip Walker2d, HumanoidStandup)
**Chosen:** Focus on Humanoid-v4, mark others as future work

**Rationale:**
- Time constraints: paper deadline in 2-3 weeks
- Humanoid is sufficient to validate regime theory (2 environments: Ant + Humanoid)
- 376-dim observation (14× more complex than Ant's 27-dim)
- Better use of resources: 80 runs with high quality > spreading thin

## Technical Fixes

### Logger Race Condition Fix (2026-02-09)
**Issue:** Multiple tasks starting simultaneously crashed when creating same log directory

**Solution:** Added `exist_ok=True` to `os.makedirs()` in `core/logger.py:52`

**Impact:** Critical fix - prevents random task failures in array jobs

### Commit .localcontrol/ to Git (2026-02-09)
**Decision:** Override .gitignore and track experiment configs

**Rationale:**
- Needed for cross-machine work (studio ↔ laptop)
- Experiment configs contain critical job setup
- Documentation (.md files) not sufficient alone
- User works across machines and needs full context
