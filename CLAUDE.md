# UPGD Research Project

Utility-Gated Perturbation-based optimizer for continual learning research.

## Project Overview
Research codebase for UPGD (Utility-Gated Perturbations with Directional control), investigating:
- Memorization vs plasticity in continual learning
- Layer-selective gating strategies (output-only, hidden-only, full)
- RL and supervised learning experiments
- Regime theory (input-shift vs output-shift dominance)

## Key Commands
- `/session-start`: Load context and running job status
- `/session-end`: Save progress, update job status in memory bank
- `/sync`: Git commit + push for cross-machine work

## Memory Bank (`.claude/memory/`)
- `projectContext.md`: Project goals, key findings, paper venues
- `activeContext.md`: Current running jobs, recent changes
- `progress.md`: Experiment status, what's complete/in-progress
- `decisions.md`: Key technical decisions
- `jobs.md`: Cluster job tracking (IDs, status, results)

## Current Setup

### Cluster Job Management
- **Tool:** localcontrol (located at `/Users/boonam/projects/localcontrol`)
- **Configs:** `.localcontrol/experiments/` (RL and supervised learning)
- **Commands:**
  ```bash
  export PATH="$HOME/projects/localcontrol/bin:$PATH"
  lc-submit --cluster gautschi --exp rl/humanoid_gautschi --sync
  lc-status gautschi
  lc-logs gautschi [job_id]
  ```

### Active Clusters
- **Gautschi:** H100 GPUs, 8 concurrent, partition "ai"
- **Gilbreth:** A100-80GB GPUs, 6 concurrent (3 nodes × 2 GPUs), partition "a100-80gb"

### Key Directories
- `core/`: Main codebase (optimizers, learners, utilities)
- `.localcontrol/experiments/`: Experiment configurations
- `slurm_runs/`: Legacy SLURM scripts (reference only)
- `monitor_humanoid_dual.sh`: Dual-cluster monitoring script

## Important Notes
- Logger race condition fixed (core/logger.py - exist_ok=True)
- Experiment configs now tracked in git (including .localcontrol/)
- WandB tracking: shin283-purdue-university/upgd-rl

## Related Projects
- Main research: `/Users/boonam/research` (paper writing, lit database)
- This is the codebase for running experiments
