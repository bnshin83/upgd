# Plan: Add Checkpoint Save + Resume to PPO Runner (40M Steps)

## Context
20M steps are insufficient for Walker2d and HumanoidStandup. We need to rerun all 4 methods × 20 seeds at 40M steps with checkpoint saving, so runs can be extended further if needed. Current runs have no checkpoints — must rerun from scratch.

## Scope
- Save a final checkpoint at end of training (not periodic)
- Add `--resume` flag to continue from a checkpoint
- Rerun Walker2d + HumanoidStandup at 40M with checkpointing
- Update SLURM scripts accordingly

## Changes

### 1. Add args to `core/run/rl/run_ppo_upgd.py`

In the `Args` dataclass (~line 42):

```python
save_model: bool = True          # Change default from False to True
"""save final model checkpoint for resume"""
resume_checkpoint: str = ""
"""path to checkpoint file to resume from"""
```

### 2. Add save function (~line 629, after training loop ends)

Save at end of training, before final JSON save:

```python
if args.save_model:
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/final.pt"
    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "iteration": args.num_iterations,
        "next_obs": next_obs.cpu(),
        "next_done": next_done.cpu(),
        "args": vars(args),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and args.cuda:
        checkpoint["rng_torch_cuda"] = torch.cuda.get_rng_state()
    if args.gating_schedule:
        checkpoint["current_macro_phase"] = current_macro_phase
    if args.track:
        checkpoint["wandb_run_id"] = wandb.run.id
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")
```

### 3. Add resume logic (~line 262, after optimizer creation)

After `optimizer = create_optimizer(agent, args)` and before training loop:

```python
start_iteration = 1
if args.resume_checkpoint:
    ckpt = torch.load(args.resume_checkpoint, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    global_step = ckpt["global_step"]
    start_iteration = ckpt["iteration"] + 1
    next_obs = ckpt["next_obs"].to(device)
    next_done = ckpt["next_done"].to(device)
    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    torch.set_rng_state(ckpt["rng_torch"])
    if "rng_torch_cuda" in ckpt and args.cuda:
        torch.cuda.set_rng_state(ckpt["rng_torch_cuda"])
    if "current_macro_phase" in ckpt:
        current_macro_phase = ckpt["current_macro_phase"]
    if args.track and "wandb_run_id" in ckpt:
        # wandb.init must use resume="must" with saved run ID
        # Handle this in the wandb.init block above
        pass
    print(f"Resumed from {args.resume_checkpoint} at iteration {start_iteration}, global_step={global_step}")
```

Change training loop from:
```python
for iteration in range(1, args.num_iterations + 1):
```
to:
```python
for iteration in range(start_iteration, args.num_iterations + 1):
```

### 4. Fix LR annealing for resume

Current annealing uses `args.num_iterations` as denominator. When resuming a 20M→40M extension, `num_iterations` changes. The annealing fraction should account for resume:

```python
if args.anneal_lr:
    frac = 1.0 - (iteration - 1.0) / args.num_iterations
    lrnow = frac * args.learning_rate
    optimizer.param_groups[0]["lr"] = lrnow
```

This already works correctly — `args.num_iterations` will be the NEW total (40M/2048 ≈ 19531), and `iteration` starts from where we left off, so the LR continues to anneal from its current value toward 0. No change needed.

### 5. WandB resume handling

In the wandb.init block (~line 230), add resume support:

```python
if args.track:
    import wandb
    wandb_kwargs = dict(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        if "wandb_run_id" in ckpt:
            wandb_kwargs["id"] = ckpt["wandb_run_id"]
            wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)
```

Note: The checkpoint needs to be loaded BEFORE wandb.init for this to work. May need to restructure slightly — load checkpoint path early just to get wandb_run_id, then full load later.

### 6. Update SLURM scripts

For the 40M rerun, update both Walker2d and HumanoidStandup scripts:

```bash
python3 core/run/rl/run_ppo_upgd.py \
    --total_timesteps 40000000 \   # Changed from 20000000
    --save_model \                  # Enable checkpoint saving
    ...  # rest unchanged
```

Create new scripts or update existing:
- `slurm_runs/slurm_rl_walker2d_40m.sh`
- `slurm_runs/slurm_rl_humanoidstandup_40m.sh`

Time estimate: ~22-24h per task on H100 (double of ~11h for 20M).
SLURM time limit: increase from `3-00:00:00` to `3-00:00:00` (72h already sufficient).

### 7. Checkpoint disk usage

- Each checkpoint: ~2-3MB (MLP 64×64 is small)
- 80 runs × 1 checkpoint = ~200MB total per env
- Saved to: `checkpoints/{env}__{method}__{seed}__{timestamp}/final.pt`

## Files to Modify
1. `core/run/rl/run_ppo_upgd.py` — checkpoint save/load + resume args
2. `slurm_runs/slurm_rl_walker2d_40m.sh` — new 40M script
3. `slurm_runs/slurm_rl_humanoidstandup_40m.sh` — new 40M script

## Verification
1. Syntax check: `python3 -c "import ast; ast.parse(open('core/run/rl/run_ppo_upgd.py').read())"`
2. Confirm checkpoint file is created after a run completes
3. Confirm resume loads correctly and training continues from saved iteration
