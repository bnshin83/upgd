# Session Progress — 2026-02-23 (Tue 3pm update)

## Overview
20-seed production runs for Walker2d and HumanoidStandup (4 methods × 20 seeds each, 20M steps). Planning 40M extension with checkpoint support.

## Active Jobs

### Gautschi (H100)
| Job | Env | Tasks | Method | Status | ETA |
|-----|-----|-------|--------|--------|-----|
| 8107499 | HumanoidStandup | 42-59 all running | hidden_only | Running (18 tasks, 2-11h in) | Tue evening |

### Gilbreth (A100)
| Job | Env | Tasks | Method | Status | ETA |
|-----|-----|-------|--------|--------|-----|
| 10330476 | HumanoidStandup | 60-65 running, 66-79 pending | output_only | Running (6 concurrent, ~8h in) | Wed-Thu |

### Completed This Session
- Gautschi 8103978: Walker2d ALL 80 tasks DONE (adam + upgd_full + hidden_only + output_only, all 20 seeds each)
- Gilbreth 10320607: HumanoidStandup upgd_full tasks 36-39 (all done)
- Gilbreth 10329294: HumanoidStandup hidden_only tasks 40-41 (all done)

## Run Status

### Walker2d-v4 (Gautschi) — 20 seeds each — COMPLETE
| Method | Seeds | Status |
|--------|-------|--------|
| adam | 20/20 | Done (full 20M) |
| upgd_full | 20/20 | Done (full 20M) |
| upgd_hidden_only | 20/20 | Done (full 20M) |
| upgd_output_only | 20/20 | Done (full 20M) |

### HumanoidStandup-v4 — 20 seeds each
| Method | Seeds | Status |
|--------|-------|--------|
| adam | 20/20 | Done (full 20M, Gilbreth) |
| upgd_full | 16/20 | Done (full 20M; 4 seeds had final_step=None, skipped) |
| upgd_hidden_only | 2 done (Gilbreth) + 18 running (Gautschi) | Running (18 seeds plotted, aligned to 2.0M, ETA Tue evening) |
| upgd_output_only | 6 running + 14 pending (Gilbreth) | Running (6 seeds at 11.1M plotted, ETA Wed-Thu) |

## SLURM Scripts

### Gautschi (H100)
- `slurm_runs/slurm_rl_walker2d_all_20seeds.sh` — Walker2d 4 methods × 20 seeds
- `slurm_runs/slurm_rl_humanoidstandup_all_20seeds.sh` — HumanoidStandup 4 methods × 20 seeds

### Gilbreth (A100)
- `slurm_runs/gilbreth_slurm_rl_humanoidstandup_all_20seeds.sh` — HumanoidStandup
- `slurm_runs/gilbreth_slurm_rl_halfcheetah_all_20seeds.sh` — HalfCheetah

### Task-to-method mapping (all scripts)
```
Tasks  0-19: adam (seeds 0-19)
Tasks 20-39: upgd_full (seeds 0-19)
Tasks 40-59: upgd_hidden_only (seeds 0-19)
Tasks 60-79: upgd_output_only (seeds 0-19)
```

### Common hyperparameters
```bash
--total_timesteps 20000000
--learning_rate 3e-4
--weight_decay 0.0
--beta_utility 0.999
--sigma 0.001
--non_gated_scale 0.5  # Intentional: matches neutral utility 1-sigmoid(0)
```

## Plots

### Current plots
- `results/walker2d_4methods_partial.png` — 4-method comparison (all 20 seeds, ~20M each)
- `results/humanoidstandup_4methods_partial.png` — 4-method comparison (hidden/output in-progress)
- `results/walker2d_adam_vs_upgd_full.png` — 2-method comparison (complete)
- `results/humanoidstandup_adam_vs_upgd_full.png` — 2-method comparison (complete)

### How to regenerate (the "update" command)

**Step 1: Pull partial data from clusters**

Walker2d from Gautschi:
```bash
ssh gautschi "python3 -c \"
import os, json, re

results = {}
log_dir = '/scratch/gautschi/shin283/upgd/logs'
job_id = '8103978'

for task_id in range(40, 80):
    fname = f'{job_id}_{task_id}_rl_walker2d.out'
    fpath = os.path.join(log_dir, fname)
    if not os.path.exists(fpath):
        continue
    method_idx = task_id // 20
    seed = task_id % 20
    method = ['adam', 'upgd_full', 'upgd_hidden_only', 'upgd_output_only'][method_idx]
    steps = []
    returns = []
    with open(fpath) as f:
        for line in f:
            m = re.search(r'global_step=(\d+), episodic_return=([\d.\-e+]+)', line)
            if m:
                steps.append(int(m.group(1)))
                returns.append(float(m.group(2)))
    if steps:
        key = f'{method}_seed{seed}'
        results[key] = {
            'method': method, 'seed': seed,
            'steps': steps[::100], 'returns': returns[::100],
            'total_points': len(steps), 'max_step': steps[-1]
        }

print(json.dumps({'walker2d_partial': results}))
\"" > /tmp/walker2d_partial_raw.json
```

HumanoidStandup from Gilbreth (hidden_only seeds 0-1 + output_only):
```bash
ssh gilbreth "python3 -c \"
import os, json, re

results = {}
log_dir = '/scratch/gilbreth/shin283/upgd/logs'

# Completed hidden_only seeds 0-1 from job 10329294
for task_id in [40, 41]:
    fname = f'10329294_{task_id}_rl_humanoidstandup.out'
    fpath = os.path.join(log_dir, fname)
    if not os.path.exists(fpath):
        continue
    seed = task_id - 40
    steps, returns = [], []
    with open(fpath) as f:
        for line in f:
            m = re.search(r'global_step=(\d+), episodic_return=([\d.\-e+]+)', line)
            if m:
                steps.append(int(m.group(1)))
                returns.append(float(m.group(2)))
    if steps:
        results[f'upgd_hidden_only_seed{seed}'] = {
            'method': 'upgd_hidden_only', 'seed': seed,
            'steps': steps[::100], 'returns': returns[::100],
            'total_points': len(steps), 'max_step': steps[-1]
        }

# Output_only from job 10330476
for task_id in range(60, 80):
    fname = f'10330476_{task_id}_rl_humanoidstandup.out'
    fpath = os.path.join(log_dir, fname)
    if not os.path.exists(fpath):
        continue
    seed = task_id % 20
    steps, returns = [], []
    with open(fpath) as f:
        for line in f:
            m = re.search(r'global_step=(\d+), episodic_return=([\d.\-e+]+)', line)
            if m:
                steps.append(int(m.group(1)))
                returns.append(float(m.group(2)))
    if steps:
        results[f'upgd_output_only_seed{seed}'] = {
            'method': 'upgd_output_only', 'seed': seed,
            'steps': steps[::100], 'returns': returns[::100],
            'total_points': len(steps), 'max_step': steps[-1]
        }

print(json.dumps(results))
\"" > /tmp/humanoidstandup_gilbreth_partial.json
```

HumanoidStandup from Gautschi (hidden_only seeds 2-19):
```bash
ssh gautschi "python3 -c \"
import os, json, re

results = {}
log_dir = '/scratch/gautschi/shin283/upgd/logs'
job_id = '8107499'

for task_id in range(42, 60):
    fname = f'{job_id}_{task_id}_rl_humanoidstandup.out'
    fpath = os.path.join(log_dir, fname)
    if not os.path.exists(fpath):
        continue
    seed = task_id % 20
    steps, returns = [], []
    with open(fpath) as f:
        for line in f:
            m = re.search(r'global_step=(\d+), episodic_return=([\d.\-e+]+)', line)
            if m:
                steps.append(int(m.group(1)))
                returns.append(float(m.group(2)))
    if steps:
        results[f'upgd_hidden_only_seed{seed}'] = {
            'method': 'upgd_hidden_only', 'seed': seed,
            'steps': steps[::100], 'returns': returns[::100],
            'total_points': len(steps), 'max_step': steps[-1]
        }

print(json.dumps(results))
\"" > /tmp/humanoidstandup_gautschi_partial.json
```

**Step 2: Plot**

```python
source ~/venvs/paper/bin/activate
python3 << 'PYEOF'
import json
import numpy as np
import matplotlib.pyplot as plt

MIN_STEPS = 2_000_000  # filter seeds with < 2M steps

def load_json(path, key=None):
    try:
        with open(path) as f:
            d = json.load(f)
        return d[key] if key and key in d else d
    except:
        return {}

def align_and_smooth(runs, n_points=500, smooth_window=30):
    min_max_step = min(r[0][-1] for r in runs)
    x = np.linspace(0, min_max_step, n_points)
    aligned = []
    for steps, returns in runs:
        interp = np.interp(x, steps, returns)
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(interp, kernel, mode='same')
        for i in range(smooth_window // 2):
            smoothed[i] = np.mean(interp[:max(1, i + smooth_window // 2 + 1)])
            smoothed[-(i+1)] = np.mean(interp[-(i + smooth_window // 2 + 1):])
        aligned.append(smoothed)
    return x, np.array(aligned)

colors = {'adam': '#1f77b4', 'upgd_full': '#d62728',
          'upgd_hidden_only': '#2ca02c', 'upgd_output_only': '#ff7f0e'}
labels = {'adam': 'Adam', 'upgd_full': 'UPGD Full',
          'upgd_hidden_only': 'UPGD Hidden-Only', 'upgd_output_only': 'UPGD Output-Only'}
method_order = ['adam', 'upgd_full', 'upgd_hidden_only', 'upgd_output_only']

# --- Walker2d ---
completed_w = load_json('results/walker2d_returns.json')
partial_w = load_json('/tmp/walker2d_partial_raw.json', 'walker2d_partial')

all_w = {}
for key, run in completed_w.items():
    method = key.rsplit('_s', 1)[0]
    if method not in all_w: all_w[method] = []
    all_w[method].append((np.linspace(0, run['final_step'], len(run['returns'])),
                          np.array(run['returns'])))
for key, run in partial_w.items():
    method = run['method']
    if method not in all_w: all_w[method] = []
    all_w[method].append((np.array(run['steps']), np.array(run['returns'])))

# --- HumanoidStandup ---
completed_h = load_json('results/humanoidstandup_returns.json')
gilbreth_h = load_json('/tmp/humanoidstandup_gilbreth_partial.json')
gautschi_h = load_json('/tmp/humanoidstandup_gautschi_partial.json')

all_h = {}
for key, run in completed_h.items():
    method = key.rsplit('_s', 1)[0]
    if method not in all_h: all_h[method] = []
    if run['final_step'] is None: continue  # skip in-progress upgd_full
    all_h[method].append((np.linspace(0, run['final_step'], len(run['returns'])),
                          np.array(run['returns'])))
for src in [gilbreth_h, gautschi_h]:
    for key, run in src.items():
        if run['max_step'] < MIN_STEPS: continue
        method = run['method']
        if method not in all_h: all_h[method] = []
        all_h[method].append((np.array(run['steps']), np.array(run['returns'])))

# --- Generate both plots ---
for env, all_data, outfile in [
    ('Walker2d-v4', all_w, 'results/walker2d_4methods_partial.png'),
    ('HumanoidStandup-v4', all_h, 'results/humanoidstandup_4methods_partial.png')
]:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for method in method_order:
        if method not in all_data or len(all_data[method]) == 0: continue
        runs = all_data[method]
        n_seeds = len(runs)
        x, aligned = align_and_smooth(runs)
        mean = np.mean(aligned, axis=0)
        se = np.std(aligned, axis=0) / np.sqrt(n_seeds)
        max_step_M = x[-1] / 1e6
        is_partial = method in ('upgd_hidden_only', 'upgd_output_only')
        suffix = ' (in-progress)' if is_partial else ''
        label = f'{labels[method]} ({n_seeds} seeds, {max_step_M:.0f}M){suffix}'
        linestyle = '--' if is_partial else '-'
        ax.plot(x / 1e6, mean, color=colors[method], label=label,
                linewidth=2, linestyle=linestyle)
        ax.fill_between(x / 1e6, mean - se, mean + se,
                        color=colors[method], alpha=0.15)
    ax.set_xlabel('Environment Steps (M)', fontsize=13)
    ax.set_ylabel('Episodic Return', fontsize=13)
    ax.set_title(f'{env} — 4 Methods Comparison', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
PYEOF
```

### Data sources
- **Completed runs (adam, upgd_full):** `results/walker2d_returns.json`, `results/humanoidstandup_returns.json`
  - Pulled from cluster JSON logs (full 20M data)
  - Format: `{method}_s{seed}` → `{returns: [], lengths: [], status, final_step}`
- **In-progress runs:** Parsed from SLURM stdout logs (`global_step=X, episodic_return=Y`)
  - Subsampled every 100th point to keep transfer size manageable
- **Dashed lines** = in-progress, **solid lines** = complete
- **MIN_STEPS filter** = 2M (excludes just-started seeds from HumanoidStandup)

## Findings (as of Tue 3pm)
- **Walker2d (FINAL):** hidden_only ≈ full (~3800) >> output_only ≈ Adam (~3200-3500) — classic **input-shift regime** (same as Ant-v4)
- **HumanoidStandup (partial, emerging):** full (140K) > hidden_only (~120K) > output_only (~110K) > Adam (100K) — suggests **joint-shift regime** where protecting all layers matters
- **Different regimes across envs:** Walker2d matches Ant (input-shift dominant), HumanoidStandup shows different pattern
- **20M steps insufficient** — curves still rising, planning 40M rerun

## Next Steps
- Wait for HumanoidStandup hidden_only (Gautschi, ETA Tue evening) and output_only (Gilbreth, ETA Wed-Thu)
- `plan/checkpoint_resume_40m.md` — add checkpoint save/resume to `run_ppo_upgd.py`
- Rerun all methods at 40M after current 20M runs finish
- Check HalfCheetah status on Gilbreth (not checked this session)
- Final 4-method plots when all 20M runs complete
