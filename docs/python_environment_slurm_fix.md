# Python Environment Issue on SLURM Nodes (Gautschi Cluster)

**Date**: 2026-01-24
**Updated**: 2026-01-24
**Issue**: Python 3.8 conda environment fails with `ModuleNotFoundError: No module named 'io'`

## Problem Summary

When running Python 3.8 from a conda environment on SLURM jobs, the interpreter fails to import any modules (even stdlib modules like `sys` and `io`) with this error:

```
Could not find platform independent libraries <prefix>
Consider setting $PYTHONHOME to <prefix>[:<exec_prefix>]
Fatal Python error: init_sys_streams: can't initialize sys standard streams
Python runtime state: core initialized
ModuleNotFoundError: No module named 'io'
```

## Root Cause

The Gautschi cluster's SLURM nodes have **Python 3.11's shared library** pre-loaded in `LD_LIBRARY_PATH`:

```
/apps/spack/gautschi-cpu/apps/python/3.11.9-gcc-14.1.0-hbmvmni/lib
```

When conda's `python3.8` binary starts, it loads `libpython3.11.so` instead of `libpython3.8.so` due to the dynamic linker searching `LD_LIBRARY_PATH`. This causes incompatible library versions and Python fails to initialize its standard streams.

This issue started occurring around January 22-24, 2026 due to cluster environment changes.

## Solution (RECOMMENDED)

**Use the Python 3.11 virtual environment** instead of conda's Python 3.8. This venv uses the cluster's native Python 3.11, avoiding all library conflicts.

### Setup (one-time)

If the venv doesn't exist, create it:
```bash
module load python
python3 -m venv /scratch/gautschi/shin283/loss-of-plasticity/.lop_venv_compute
source /scratch/gautschi/shin283/loss-of-plasticity/.lop_venv_compute/bin/activate
pip install -r /scratch/gautschi/shin283/loss-of-plasticity/requirements.txt
pip install -e /scratch/gautschi/shin283/loss-of-plasticity
```

### Complete SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_output.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_output.err

cd /path/to/working/directory

# Load modules
module load cuda python

# Display GPU information
nvidia-smi

# Activate the Python 3.11 venv
source /scratch/gautschi/shin283/loss-of-plasticity/.lop_venv_compute/bin/activate

# Set PYTHONPATH to include project
export PYTHONPATH="/scratch/gautschi/shin283/loss-of-plasticity:$PYTHONPATH"

# Run your Python script (use 'python', not 'python3.8')
python your_script.py
```

## Key Points

1. **Use `module load cuda python`** - load both modules
2. **Use the Python 3.11 venv** at `.lop_venv_compute`
3. **Use `python`** (not `python3.8`) - the venv provides Python 3.11
4. **Set PYTHONPATH** to include the project directory

## Alternative: Conda with LD_LIBRARY_PATH fix (NOT RECOMMENDED)

This approach was attempted but proved unreliable due to subprocess/multiprocessing issues:

```bash
# This does NOT reliably work - subprocess workers still fail
module load cuda
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's|/apps/spack/gautschi-cpu/apps/python/3.11.9-gcc-14.1.0-hbmvmni/lib:||g')
export LD_LIBRARY_PATH="/scratch/gautschi/shin283/conda_envs/lop/lib:$LD_LIBRARY_PATH"
python3.8 your_script.py  # May fail in DataLoader workers
```

## Debug Scripts

Debug scripts used to diagnose this issue are saved in:
- `slurm_runs/debug_python_env.sh` - Initial environment check
- `slurm_runs/debug_python_env2.sh` - Import testing
