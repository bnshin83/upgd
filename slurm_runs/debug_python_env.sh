#!/bin/bash
#SBATCH --job-name=debug_python
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:05:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_debug_python.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_debug_python.err

# Debug: Print environment BEFORE any module loads
echo "=== INITIAL ENVIRONMENT ==="
echo "PYTHONHOME=$PYTHONHOME"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# Change to the incremental_cifar directory
cd /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar

# Load only CUDA
module load cuda

echo "=== AFTER module load cuda ==="
echo "PYTHONHOME=$PYTHONHOME"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

echo "=== AFTER conda activate ==="
echo "PYTHONHOME=$PYTHONHOME"
echo "PYTHONPATH=$PYTHONPATH"  
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PATH=$PATH"
echo ""
echo "which python: $(which python)"
echo "which python3.8: $(which python3.8)"
echo ""

# Try running python
echo "=== Testing python ==="
python --version
echo "Return code: $?"

echo "=== Testing python3.8 ==="
python3.8 --version
echo "Return code: $?"

# Try with explicit unset
echo ""
echo "=== Testing with unset PYTHONHOME ==="
unset PYTHONHOME
python --version
echo "Return code: $?"
