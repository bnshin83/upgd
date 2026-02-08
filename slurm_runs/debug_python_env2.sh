#!/bin/bash
#SBATCH --job-name=debug_py2
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:05:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_debug_python2.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_debug_python2.err

cd /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar

# Load only CUDA - do NOT load python module
module load cuda

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Ensure conda's library paths take precedence
export LD_LIBRARY_PATH="/scratch/gautschi/shin283/conda_envs/lop/lib:$LD_LIBRARY_PATH"

echo "=== Environment ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "which python3.8: $(which python3.8)"

echo ""
echo "=== Test 1: python3.8 --version ==="
python3.8 --version
echo "Return code: $?"

echo ""
echo "=== Test 2: python3.8 -c 'import sys; print(sys.version)' ==="
python3.8 -c 'import sys; print(sys.version)'
echo "Return code: $?"

echo ""
echo "=== Test 3: python3.8 -c 'import io' ==="
python3.8 -c 'import io; print("io module loaded successfully")'
echo "Return code: $?"

echo ""
echo "=== Test 4: Full path python3.8 -c 'import io' ==="
/scratch/gautschi/shin283/conda_envs/lop/bin/python3.8 -c 'import io; print("io module loaded successfully")'
echo "Return code: $?"

echo ""
echo "=== Test 5: Run actual experiment with full path ==="
/scratch/gautschi/shin283/conda_envs/lop/bin/python3.8 incremental_cifar_experiment.py --help 2>&1 | head -20
echo "Return code: $?"
