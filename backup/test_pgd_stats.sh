#!/bin/bash
#SBATCH --job-name=test_pgd_stats
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_pgd_stats.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_pgd_stats.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

echo "========================================="
echo "Testing single PGD statistics command"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "========================================="

# Run single test command with run_stats.py to get plasticity measurements
export CUDA_VISIBLE_DEVICES=0
echo "Running: python3 core/run/run_stats.py --task input_permuted_mnist_stats --learner pgd --seed 0 --lr 0.01 --sigma 0.001 --network fully_connected_relu_with_hooks --n_samples 1000000"
python3 core/run/run_stats.py --task input_permuted_mnist_stats --learner pgd --seed 0 --lr 0.01 --sigma 0.001 --network fully_connected_relu_with_hooks --n_samples 1000000

echo "========================================="
echo "Test completed"
echo "End time: $(date)"
echo "========================================="