#!/bin/bash
#SBATCH --job-name=test_si_input_mnist_stats_seed0
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_si_input_mnist_stats_seed0.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_si_input_mnist_stats_seed0.err

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
echo "Running SI Input Permuted MNIST Statistics (Seed 0)"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "========================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run SI experiment with input permuted MNIST
echo "Running: python3 core/run/run_stats.py --task input_permuted_mnist_stats --learner si --seed 0 --lr 0.001 --beta_weight 0.999 --beta_importance 0.9 --lamda 1.0 --network fully_connected_relu_with_hooks --n_samples 500000"

python3 core/run/run_stats.py --task input_permuted_mnist_stats --learner si --seed 0 --lr 0.001 --beta_weight 0.999 --beta_importance 0.9 --lamda 1.0 --network fully_connected_relu_with_hooks --n_samples 500000

echo "========================================="
echo "SI test completed"
echo "End time: $(date)"
echo "========================================"