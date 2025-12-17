#!/bin/bash
#SBATCH --job-name=test_small_sample
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:10:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_small_sample.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_small_sample.err

# Test script with small sample size to verify the fixes
cd /scratch/gautschi/shin283/upgd

# Load modules
module load cuda
module load python

# Activate environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

echo "========================================="
echo "Testing Input-Aware with Small Sample Size"
echo "Start time: $(date)"
echo "Testing with 100 samples to verify step-level saving"
echo "========================================="

# Run with very small sample size for testing
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner input_aware_upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 100 \
    --curvature_threshold 0.5 \
    --lambda_max 1.0 \
    --hutchinson_samples 10 \
    --compute_curvature_every 1

echo "========================================="
echo "Test completed"
echo "End time: $(date)"
echo "Check logs/input_permuted_mnist_stats/input_aware_upgd_fo_global/ for results"
echo "========================================="