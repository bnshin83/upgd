#!/bin/bash
#SBATCH --job-name=test_inputaware_input_mnist_stats_curv_0.01
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/input_mnist_stats_curv_0.01/%j.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/input_mnist_stats_curv_0.01/%j.err

# Change to the submission directory
cd /scratch/gautschi/shin283/upgd

# Load CUDA and Python modules
module load cuda
module load python

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# Install wandb if not already installed
pip install wandb

# Initialize wandb configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="input_mnist_stats_curv_0.01_${SLURM_JOB_ID}"
export WANDB_MODE="online"

echo "========================================="
echo "Running Input-Aware Input Permuted MNIST Statistics with Curvature Tracking"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Compute curvature every 1 step(s)"
echo "Hutchinson samples: 5"
echo "Curvature threshold: 0.01"
echo "Lambda max: 1.0"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "========================================="

# Run input-aware UPGD with maximum protection configuration and curvature tracking
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner input_aware_upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --curvature_threshold 0.01 \
    --lambda_max 1.0 \
    --hutchinson_samples 5 \
    --compute_curvature_every 1

echo "========================================="
echo "Input-aware statistics with curvature tracking completed"
echo "End time: $(date)"
echo "Check logs/ directory for detailed curvature statistics"
echo "========================================="