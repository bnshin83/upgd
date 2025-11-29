#!/bin/bash
#SBATCH --job-name=sgd_input_mnist_stats_samples_1000000_seed_2_sigma_0.1_wd_0.02_half_lr_better_hist_more_bins
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_sgd_input_mnist_stats_samples_1000000_seed_2_sigma_0.1_wd_0.02_half_lr_better_hist_more_bins.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_sgd_input_mnist_stats_samples_1000000_seed_2_sigma_0.1_wd_0.02_half_lr_better_hist_more_bins.err

# Change to the submission directory
cd /scratch/gautschi/shin283/upgd

# Load CUDA and Python modules
module load cuda
module load python

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# WandB Configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_sgd_input_mnist_lr_0.005_sigma_0.1_wd_0.02_seed_2_more_bins"
export WANDB_MODE="online"

echo "========================================="
echo "Running SGD Input MNIST - Matching UPGD Hyperparameters (with noise)"
echo "Testing: Baseline comparison with same lr, wd, sigma, seed as UPGD"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.005 (half of UPGD's 0.01)"
echo "Sigma: 0.1 (matching UPGD noise)"
echo "Weight Decay: 0.02 (matching UPGD)"
echo "Beta Utility: 0.9999 (for utility tracking)"
echo "Seed: 2 (matching UPGD experiment)"
echo "Total samples: 1000000"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "========================================="

# Run SGD Input MNIST with same hyperparameters as UPGD (including noise)
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner sgd \
    --seed 2 \
    --lr 0.005 \
    --sigma 0.1 \
    --weight_decay 0.02 \
    --beta_utility 0.9999 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --save_path logs \
    --compute_curvature_every 1000000 \

echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/input_permuted_mnist_stats/sgd/fully_connected_relu_with_hooks/lr_0.005_sigma_0.1_weight_decay_0.02_beta_utility_0.9999_n_samples_1000000/2_more_bins.json"
echo "WandB Run: ${WANDB_RUN_NAME}" 
echo "========================================="
echo "Comparison:"
echo "  UPGD:       lr=0.01, sigma=0.1, gating=(1-utility) on (grad+noise), wd=0.02"
echo "  SGD (this): lr=0.005, sigma=0.1, no gating, wd=0.02"
echo ""
echo "This isolates UPGD's gating mechanism:"
echo "  - Both use same lr, noise, weight decay, seed"
echo "  - Only difference is UPGD's (1-utility) gating"
echo "  - Tests if selective protection matters beyond noise injection"
echo "========================================"
