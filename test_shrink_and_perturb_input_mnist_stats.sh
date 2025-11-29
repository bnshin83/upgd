#!/bin/bash
#SBATCH --job-name=shrink_and_perturb_input_mnist_stats_samples_1000000_seed_2_sigma_0.1_weight_decay_0.01_half_lr
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_shrink_and_perturb_input_mnist_stats_samples_1000000_seed_2_sigma_0.1_weight_decay_0.01_half_lr.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_shrink_and_perturb_input_mnist_stats_samples_1000000_seed_2_sigma_0.1_weight_decay_0.01_half_lr.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_shrink_and_perturb_input_mnist_lr_0.005_sigma_0.1_weight_decay_0.01_seed_2"
export WANDB_MODE="online"

echo "========================================="
echo "Running Shrink and Perturb Input MNIST - Matching SGD Hyperparameters"
echo "Testing: Baseline comparison with same lr, weight_decay, sigma, seed as SGD"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.005 (half of UPGD's 0.01)"
echo "Sigma: 0.1 (matching noise)"
echo "Weight Decay: 0.01 (matching weight decay)"
echo "Seed: 2 (matching experiment)"
echo "Total samples: 1000000"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "========================================="

# Run Shrink and Perturb Input MNIST
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner shrink_and_perturb \
    --seed 2 \
    --lr 0.005 \
    --sigma 0.1 \
    --weight_decay 0.01 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1 \
    --save_path logs

echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/input_permuted_mnist_stats/shrink_and_perturb/fully_connected_relu_with_hooks/lr_0.005_sigma_0.1_weight_decay_0.01_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "========================================="
echo "Comparison:"
echo "  UPGD:                lr=0.01, sigma=0.1, gating=(1-utility) on (grad+noise), wd=0.01"
echo "  SGD:                 lr=0.005, sigma=0.1, no gating, wd=0.01"
echo "  Shrink&Perturb(this): lr=0.005, sigma=0.1, no gating, wd=0.01"
echo ""
echo "Note: Shrink&Perturb is mathematically identical to SGD"
echo "  - Both: w ← w·(1 - α·β) - α·(∂L/∂w + ξ)"
echo "  - Different implementation: mul_() then add_() vs single add_()"
echo "========================================="
