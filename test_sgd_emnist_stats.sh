#!/bin/bash
#SBATCH --job-name=sgd_emnist_seed_2_baseline_lr_0.005
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_sgd_emnist_seed_2_baseline_lr_0.005.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_sgd_emnist_seed_2_baseline_lr_0.005.err

# Change to the submission directory
cd /scratch/gautschi/shin283/upgd

# Load CUDA and Python modules
module load cuda
module load python

# Set PYTHONPATH first (before activating venv to ensure priority)
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_sgd_emnist_seed_2_baseline_lr_0.005"
export WANDB_MODE="online"

echo "========================================="
echo "Running SGD EMNIST Baseline (for comparison with UPGD)"
echo "Using half the learning rate from the UPGD-W paper"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Dataset: EMNIST (47 classes, 28x28 grayscale images)"
echo "Learning Rate: 0.005 (half of UPGD's 0.01)"
echo "Sigma: 0.001 (matching UPGD paper value)"
echo "Beta Utility: 0.9 (for utility tracking)"
echo "Weight Decay: 0.0 (matching UPGD paper value)"
echo "Seed: 2"
echo "Total samples: 1000000"
echo "Network: fully_connected_relu_with_hooks"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run SGD EMNIST baseline with same hyperparameters as UPGD (half lr)
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_emnist_stats \
    --learner sgd \
    --seed 2 \
    --lr 0.005 \
    --sigma 0.001 \
    --beta_utility 0.9 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "========================================="
echo "SGD Baseline experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/label_permuted_emnist_stats/sgd/fully_connected_relu_with_hooks/lr_0.005_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}.out"
echo "========================================="
echo "Comparison:"
echo "  UPGD: lr=0.01, sigma=0.001, gating=(1-utility) on (grad+noise), wd=0.0"
echo "  SGD:  lr=0.005, sigma=0.001, no gating, wd=0.0"
echo ""
echo "This isolates UPGD's gating mechanism:"
echo "  - Both use same noise, weight decay, seed"
echo "  - SGD uses half learning rate"
echo "  - Only difference is UPGD's (1-utility) gating"
echo "========================================"
