#!/bin/bash
#SBATCH --job-name=upgd_imagenet
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_imagenet.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_imagenet.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G

# =============================================================================
# Phase 2: Label-Permuted Mini-ImageNet (Single Seed)
# Hyperparameters: α = 0.01, σ = 0.01, βu = 0.9, λ = 0.001
# =============================================================================

set -e

cd /scratch/gilbreth/shin283/upgd
module load cuda
export PYTHONPATH=/scratch/gilbreth/shin283/upgd:/scratch/gilbreth/shin283/upgd/logs:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

mkdir -p logs

# WandB Configuration
export WANDB_PROJECT="upgd-utility-dynamics"
export WANDB_MODE="online"
export WANDB_API_KEY="9ac056cc70ed02df5b4c069e79ebedf6cf17605d"
export WANDB_RUN_NAME="upgd_imagenet_lr0.01_sigma0.01_beta0.9_wd0.001_${SLURM_JOB_ID}_seed0"

echo "========================================="
echo "🚀 Phase 2: Label-Permuted Mini-ImageNet"
echo "Hyperparameters: lr=0.01, sigma=0.01, beta_utility=0.9, weight_decay=0.001"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "========================================="

python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_mini_imagenet_stats \
    --learner upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --beta_utility 0.9 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 100000 \
    --compute_curvature_every 1 \
    --save_path logs

echo ""
echo "========================================="
echo "✅ Phase 2 Complete"
echo "End time: $(date)"
echo "========================================="
