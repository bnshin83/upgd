#!/bin/bash
#SBATCH --job-name=upgd_imnist
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_imnist_lr0.01_sigma0.1_beta0.9999.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_imnist_lr0.01_sigma0.1_beta0.9999.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G

# =============================================================================
# Phase 1: Input-Permuted MNIST
# Hyperparameters: α = 0.01, σ = 0.1, βu = 0.9999, λ = 0.01
# =============================================================================

set -e

cd /scratch/gilbreth/shin283/upgd
module load cuda
export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

mkdir -p logs

# WandB Configuration
export WANDB_PROJECT="upgd-utility-dynamics"
export WANDB_MODE="online"
export WANDB_API_KEY="9ac056cc70ed02df5b4c069e79ebedf6cf17605d"

echo "========================================="
echo "🚀 Phase 1: Input-Permuted MNIST"
echo "Hyperparameters: lr=0.01, sigma=0.1, beta_utility=0.9999, weight_decay=0.01"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================="

for seed in 0 1 2; do
    export WANDB_RUN_NAME="upgd_imnist_lr0.01_sigma0.1_beta0.9999_${SLURM_JOB_ID}_seed${seed}"
    
    echo ""
    echo "📊 Running seed=${seed}"
    echo "WandB Run: ${WANDB_RUN_NAME}"
    echo "Start: $(date)"
    
    python3 core/run/run_stats_with_curvature.py \
        --task input_permuted_mnist_stats \
        --learner upgd_fo_global \
        --seed ${seed} \
        --lr 0.01 \
        --sigma 0.1 \
        --beta_utility 0.9999 \
        --weight_decay 0.01 \
        --network fully_connected_relu_with_hooks \
        --n_samples 50000 \
        --compute_curvature_every 1 \
        --save_path logs
    
    echo "✅ Completed seed=${seed}"
done

echo ""
echo "========================================="
echo "✅ Phase 1 Complete"
echo "End time: $(date)"
echo "========================================="

