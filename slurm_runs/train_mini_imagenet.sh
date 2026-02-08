#!/bin/bash
#SBATCH --job-name=upgd_mini_imagenet
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_train_mini_imagenet.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_train_mini_imagenet.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G

# =============================================================================
# Mini-ImageNet Training: UPGD vs Baselines
# 1M steps, 3 seeds per learner
# =============================================================================

set -e

cd /scratch/gilbreth/shin283/upgd
module load cuda
export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

mkdir -p logs

# WandB Configuration
export WANDB_PROJECT="upgd-mini-imagenet"
export WANDB_MODE="online"
export WANDB_API_KEY="9ac056cc70ed02df5b4c069e79ebedf6cf17605d"

echo "========================================="
echo "🚀 Mini-ImageNet Training Experiment"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================="

N_SAMPLES=1000000

# Run UPGD
for seed in 0 1 2; do
    export WANDB_RUN_NAME="upgd_imagenet_${SLURM_JOB_ID}_seed${seed}"
    
    echo ""
    echo "📊 UPGD seed=${seed}"
    echo "WandB Run: ${WANDB_RUN_NAME}"
    echo "Start: $(date)"
    
    python3 core/run/run_stats_with_curvature.py \
        --task label_permuted_mini_imagenet \
        --learner upgd_fo_global \
        --seed ${seed} \
        --lr 0.001 \
        --sigma 0.1 \
        --beta_utility 0.9999 \
        --weight_decay 0.01 \
        --network fcn_relu \
        --n_samples ${N_SAMPLES} \
        --compute_curvature_every 100 \
        --save_path logs
    
    echo "✅ Completed UPGD seed=${seed}"
done

# Run SGD
for seed in 0 1 2; do
    export WANDB_RUN_NAME="sgd_imagenet_${SLURM_JOB_ID}_seed${seed}"
    
    echo ""
    echo "📊 SGD seed=${seed}"
    echo "WandB Run: ${WANDB_RUN_NAME}"
    echo "Start: $(date)"
    
    python3 core/run/run_stats_with_curvature.py \
        --task label_permuted_mini_imagenet \
        --learner sgd \
        --seed ${seed} \
        --lr 0.001 \
        --network fcn_relu \
        --n_samples ${N_SAMPLES} \
        --compute_curvature_every 100 \
        --save_path logs
    
    echo "✅ Completed SGD seed=${seed}"
done

# Run PGD
for seed in 0 1 2; do
    export WANDB_RUN_NAME="pgd_imagenet_${SLURM_JOB_ID}_seed${seed}"
    
    echo ""
    echo "📊 PGD seed=${seed}"
    echo "WandB Run: ${WANDB_RUN_NAME}"
    echo "Start: $(date)"
    
    python3 core/run/run_stats_with_curvature.py \
        --task label_permuted_mini_imagenet \
        --learner pgd \
        --seed ${seed} \
        --lr 0.001 \
        --sigma 0.05 \
        --network fcn_relu \
        --n_samples ${N_SAMPLES} \
        --compute_curvature_every 100 \
        --save_path logs
    
    echo "✅ Completed PGD seed=${seed}"
done

# Run Shrink & Perturb
for seed in 0 1 2; do
    export WANDB_RUN_NAME="sp_imagenet_${SLURM_JOB_ID}_seed${seed}"
    
    echo ""
    echo "📊 Shrink&Perturb seed=${seed}"
    echo "WandB Run: ${WANDB_RUN_NAME}"
    echo "Start: $(date)"
    
    python3 core/run/run_stats_with_curvature.py \
        --task label_permuted_mini_imagenet \
        --learner shrink_and_perturb \
        --seed ${seed} \
        --lr 0.001 \
        --sigma 0.05 \
        --decay 0.01 \
        --network fcn_relu \
        --n_samples ${N_SAMPLES} \
        --compute_curvature_every 100 \
        --save_path logs
    
    echo "✅ Completed Shrink&Perturb seed=${seed}"
done

echo ""
echo "========================================="
echo "✅ All experiments complete"
echo "End time: $(date)"
echo "========================================="
