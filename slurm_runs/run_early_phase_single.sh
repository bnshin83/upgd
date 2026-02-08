#!/bin/bash
#SBATCH --job-name=early_phase_%j
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_early_phase.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_early_phase.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G

# =============================================================================
# Parametric Early-Phase Utility Dynamics Experiment
# Usage: sbatch run_early_phase_single.sh <task> <dataset_name> <learner> <seed> [n_samples] [lr] [sigma] [beta_utility] [weight_decay]
#
# Dataset naming: imnist, emnist, cifar10, mini-imagenet
# WandB naming: optimizer_dataset_lr_sigma_beta_jobid
#
# Examples:
#   sbatch run_early_phase_single.sh input_permuted_mnist_stats imnist upgd_fo_global 0 50000 0.01 0.1 0.9 0.0
#   sbatch run_early_phase_single.sh label_permuted_cifar10_stats cifar10 upgd_fo_global 0 50000 0.01 0.001 0.999 0.0
# =============================================================================

# Parameters (from command line or defaults)
TASK=${1:-"input_permuted_mnist_stats"}
DATASET_NAME=${2:-"imnist"}
LEARNER=${3:-"upgd_fo_global"}
SEED=${4:-0}
N_SAMPLES=${5:-50000}
LR=${6:-0.01}
SIGMA=${7:-0.1}
BETA_UTILITY=${8:-0.9}
WEIGHT_DECAY=${9:-0.0}

# Derive optimizer name from learner (e.g., upgd_fo_global -> upgd)
OPTIMIZER=$(echo ${LEARNER} | cut -d'_' -f1)

# Stop on first error
set -e

# Load environment
cd /scratch/gilbreth/shin283/upgd

# Load modules (Gilbreth cluster)
module load cuda

# Set PYTHONPATH
export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH

# Activate the UPGD conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# Create logs directory
mkdir -p logs

# WandB Configuration
# Naming: optimizer_dataset_lr_sigma_beta_jobid
export WANDB_PROJECT="upgd-utility-dynamics"
export WANDB_MODE="online"
export WANDB_API_KEY="9ac056cc70ed02df5b4c069e79ebedf6cf17605d"
export WANDB_RUN_NAME="${OPTIMIZER}_${DATASET_NAME}_lr${LR}_sigma${SIGMA}_beta${BETA_UTILITY}_${SLURM_JOB_ID}"

# Verify environment
echo "========================================="
echo "🚀 Early-Phase Utility Experiment (Parametric)"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task: ${TASK}"
echo "Learner: ${LEARNER}"
echo "Seed: ${SEED}"
echo "Samples: ${N_SAMPLES}"
echo "LR: ${LR}"
echo "Sigma: ${SIGMA}"
echo "Beta Utility: ${BETA_UTILITY}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "Start: $(date)"
echo "========================================="

# Run experiment
python3 core/run/run_stats_with_curvature.py \
    --task ${TASK} \
    --learner ${LEARNER} \
    --seed ${SEED} \
    --lr ${LR} \
    --sigma ${SIGMA} \
    --beta_utility ${BETA_UTILITY} \
    --weight_decay ${WEIGHT_DECAY} \
    --network fully_connected_relu_with_hooks \
    --n_samples ${N_SAMPLES} \
    --compute_curvature_every 100000 \
    --save_path logs

echo "========================================="
echo "✅ Experiment Complete"
echo "End: $(date)"
echo "========================================="

