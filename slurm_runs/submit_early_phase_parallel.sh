#!/bin/bash
# =============================================================================
# Submit Early-Phase Utility Experiments in Parallel
# Submits up to 6 jobs at once (one per GPU)
#
# Usage: ./submit_early_phase_parallel.sh [max_jobs]
#   max_jobs: Maximum concurrent jobs (default: 6)
# =============================================================================

MAX_JOBS=${1:-6}
SCRIPT="/scratch/gilbreth/shin283/upgd/run_early_phase_single.sh"

echo "🚀 Submitting Early-Phase Utility Experiments (max ${MAX_JOBS} parallel)"
echo "========================================="

# Define all experiments: TASK DATASET_NAME LEARNER SEED N_SAMPLES LR SIGMA BETA_UTILITY WEIGHT_DECAY
# Dataset naming: imnist (input-permuted MNIST), emnist (label-permuted EMNIST), cifar10, mini-imagenet
# Only UPGD (not input-aware)
EXPERIMENTS=(
    # 1. Input-Permuted MNIST (imnist)
    "input_permuted_mnist_stats imnist upgd_fo_global 0 50000 0.01 0.1 0.9 0.0"
    
    # 2. Label-Permuted EMNIST (emnist)
    "label_permuted_emnist_stats emnist upgd_fo_global 0 50000 0.01 0.001 0.9 0.0"
    
    # 3. Label-Permuted CIFAR-10 (cifar10)
    "label_permuted_cifar10_stats cifar10 upgd_fo_global 0 50000 0.01 0.001 0.999 0.0"
    
    # 4. Label-Permuted Mini-ImageNet (mini-imagenet)
    "label_permuted_mini_imagenet_stats mini-imagenet upgd_fo_global 0 50000 0.01 0.1 0.9 0.0"
)

SUBMITTED=0
JOB_IDS=()

for exp in "${EXPERIMENTS[@]}"; do
    read -r TASK DATASET_NAME LEARNER SEED N_SAMPLES LR SIGMA BETA_UTILITY WEIGHT_DECAY <<< "$exp"
    
    # Check if we've hit the max jobs limit
    if [ $SUBMITTED -ge $MAX_JOBS ]; then
        echo "⚠️  Reached max jobs limit (${MAX_JOBS}). Remaining experiments not submitted."
        break
    fi
    
    echo "📤 Submitting: upgd on ${DATASET_NAME} (lr=${LR}, sigma=${SIGMA}, beta=${BETA_UTILITY})"
    JOB_ID=$(sbatch --parsable ${SCRIPT} ${TASK} ${DATASET_NAME} ${LEARNER} ${SEED} ${N_SAMPLES} ${LR} ${SIGMA} ${BETA_UTILITY} ${WEIGHT_DECAY})
    JOB_IDS+=($JOB_ID)
    echo "   → Job ID: ${JOB_ID}"
    
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "========================================="
echo "✅ Submitted ${SUBMITTED} jobs"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with: squeue -u $USER"
echo "WandB: https://wandb.ai/minds_rl/upgd-utility-dynamics"
echo "========================================="

