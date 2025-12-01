#!/bin/bash
#SBATCH --job-name=early_phase_utility
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_early_phase_utility.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_early_phase_utility.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G

# =============================================================================
# Early-Phase and Layer-wise Utility Dynamics Experiment
# From 11292025.md lines 294-309
#
# Goal: Show that Mini-ImageNet has wider/structured early-phase utilities 
#       (and layer differences), while input-permuted MNIST rapidly collapses 
#       to uniform ~0.5 everywhere.
# =============================================================================

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

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# WandB Configuration
export WANDB_PROJECT="upgd-utility-dynamics"
export WANDB_MODE="online"
export WANDB_API_KEY="9ac056cc70ed02df5b4c069e79ebedf6cf17605d"

echo "========================================="
echo "🚀 Early-Phase Utility Dynamics Experiment"
echo "Start time: $(date)"
echo "========================================="

# Function to run UPGD experiment with task-specific hyperparameters
run_upgd_experiment() {
    local task=$1           # e.g., "input_permuted_mnist_stats"
    local learner=$2        # e.g., "upgd_fo_global"
    local seed=$3           # e.g., 0, 1, 2
    local n_samples=$4      # e.g., 50000
    local description=$5    # Human-readable description
    
    # Task-specific hyperparameters
    # ==============================
    # imnist:   α = 0.01, σ = 0.1,   βu = 0.9999, λ = 0.01
    # imagenet: α = 0.01, σ = 0.001, βu = 0.9,    λ = 0.0
    # emnist:   α = 0.01, σ = 0.001, βu = 0.9,    λ = 0.0
    # cifar10:  α = 0.01, σ = 0.001, βu = 0.999,  λ = 0.0
    
    local lr=0.01
    local sigma
    local beta_utility
    local weight_decay
    
    case ${task} in
        *mnist*)
            # Input-permuted MNIST
            sigma=0.1
            beta_utility=0.9999
            weight_decay=0.01
            ;;
        *imagenet*)
            # Mini-ImageNet
            sigma=0.001
            beta_utility=0.9
            weight_decay=0.0
            ;;
        *emnist*)
            # EMNIST
            sigma=0.001
            beta_utility=0.9
            weight_decay=0.0
            ;;
        *cifar*)
            # CIFAR-10
            sigma=0.001
            beta_utility=0.999
            weight_decay=0.0
            ;;
        *)
            # Default fallback
            sigma=0.1
            beta_utility=0.9
            weight_decay=0.0
            ;;
    esac
    
    export WANDB_RUN_NAME="${SLURM_JOB_ID}_${learner}_${task}_seed${seed}"
    
    echo ""
    echo "========================================="
    echo "📊 ${description}"
    echo "Task: ${task}"
    echo "Learner: ${learner}"
    echo "Seed: ${seed}"
    echo "Samples: ${n_samples}"
    echo "Hyperparameters: lr=${lr}, sigma=${sigma}, beta_utility=${beta_utility}, weight_decay=${weight_decay}"
    echo "WandB Run: ${WANDB_RUN_NAME}"
    echo "Start: $(date)"
    echo "========================================="
    
    python3 core/run/run_stats_with_curvature.py \
        --task ${task} \
        --learner ${learner} \
        --seed ${seed} \
        --lr ${lr} \
        --sigma ${sigma} \
        --beta_utility ${beta_utility} \
        --weight_decay ${weight_decay} \
        --network fully_connected_relu_with_hooks \
        --n_samples ${n_samples} \
        --compute_curvature_every 1 \
        --save_path logs
    
    echo "✅ Completed: ${description}"
    echo "End: $(date)"
}

# =============================================================================
# Phase 1: Input-Permuted MNIST
# Hyperparameters: α = 0.01, σ = 0.1, βu = 0.9999, λ = 0.01
# =============================================================================
echo ""
echo "📈 Phase 1: Input-Permuted MNIST"
echo "================================="

for seed in 0 1 2; do
    run_upgd_experiment \
        "input_permuted_mnist_stats" \
        "upgd_fo_global" \
        ${seed} \
        50000 \
        "Input-Permuted MNIST - UPGD FO Global (seed=${seed})"
done

# =============================================================================
# Phase 2: Label-Permuted Mini-ImageNet
# Hyperparameters: α = 0.01, σ = 0.001, βu = 0.9, λ = 0.0
# =============================================================================
echo ""
echo "📈 Phase 2: Label-Permuted Mini-ImageNet"
echo "========================================="

for seed in 0 1 2; do
    run_upgd_experiment \
        "label_permuted_mini_imagenet_stats" \
        "upgd_fo_global" \
        ${seed} \
        50000 \
        "Label-Permuted Mini-ImageNet - UPGD FO Global (seed=${seed})"
done

# =============================================================================
# Phase 3: Label-Permuted EMNIST
# Hyperparameters: α = 0.01, σ = 0.001, βu = 0.9, λ = 0.0
# =============================================================================
echo ""
echo "📈 Phase 3: Label-Permuted EMNIST"
echo "=================================="

for seed in 0 1 2; do
    run_upgd_experiment \
        "label_permuted_emnist_stats" \
        "upgd_fo_global" \
        ${seed} \
        50000 \
        "Label-Permuted EMNIST - UPGD FO Global (seed=${seed})"
done

# =============================================================================
# Phase 4: Label-Permuted CIFAR-10
# Hyperparameters: α = 0.01, σ = 0.001, βu = 0.999, λ = 0.0
# =============================================================================
echo ""
echo "📈 Phase 4: Label-Permuted CIFAR-10"
echo "===================================="

for seed in 0 1 2; do
    run_upgd_experiment \
        "label_permuted_cifar10_stats" \
        "upgd_fo_global" \
        ${seed} \
        50000 \
        "Label-Permuted CIFAR-10 - UPGD FO Global (seed=${seed})"
done

echo ""
echo "========================================="
echo "✅ Early-Phase Utility Experiment Complete"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "📊 Results Analysis:"
echo "1. Check WandB dashboard: upgd-utility-dynamics"
echo "2. JSON results in: logs/"
echo "3. Compare utility/hist_* bins between datasets"
echo ""
echo "Hyperparameters used:"
echo "- imnist:   σ=0.1,   βu=0.9999, λ=0.01"
echo "- imagenet: σ=0.001, βu=0.9,    λ=0.0"
echo "- emnist:   σ=0.001, βu=0.9,    λ=0.0"
echo "- cifar10:  σ=0.001, βu=0.999,  λ=0.0"
echo "========================================="

