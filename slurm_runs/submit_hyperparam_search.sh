#!/bin/bash
# =============================================================================
# Submit Hyperparameter Search Jobs (6 GPUs Parallel)
#
# Uses SLURM array with %6 throttle to run 6 jobs in parallel,
# automatically batching the rest sequentially.
#
# Usage:
#   ./submit_hyperparam_search.sh [mode]
#
# Modes:
#   full     - All 486 experiments (81 batches of 6)
#   cifar10  - Only CIFAR-10 (243 experiments, 41 batches)
#   emnist   - Only EMNIST (243 experiments, 41 batches)
#   quick    - Reduced grid, no WD variations (162 experiments, 27 batches)
#   test     - Quick test (18 experiments, 3 batches)
#
# =============================================================================

MODE=${1:-"full"}

echo "========================================="
echo "🔍 Hyperparameter Search (6 GPUs Parallel)"
echo "========================================="
echo "Mode: ${MODE}"
echo ""

# Grid info:
# Total = 2 datasets * 3 lr * 3 sigma * 3 beta * 3 wd * 3 seeds = 486
# Per dataset = 243
# Without WD variation = 162
# Quick test = 18

case $MODE in
    "full")
        echo "📊 Full Grid Search"
        echo "   - 486 experiments total"
        echo "   - 6 parallel GPUs"
        echo "   - 81 sequential batches"
        echo ""
        JOB_ID=$(sbatch --parsable --array=0-485%6 /scratch/gilbreth/shin283/upgd/hyperparam_search.sh)
        ;;
    "cifar10")
        echo "📊 CIFAR-10 Only"
        echo "   - 243 experiments"
        echo "   - 6 parallel GPUs"
        echo "   - 41 sequential batches"
        echo ""
        # CIFAR-10 is dataset_idx=0, which corresponds to even-numbered configs in the first half
        JOB_ID=$(sbatch --parsable --array=0-242%6 /scratch/gilbreth/shin283/upgd/hyperparam_search.sh)
        ;;
    "emnist")
        echo "📊 EMNIST Only"
        echo "   - 243 experiments"
        echo "   - 6 parallel GPUs"
        echo "   - 41 sequential batches"
        echo ""
        # EMNIST is dataset_idx=1, which corresponds to configs 243-485
        JOB_ID=$(sbatch --parsable --array=243-485%6 /scratch/gilbreth/shin283/upgd/hyperparam_search.sh)
        ;;
    "quick")
        echo "📊 Quick Search (no WD variations)"
        echo "   - 162 experiments (WD=0.0 only)"
        echo "   - 6 parallel GPUs"
        echo "   - 27 sequential batches"
        echo ""
        # WD=0.0 configs: every 9th group of 3 (wd_idx=0 means configs where (idx/3)%3==0)
        # Simplified: first 162 configs with WD=0
        JOB_ID=$(sbatch --parsable --array=0-161%6 /scratch/gilbreth/shin283/upgd/hyperparam_search.sh)
        ;;
    "test")
        echo "📊 Test Run"
        echo "   - 18 experiments"
        echo "   - 6 parallel GPUs"
        echo "   - 3 sequential batches"
        echo ""
        JOB_ID=$(sbatch --parsable --array=0-17%6 /scratch/gilbreth/shin283/upgd/hyperparam_search.sh)
        ;;
    *)
        echo "❌ Unknown mode: ${MODE}"
        echo ""
        echo "Available modes:"
        echo "  full     - All 486 experiments"
        echo "  cifar10  - CIFAR-10 only (243)"
        echo "  emnist   - EMNIST only (243)"
        echo "  quick    - No WD variations (162)"
        echo "  test     - Quick test (18)"
        exit 1
        ;;
esac

echo "========================================="
echo "✅ Submitted Job Array: ${JOB_ID}"
echo ""
echo "📋 Monitor Progress:"
echo "   squeue -u $USER"
echo "   squeue -j ${JOB_ID}"
echo ""
echo "📁 Log Files:"
echo "   ls -la /scratch/gilbreth/shin283/upgd/logs/hpsearch_${JOB_ID}_*.out"
echo "   tail -f /scratch/gilbreth/shin283/upgd/logs/hpsearch_${JOB_ID}_0.out"
echo ""
echo "📊 WandB Dashboard:"
echo "   https://wandb.ai/minds_rl/upgd-hyperparam-search"
echo ""
echo "🔄 Cancel all jobs:"
echo "   scancel ${JOB_ID}"
echo "========================================="
