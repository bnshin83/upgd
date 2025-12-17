#!/bin/bash
#SBATCH --job-name=sgd_curv_gating_input_mnist_stats_samples_1000000_seed_2_scale_0.1_histogram
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_sgd_curv_gating_input_mnist_stats_samples_1000000_seed_2_scale_0.1_histogram.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_sgd_curv_gating_input_mnist_stats_samples_1000000_seed_2_scale_0.1_histogram.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_sgd_curvature_gating_lr_0.01_sigma_0.1_curv_threshold_0.01_scale_0.1_seed_2_histogram"
export WANDB_MODE="online"

echo "========================================="
echo "Running SGD with Curvature-Based Gating"
echo "Testing: Option D - g = sigmoid(-(κ - τ) / s)"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.1 (noise)"
echo "Weight Decay: 0.01"
echo "Beta Utility: 0.9999 (for monitoring)"
echo "Curvature Threshold (τ): 0.01"
echo "Curvature Scale (s): 0.1 (smoother transition)"
echo "Beta Curvature: 0.9"
echo "Seed: 2"
echo "Total samples: 1000000"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "========================================="

# Run SGD with curvature-based gating
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner sgd_curvature_gating \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.1 \
    --weight_decay 0.01 \
    --beta_utility 0.9999 \
    --curvature_threshold 0.01 \
    --curvature_scale 0.1 \
    --beta_curvature 0.9 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1 \
    --save_path logs

echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/input_permuted_mnist_stats/sgd_curvature_gating/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_weight_decay_0.01_curvature_threshold_0.01_curvature_scale_0.1_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "========================================="
echo "Gating Equation:"
echo "  g = sigmoid(-(κ - τ) / s)"
echo "  where:"
echo "    κ = current input curvature"
echo "    τ = 0.01 (threshold)"
echo "    s = 0.1 (scale, smoother transition)"
echo ""
echo "Behavior:"
echo "  High curvature (κ >> τ) → g ≈ 0 (strong protection)"
echo "  Low curvature (κ << τ) → g ≈ 1 (normal update)"
echo "  κ = τ → g = 0.5"
echo ""
echo "Comparison:"
echo "  UPGD:                    gating = (1 - utility), utility-based protection"
echo "  SGD Curvature Gating:    gating = sigmoid(-(κ-τ)/s), curvature-based protection"
echo "  SGD (half-lr):           no gating, lr=0.005"
echo "  SGD (normal):            no gating, lr=0.01"
echo "========================================"
