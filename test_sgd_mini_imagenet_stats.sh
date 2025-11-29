#!/bin/bash
#SBATCH --job-name=sgd_mini_imagenet_seed_2_fair_comparison_lr_0.005_wd_0.002
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_sgd_mini_imagenet_seed_2_fair_comparison_lr_0.005_wd_0.002.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_sgd_mini_imagenet_seed_2_fair_comparison_lr_0.005_wd_0.002.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_sgd_mini_imagenet_seed_2_fair_comparison_lr_0.005_wd_0.002"
export WANDB_MODE="online"

echo "========================================="
echo "Running SGD Mini-ImageNet Fair Comparison (with UPGD FO Global)"
echo "Matching shrink factor: lr'=0.5×lr_upgd, wd'=2×wd_upgd for u≈0.5"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Dataset: Mini-ImageNet (100 classes, ResNet50 bottleneck features, 2048-dim)"
echo "Learning Rate: 0.005 (= 0.5 × UPGD's 0.01)"
echo "Beta Utility: 0.9 (for utility tracking)"
echo "Weight Decay: 0.002 (= 2 × UPGD's 0.001)"
echo "Seed: 2"
echo "Total samples: 1000000"
echo "Network: fully_connected_relu_with_hooks"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "NOTE: This requires Mini-ImageNet dataset files in dataset/ directory"
echo "      (mini-imagenet_targets.pkl and processed_imagenet.pkl)"
echo "========================================="

# Run SGD Mini-ImageNet with matched shrink factor (fair comparison)
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_mini_imagenet_stats \
    --learner sgd \
    --seed 2 \
    --lr 0.005 \
    --sigma 0.01 \
    --beta_utility 0.9 \
    --weight_decay 0.002 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "========================================="
echo "SGD Fair Comparison experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/label_permuted_mini_imagenet_stats/sgd/fully_connected_relu_with_hooks/lr_0.005_sigma_0.005_beta_utility_0.9_weight_decay_0.002_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}.out"
echo "========================================="
echo "Fair Comparison (matched shrink factor for u≈0.5):"
echo "  UPGD: lr=0.01, sigma=0.005, wd=0.001, gating=(1-utility) on (grad+noise)"
echo "  SGD:  lr=0.005, sigma=0.005, wd=0.002, no gating"
echo ""
echo "Rationale:"
echo "  UPGD applies: θ ← (1-ηλ)θ - η(∇L+σε)(1-u)"
echo "  With u≈0.5: full decay (1-ηλ) but ~0.5η effective gradient lr"
echo "  SGD matches: lr'=0.5η, wd'=2λ → same shrink factor (1-ηλ)"
echo "  This isolates utility gating's anisotropy from decay mismatch"
echo "========================================"
