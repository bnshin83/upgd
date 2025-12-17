#!/bin/bash
#SBATCH --job-name=snp_mini_imagenet_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_snp_mini_imagenet_seed_2.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_snp_mini_imagenet_seed_2.err

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
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_snp_mini_imagenet_seed_2"
export WANDB_MODE="online"

echo "========================================="
echo "Running S&P (Shrink & Perturb) Mini-ImageNet Baseline"
echo "Using paper hyperparameters from optimizer_best_sets.csv"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Dataset: Mini-ImageNet (100 classes, ResNet50 bottleneck features, 2048-dim)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.01"
echo "Beta Utility: 0.9"
echo "Weight Decay: 0.001"
echo "Seed: 2"
echo "Total samples: 1000000"
echo "Network: fully_connected_relu_with_hooks"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run S&P Mini-ImageNet with paper hyperparameters
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_mini_imagenet_stats \
    --learner sgd \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.01 \
    --beta_utility 0.9 \
    --weight_decay 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "========================================="
echo "S&P Baseline experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/label_permuted_mini_imagenet_stats/sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.9_weight_decay_0.001_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}_snp_mini_imagenet_seed_2.out"
echo "========================================"
