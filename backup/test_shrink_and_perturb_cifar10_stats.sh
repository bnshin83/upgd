#!/bin/bash
#SBATCH --job-name=shrink_and_perturb_cifar10_seed_2_lr_0.01_weight_decay_0.00_sigma_0.001
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_shrink_and_perturb_cifar10_seed_2_lr_0.01_weight_decay_0.00_sigma_0.001.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_shrink_and_perturb_cifar10_seed_2_lr_0.01_weight_decay_0.00_sigma_0.001.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_shrink_and_perturb_cifar10_seed_2_lr_0.01_weight_decay_0.00_sigma_0.001"
export WANDB_MODE="online"

echo "========================================="
echo "Running Shrink and Perturb CIFAR-10 Baseline"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Dataset: CIFAR-10 (10 classes, 32x32x3 RGB images)"
echo "Learning Rate: 0.01"
echo "Weight Decay: 0.00"
echo "Sigma: 0.001"
echo "Total samples: 1000000"
echo "Compute curvature every: 1000000 step(s) (disabled)"
echo "Network: fully_connected_relu_with_hooks"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run Shrink and Perturb CIFAR-10 baseline
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_cifar10_stats \
    --learner shrink_and_perturb \
    --seed 2 \
    --lr 0.01 \
    --weight_decay 0.00 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1 \
    --save_path logs

echo "========================================="
echo "Shrink and Perturb experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/label_permuted_cifar10_stats/shrink_and_perturb/fully_connected_relu_with_hooks/lr_0.01_weight_decay_0.00_sigma_0.001_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}_shrink_and_perturb_cifar10_seed_2_lr_0.01_weight_decay_0.00_sigma_0.001.out"
echo "Check the log directory and WandB dashboard for detailed results"
echo "========================================="
