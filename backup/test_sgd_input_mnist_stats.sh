#!/bin/bash
#SBATCH --job-name=sgd_input_mnist_stats_samples_1000000_seed_0_wd_0.02
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_sgd_input_mnist_stats_samples_1000000_seed_0_wd_0.02.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_sgd_input_mnist_stats_samples_1000000_seed_0_wd_0.02.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_sgd_input_mnist_stats_lr_0.01_wd_0.02_samples_1000000_seed_0"
export WANDB_MODE="online"

echo "========================================="
echo "Running SGD Input MNIST Statistics with Enhanced Monitoring"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Weight Decay: 0.02"
echo "Beta Utility: 0.9999 (for utility tracking)"
echo "Total samples: 1000000"
echo "Computing: Curvature, Utility, Utility Norms, Utility Histograms"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run SGD Input MNIST statistics with run_stats_with_curvature.py
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner sgd \
    --seed 0 \
    --lr 0.005 \
    --weight_decay 0.02 \
    --beta_utility 0.9999 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1 \
    --save_path logs

echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/input_permuted_mnist_stats/sgd/fully_connected_relu_with_hooks/lr_0.01_weight_decay_0.0001_n_samples_1000000/0.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}_sgd_input_mnist_stats_samples_1000000_seed_0.out"
echo "Check the log directory and WandB dashboard for detailed results"
echo "========================================"