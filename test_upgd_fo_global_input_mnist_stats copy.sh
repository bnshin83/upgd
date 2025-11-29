#!/bin/bash
#SBATCH --job-name=upgd_fo_global_input_mnist_stats_samples_1000000_seed_0_sigma_0.1
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_fo_global_input_mnist_stats_samples_1000000_seed_0_sigma_0.1.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_fo_global_input_mnist_stats_samples_1000000_seed_0_sigma_0.1.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_fo_global_input_mnist_stats_lr_0.01_sigma_0.1_samples_1000000_seed_0_sigma_0.1"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD FO Global Input MNIST Statistics with Enhanced Monitoring"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.1"
echo "Total samples: 1000000"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run UPGD FO Global Input MNIST statistics with run_stats.py
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.1 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1 \
    --save_path logs

echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
1echo "JSON results saved to: logs/input_permuted_mnist_stats/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_n_samples_1000000/0.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}_upgd_fo_global_input_mnist_stats_samples_1000000_seed_0_sigma_0.1.out"
echo "Check the log directory and WandB dashboard for detailed results"
echo "========================================"