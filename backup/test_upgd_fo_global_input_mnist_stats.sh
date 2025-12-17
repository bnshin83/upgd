#!/bin/bash
#SBATCH --job-name=upgd_fo_global_input_mnist_stats_samples_1000000_seed_2_centered_linear_with_utility_norms_histograms_beta_utility_0.9
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_fo_global_input_mnist_stats_samples_1000000_seed_2_centered_linear_with_utility_norms_histograms_more_bins_beta_utility_0.9.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_fo_global_input_mnist_stats_samples_1000000_seed_2_centered_linear_with_utility_norms_histograms_more_bins_beta_utility_0.9.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_fo_global_input_mnist_stats_samples_1000000_seed_2_centered_linear_with_utility_norms_histograms_beta_utility_0.9"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD FO Global Input MNIST - Paper Hyperparameters (Table 3)"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.1 (paper value)"
echo "Beta Utility: 0.9 (paper value)"
echo "Weight Decay: 0.01 (paper value)"
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
    --seed 2 \
    --lr 0.005 \
    --sigma 0.1 \
    --beta_utility 0.9 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1 \
    --save_path logs

echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/input_permuted_mnist_stats/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9_weight_decay_0.01_seed_2_n_samples_1000000/0.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}_upgd_fo_global_input_mnist_stats_samples_1000000_seed_2.out"
echo "Check the log directory and WandB dashboard for detailed results"
echo "========================================"