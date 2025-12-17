#!/bin/bash
#SBATCH --job-name=test_charts_verification
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_charts_verification.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_charts_verification.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_test_charts_verification_upgd_fo_global_samples_100_seed_0"
export WANDB_MODE="online"

echo "========================================="
echo "Testing Charts Tab Creation - Standard UPGD with Small Sample Size"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.001"
echo "Total samples: 100"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run UPGD FO Global with very small sample size to ensure Charts tab creation
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 100 \
    --compute_curvature_every 100 \
    --save_path logs

echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/input_permuted_mnist_stats/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_n_samples_100/0.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}_test_charts_verification.out"
echo "Check WandB dashboard to verify Charts tab is present"
echo "========================================"