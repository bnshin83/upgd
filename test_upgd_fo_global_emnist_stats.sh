#!/bin/bash
#SBATCH --job-name=upgd_fo_global_emnist_seed_2_baseline_paper_hp
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_fo_global_emnist_seed_2_baseline_paper_hp.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_fo_global_emnist_seed_2_baseline_paper_hp.err

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
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_fo_global_emnist_seed_2_baseline_paper_hp"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD FO Global EMNIST Baseline (for comparison with Input-Aware)"
echo "Using best hyperparameters from the UPGD-W paper"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Dataset: EMNIST (47 classes, 28x28 grayscale images)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.001 (paper value)"
echo "Beta Utility: 0.9 (paper value)"
echo "Weight Decay (lambda): 0.0 (paper value)"
echo "Total samples: 1000000"
echo "Compute curvature every: 1 step(s)"
echo "Network: fully_connected_relu_with_hooks"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run UPGD FO Global EMNIST baseline with same config as input-aware
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_emnist_stats \
    --learner upgd_fo_global \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.001 \
    --beta_utility 0.9 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1 \
    --save_path logs

echo "========================================="
echo "Baseline experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/label_permuted_emnist_stats/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}.out"
echo "Check the log directory and WandB dashboard for detailed results"
echo "========================================"
