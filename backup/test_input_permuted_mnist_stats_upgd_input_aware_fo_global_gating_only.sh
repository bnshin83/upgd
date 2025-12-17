#!/bin/bash
#SBATCH --job-name=upgd_input_aware_gating_only_curv_0.05_samples_1000000_seed_2_lambda_max_1.0_lambda_scale_0.001
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_input_aware_gating_only_curv_0.05_samples_1000000_seed_2_lambda_max_1.0_lambda_scale_0.001.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_input_aware_gating_only_curv_0.05_samples_1000000_seed_2_lambda_max_1.0_lambda_scale_0.001.err

# Change to the submission directory
cd /scratch/gautschi/shin283/upgd

# Load CUDA and Python modules
module load cuda
module load python

# Set PYTHONPATH first (before activating venv to ensure priority)
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# wandb should already be installed in the environment

# Initialize wandb configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_input_aware_gating_only_curv_0.05_samples_1000000_seed_2_lambda_max_1.0_lambda_scale_0.001"
export WANDB_MODE="online"

echo "========================================="
echo "Running Input-Aware UPGD Input Permuted MNIST Statistics - GATING ONLY"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.1"
echo "Beta Utility: 0.9999 (paper value)"
echo "Weight Decay: 0.01 (paper value)"
echo "Curvature threshold: 0.05"
echo "Lambda max: 1.0"
echo "Lambda scale: 0.001 (EXPLICITLY SET)"
echo "Hutchinson samples: 5"
echo "Compute curvature every: 1 step(s)"
echo "Total samples: 1000000"
echo "Configuration: GATING ONLY (disable_regularization=True, disable_gating=False)"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run input-aware UPGD with GATING ONLY (disable regularization)
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_input_aware_fo_global \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.1 \
    --beta_utility 0.9999 \
    --weight_decay 0.01 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --curvature_threshold 0.05 \
    --lambda_max 1.0 \
    --lambda_scale 0.001 \
    --hutchinson_samples 5 \
    --compute_curvature_every 1 \
    --disable_regularization True \
    --disable_gating False \
    --save_path logs

echo "========================================="
echo "Experiment completed - GATING ONLY"
echo "End time: $(date)"
echo "JSON results saved to: logs/input_permuted_mnist_stats/input_aware_upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_curvature_threshold_0.05_lambda_max_1.0_lambda_scale_0.001_hutchinson_samples_5_n_samples_1000000_disable_regularization_True_disable_gating_False/0.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}.out"
echo "Check the log directory and WandB dashboard for detailed results"
echo "========================================="