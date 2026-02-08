#!/bin/bash
#SBATCH --job-name=snp_imnist_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_snp_imnist_seed_2.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_snp_imnist_seed_2.err

# Change to the submission directory
cd /scratch/gilbreth/shin283/upgd

# Load CUDA and Python modules
module load cuda

# Set PYTHONPATH first (before activating venv to ensure priority)
export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH

# Activate the UPGD virtual environment
# Activate the UPGD conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_snp_imnist_seed_2"
export WANDB_MODE="online"

echo "========================================="
echo "Running S&P (Shrink & Perturb) Input-MNIST Baseline"
echo "Using paper hyperparameters from optimizer_best_sets.csv"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Dataset: Input-Permuted MNIST (10 classes)"
echo "Learning Rate: 0.001"
echo "Sigma: 0.1"
echo "Beta Utility: 0.9999"
echo "Weight Decay: 0.01"
echo "Seed: 2"
echo "Total samples: 1000000"
echo "Network: fully_connected_relu_with_hooks"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run S&P Input-MNIST with paper hyperparameters
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner sgd \
    --seed 2 \
    --lr 0.001 \
    --sigma 0.1 \
    --beta_utility 0.9999 \
    --weight_decay 0.01 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "========================================="
echo "S&P Baseline experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/input_permuted_mnist_stats/sgd/fully_connected_relu_with_hooks/lr_0.001_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gilbreth/shin283/upgd/logs/${SLURM_JOB_ID}_snp_imnist_seed_2.out"
echo "========================================"
