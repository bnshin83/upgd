#!/bin/bash
#SBATCH --job-name=incr_cifar_upgd
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_incr_cifar_upgd.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_incr_cifar_upgd.err

cd /scratch/gilbreth/shin283/loss-of-plasticity/lop/incremental_cifar
module load cuda

# Display GPU information
nvidia-smi

# Activate conda environment for loss-of-plasticity
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/lop

# WandB Configuration
export WANDB_PROJECT="upgd-incremental-cifar"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

# Create results directory
mkdir -p /scratch/gilbreth/shin283/loss-of-plasticity/lop/incremental_cifar/results

echo "========================================="
echo "Running Incremental CIFAR-100 with UPGD"
echo "Experiment: UPGD Baseline"
echo "Seed: 0"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_JOB_ID}_incr_cifar_upgd_seed_0"

python3 incremental_cifar_experiment.py \
    --config ./cfg/upgd_baseline.json \
    --verbose \
    --experiment-index 0

echo "========================================="
echo "UPGD Incremental CIFAR-100 experiment completed"
echo "End time: $(date)"
echo "========================================="
