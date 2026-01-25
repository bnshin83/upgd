#!/bin/bash
#SBATCH --job-name=incr_cifar_sgd
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_incr_cifar_sgd.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_incr_cifar_sgd.err

cd /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar

# Load modules
module load cuda python

# Display GPU information
nvidia-smi

# Activate Python 3.11 venv (avoids conda Python 3.8 library conflicts)
source /scratch/gautschi/shin283/loss-of-plasticity/.lop_venv_compute/bin/activate
export PYTHONPATH="/scratch/gautschi/shin283/loss-of-plasticity:$PYTHONPATH"

# WandB Configuration
export WANDB_PROJECT="upgd-incremental-cifar"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

# Create results directory
mkdir -p /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results

echo "========================================="
echo "Running Incremental CIFAR-100 with SGD"
echo "Experiment: SGD Baseline (for comparison)"
echo "Seed: 0"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_JOB_ID}_incr_cifar_sgd_seed_0"

python incremental_cifar_experiment.py \
    --config ./cfg/base_deep_learning_system.json \
    --verbose \
    --experiment-index 0 \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-run-name "${WANDB_RUN_NAME}"

echo "========================================="
echo "SGD Incremental CIFAR-100 experiment completed"
echo "End time: $(date)"
echo "========================================="
