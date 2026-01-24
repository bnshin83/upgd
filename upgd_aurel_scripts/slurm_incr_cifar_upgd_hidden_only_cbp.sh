#!/bin/bash
#SBATCH --job-name=incr_cifar_upgd_hidden_cbp
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_incr_cifar_upgd_hidden_only_cbp.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_incr_cifar_upgd_hidden_only_cbp.err

cd /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar
module load cuda
module load python

# Display GPU information
nvidia-smi

# Activate conda environment for loss-of-plasticity
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# WandB Configuration
export WANDB_PROJECT="upgd-incremental-cifar"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

# Create results directory
mkdir -p /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results

echo "========================================="
echo "Running Incremental CIFAR-100 with UPGD + CBP"
echo "Experiment: UPGD Hidden-Only Gating + Continual Backprop"
echo "Seed: 0"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_JOB_ID}_incr_cifar_upgd_hidden_only_cbp_seed_0"

python3.8 incremental_cifar_experiment.py \
    --config ./cfg/upgd_hidden_only_cbp.json \
    --verbose \
    --experiment-index 0

echo "========================================="
echo "UPGD Hidden-Only + CBP experiment completed"
echo "End time: $(date)"
echo "========================================="
