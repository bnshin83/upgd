#!/bin/bash
#SBATCH --job-name=incr_cifar_upgd_all
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_incr_cifar_upgd_all.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_incr_cifar_upgd_all.err

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

# Array of config files and variant names
declare -a CONFIGS=("sgd_baseline.json" "upgd_baseline.json" "upgd_output_only.json" "upgd_hidden_only.json" "upgd_output_only_cbp.json" "upgd_hidden_only_cbp.json")
declare -a VARIANTS=("sgd_baseline" "upgd_full" "upgd_output_only" "upgd_hidden_only" "upgd_output_only_cbp" "upgd_hidden_only_cbp")
declare -a DESCRIPTIONS=("SGD Baseline (no UPGD)" "UPGD Full Gating (all layers)" "UPGD Output-Only Gating (fc layer only)" "UPGD Hidden-Only Gating (all except fc)" "UPGD Output-Only + Continual Backprop" "UPGD Hidden-Only + Continual Backprop")

# Select config based on array task ID
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
DESCRIPTION=${DESCRIPTIONS[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "Running Incremental CIFAR-100"
echo "Experiment: $DESCRIPTION"
echo "Config: $CONFIG"
echo "Variant: $VARIANT"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Seed: 0"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_incr_cifar_${VARIANT}_seed_0"

python3.8 incremental_cifar_experiment.py \
    --config ./cfg/$CONFIG \
    --verbose \
    --experiment-index 0

echo "========================================="
echo "$DESCRIPTION experiment completed"
echo "End time: $(date)"
echo "========================================="
