#!/bin/bash
#SBATCH --job-name=ewc_baselines
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19%6
#SBATCH --nodelist=gilbreth-k[001-031,033-050]
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%A_%a_ewc_baselines.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%A_%a_ewc_baselines.err

# EWC baseline: 4 datasets x 5 seeds = 20 jobs

cd /scratch/gilbreth/shin283/upgd
module load cuda
export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))

TASKS=("label_permuted_emnist_stats" "label_permuted_cifar10_stats" "label_permuted_mini_imagenet_stats" "input_permuted_mnist_stats")
TASK_NAMES=("emnist" "cifar10" "mini_imagenet" "imnist")

TASK=${TASKS[$DATASET_IDX]}
TASK_NAME=${TASK_NAMES[$DATASET_IDX]}

LR=0.01
BETA_WEIGHT=0.9999
BETA_FISHER=0.9999
LAMDA=1.0

export WANDB_RUN_NAME="${SLURM_JOB_ID}_ewc_${TASK_NAME}_seed_${SEED}"

echo "========================================="
echo "EWC baseline: ${TASK_NAME} seed=${SEED}"
echo "lr=${LR}, lamda=${LAMDA}, beta_weight=${BETA_WEIGHT}, beta_fisher=${BETA_FISHER}"
echo "Start: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task ${TASK} \
    --learner ewc \
    --seed ${SEED} \
    --lr ${LR} \
    --beta_weight ${BETA_WEIGHT} \
    --beta_fisher ${BETA_FISHER} \
    --lamda ${LAMDA} \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "EWC ${TASK_NAME} seed=${SEED} completed at $(date)"
