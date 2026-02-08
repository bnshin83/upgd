#!/bin/bash
#SBATCH --job-name=adam_baselines
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19%8
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_adam_baselines.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_adam_baselines.err

# Adam baseline: 4 datasets x 5 seeds = 20 jobs
# Array index: dataset_idx * 5 + seed
# Dataset 0=EMNIST, 1=CIFAR-10, 2=Mini-ImageNet, 3=Input-MNIST

cd /scratch/gautschi/shin283/upgd
module load cuda && module load python
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))

# Dataset mapping
TASKS=("label_permuted_emnist_stats" "label_permuted_cifar10_stats" "label_permuted_mini_imagenet_stats" "input_permuted_mnist_stats")
TASK_NAMES=("emnist" "cifar10" "mini_imagenet" "imnist")

TASK=${TASKS[$DATASET_IDX]}
TASK_NAME=${TASK_NAMES[$DATASET_IDX]}

# Hyperparameters from existing best results:
# EMNIST: lr=0.0001, wd=0.1, beta1=0.0, beta2=0.9999
# Input-MNIST: lr=0.001, wd=0.0001
# CIFAR-10/Mini-ImageNet: use EMNIST params (regularized Adam)
LR=0.0001
WEIGHT_DECAY=0.1
BETA1=0.0
BETA2=0.9999
EPS=1e-08

export WANDB_RUN_NAME="${SLURM_JOB_ID}_adam_${TASK_NAME}_seed_${SEED}"

echo "========================================="
echo "Adam baseline: ${TASK_NAME} seed=${SEED}"
echo "lr=${LR}, wd=${WEIGHT_DECAY}, beta1=${BETA1}, beta2=${BETA2}"
echo "Start: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task ${TASK} \
    --learner adam \
    --seed ${SEED} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --eps ${EPS} \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "Adam ${TASK_NAME} seed=${SEED} completed at $(date)"
