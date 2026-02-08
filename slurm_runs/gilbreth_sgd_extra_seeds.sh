#!/bin/bash
#SBATCH --job-name=sgd_extra_seeds
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
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%A_%a_sgd_extra.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%A_%a_sgd_extra.err

# SGD extra seeds: 4 datasets x 5 seeds = 20 jobs

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

case $DATASET_IDX in
    0)  # EMNIST
        LR=0.01; WEIGHT_DECAY=0.0001 ;;
    1)  # CIFAR-10
        LR=0.01; WEIGHT_DECAY=0.0 ;;
    2)  # Mini-ImageNet
        LR=0.005; WEIGHT_DECAY=0.0 ;;
    3)  # Input-MNIST
        LR=0.01; WEIGHT_DECAY=0.0001 ;;
esac

export WANDB_RUN_NAME="${SLURM_JOB_ID}_sgd_${TASK_NAME}_seed_${SEED}"

echo "========================================="
echo "SGD baseline: ${TASK_NAME} seed=${SEED}"
echo "lr=${LR}, weight_decay=${WEIGHT_DECAY}"
echo "Start: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task ${TASK} \
    --learner sgd \
    --seed ${SEED} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "SGD ${TASK_NAME} seed=${SEED} completed at $(date)"
