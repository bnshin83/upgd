#!/bin/bash
#SBATCH --job-name=upgd_full_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19%8
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_upgd_full_extra.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_upgd_full_extra.err

# UPGD Full extra seeds: 4 datasets x 5 seeds = 20 jobs
# EMNIST has 4, CIFAR-10 has 7, others less — this ensures 5 consistent seeds

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

TASKS=("label_permuted_emnist_stats" "label_permuted_cifar10_stats" "label_permuted_mini_imagenet_stats" "input_permuted_mnist_stats")
TASK_NAMES=("emnist" "cifar10" "mini_imagenet" "imnist")

TASK=${TASKS[$DATASET_IDX]}
TASK_NAME=${TASK_NAMES[$DATASET_IDX]}

LR=0.01
SIGMA=0.001
BETA_UTILITY=0.9
WEIGHT_DECAY=0.0

export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_full_${TASK_NAME}_seed_${SEED}"

echo "========================================="
echo "UPGD Full: ${TASK_NAME} seed=${SEED}"
echo "lr=${LR}, sigma=${SIGMA}, beta=${BETA_UTILITY}, wd=${WEIGHT_DECAY}"
echo "Start: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task ${TASK} \
    --learner upgd_fo_global \
    --seed ${SEED} \
    --lr ${LR} \
    --sigma ${SIGMA} \
    --beta_utility ${BETA_UTILITY} \
    --weight_decay ${WEIGHT_DECAY} \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "UPGD Full ${TASK_NAME} seed=${SEED} completed at $(date)"
