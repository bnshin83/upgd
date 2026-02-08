#!/bin/bash
#SBATCH --job-name=upgd_outputonly_mini_imagenet_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=84
#SBATCH --gpus-per-node=6
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_outputonly_mini_imagenet_seeds.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_outputonly_mini_imagenet_seeds.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD Output-Only Gating (Mini-ImageNet)"
echo "Seeds: 1, 3-19 (excluding 0 and 2 which are already done)"
echo "Using 6 GPUs to run 18 seeds in 3 batches"
echo "Gating applied ONLY to linear_3 (output layer)"
echo "linear_1, linear_2 use fixed scaling at 0.5 (like SGD)"
echo "Start time: $(date)"
echo "========================================="

# Seeds to run: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
# Batch 1: seeds 1, 3, 4, 5, 6, 7 (6 GPUs)
# Batch 2: seeds 8, 9, 10, 11, 12, 13 (6 GPUs)
# Batch 3: seeds 14, 15, 16, 17, 18, 19 (6 GPUs)

run_seed() {
    local seed=$1
    local gpu=$2
    
    export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputonly_mini_imagenet_seed_${seed}"
    export CUDA_VISIBLE_DEVICES=$gpu
    
    echo "Starting seed $seed on GPU $gpu at $(date)"
    python3 core/run/run_stats_with_curvature.py \
        --task label_permuted_mini_imagenet_stats \
        --learner upgd_fo_global_outputonly \
        --seed $seed \
        --lr 0.01 \
        --sigma 0.001 \
        --beta_utility 0.9 \
        --weight_decay 0.0 \
        --network fully_connected_relu_with_hooks \
        --n_samples 1000000 \
        --compute_curvature_every 1000000 \
        --save_path logs
    
    echo "Completed seed $seed at $(date)"
}

echo ""
echo "========================================="
echo "BATCH 1: Seeds 1, 3, 4, 5, 6, 7"
echo "========================================="
run_seed 1 0 &
run_seed 3 1 &
run_seed 4 2 &
run_seed 5 3 &
run_seed 6 4 &
run_seed 7 5 &
wait

echo ""
echo "========================================="
echo "BATCH 2: Seeds 8, 9, 10, 11, 12, 13"
echo "========================================="
run_seed 8 0 &
run_seed 9 1 &
run_seed 10 2 &
run_seed 11 3 &
run_seed 12 4 &
run_seed 13 5 &
wait

echo ""
echo "========================================="
echo "BATCH 3: Seeds 14, 15, 16, 17, 18, 19"
echo "========================================="
run_seed 14 0 &
run_seed 15 1 &
run_seed 16 2 &
run_seed 17 3 &
run_seed 18 4 &
run_seed 19 5 &
wait

echo ""
echo "========================================="
echo "All seeds completed!"
echo "End time: $(date)"
echo "========================================="
