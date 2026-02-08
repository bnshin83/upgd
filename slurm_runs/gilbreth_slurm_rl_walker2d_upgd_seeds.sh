#!/bin/bash
#SBATCH --job-name=rl_walker2d_upgd_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=2-12:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%A_%a_rl_walker2d_upgd.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%A_%a_rl_walker2d_upgd.err

cd /scratch/gilbreth/shin283/upgd
module load cuda

export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# WandB Configuration
export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

# Seeds 0-9 (10 seeds total)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Each array task runs up to 2 seeds
START_IDX=$((SLURM_ARRAY_TASK_ID * 2))
NUM_SEEDS=${#SEEDS[@]}

echo "========================================="
echo "Running RL: Walker2d-v4 - UPGD Full Gating"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using 2 GPUs in parallel"
echo "Start time: $(date)"
echo "========================================="

PIDS=()

for i in 0 1; do
    SEED_IDX=$((START_IDX + i))
    if [ $SEED_IDX -lt $NUM_SEEDS ]; then
        SEED=${SEEDS[$SEED_IDX]}
        echo "Starting seed $SEED on GPU $i"
        
        CUDA_VISIBLE_DEVICES=$i \
        WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_walker2d_upgd_full_seed_${SEED}" \
        python3 core/run/rl/run_ppo_upgd.py \
            --env_id Walker2d-v4 \
            --seed $SEED \
            --total_timesteps 20000000 \
            --optimizer upgd_full \
            --weight_decay 0.0 \
            --cuda \
            --track \
            --wandb_project_name "upgd-rl" \
            --wandb_entity "shin283-purdue-university" &
        
        PIDS+=($!)
    fi
done

echo "Running seeds with PIDs: ${PIDS[@]}"

# Wait for all to complete
wait "${PIDS[@]}"

echo "========================================="
echo "UPGD Full experiments completed"
echo "End time: $(date)"
echo "========================================="
