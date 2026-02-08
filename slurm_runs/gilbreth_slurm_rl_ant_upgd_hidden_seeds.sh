#!/bin/bash
#SBATCH --job-name=rl_ant_upgd_hidden_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-19%3
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%A_%a_rl_ant_upgd_hidden.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%A_%a_rl_ant_upgd_hidden.err

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

# Seeds to run (excluding 0 and 2 which are already done)
# Seeds: 1, 3-40 (39 seeds total)
SEEDS=(1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40)

# Each array task runs up to 2 seeds
START_IDX=$((SLURM_ARRAY_TASK_ID * 2))
NUM_SEEDS=${#SEEDS[@]}

echo "========================================="
echo "Running RL: Ant-v4 - UPGD Hidden Only"
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
        WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_ant_upgd_hidden_seed_${SEED}" \
        python3 core/run/rl/run_ppo_upgd.py \
            --env_id Ant-v4 \
            --seed $SEED \
            --total_timesteps 20000000 \
            --optimizer upgd_hidden_only \
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
echo "UPGD Hidden Only experiments completed"
echo "End time: $(date)"
echo "========================================="
