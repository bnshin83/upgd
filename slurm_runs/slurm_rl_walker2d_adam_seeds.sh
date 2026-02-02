#!/bin/bash
#SBATCH --job-name=rl_walker2d_adam_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_rl_walker2d_adam.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_rl_walker2d_adam.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

# Seeds 0-9 (10 seeds total)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "Running RL: Walker2d-v4 - Adam"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo "========================================="

WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_walker2d_adam_seed_${SEED}" \
python3 core/run/rl/ppo_continuous_action_adam.py \
    --env_id Walker2d-v4 \
    --seed $SEED \
    --total_timesteps 20000000 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "========================================="
echo "Adam experiment completed (Seed $SEED)"
echo "End time: $(date)"
echo "========================================="
