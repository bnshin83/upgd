#!/bin/bash
#SBATCH --job-name=rl_ant_upgd_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-30%8
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_rl_ant_upgd.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_rl_ant_upgd.err

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

# Seeds 10-40 (31 seeds total)
SEEDS=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "Running RL: Ant-v4 - UPGD Full Gating"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo "========================================="

WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_ant_upgd_full_seed_${SEED}" \
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed $SEED \
    --total_timesteps 20000000 \
    --optimizer upgd_full \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "========================================="
echo "UPGD Full experiment completed (Seed $SEED)"
echo "End time: $(date)"
echo "========================================="
