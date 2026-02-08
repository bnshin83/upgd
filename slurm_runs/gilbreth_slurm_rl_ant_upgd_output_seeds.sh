#!/bin/bash
#SBATCH --job-name=rl_ant_upgd_output_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%A_%a_rl_ant_upgd_output.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%A_%a_rl_ant_upgd_output.err

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
SEEDS=(1 3 4 5 6 7 8 9)

# Each array task runs 2 seeds
SEED1=${SEEDS[$((SLURM_ARRAY_TASK_ID * 2))]}
SEED2=${SEEDS[$((SLURM_ARRAY_TASK_ID * 2 + 1))]}

echo "========================================="
echo "Running RL: Ant-v4 - UPGD Output Only"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seeds: $SEED1 and $SEED2"
echo "Using 2 GPUs in parallel"
echo "Start time: $(date)"
echo "========================================="

# Run seed1 on GPU 0
export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_ant_upgd_output_seed_${SEED1}"
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed $SEED1 \
    --total_timesteps 20000000 \
    --optimizer upgd_output_only \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university" &

PID1=$!

# Run seed2 on GPU 1
export CUDA_VISIBLE_DEVICES=1
export WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_ant_upgd_output_seed_${SEED2}"
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed $SEED2 \
    --total_timesteps 20000000 \
    --optimizer upgd_output_only \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university" &

PID2=$!

# Wait for both to complete
wait $PID1 $PID2

echo "========================================="
echo "UPGD Output Only experiments completed"
echo "Seeds: $SEED1 and $SEED2"
echo "End time: $(date)"
echo "========================================="
