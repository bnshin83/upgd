#!/bin/bash
#SBATCH --job-name=rl_ant_adam
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_rl_ant_adam.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_rl_ant_adam.err

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

echo "========================================="
echo "Running RL: Ant-v4 - Adam Baseline"
echo "Seed: 0 (proof of concept)"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_JOB_ID}_ant_adam_seed_0"
python3 core/run/rl/ppo_continuous_action_adam.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "========================================="
echo "Adam baseline completed"
echo "End time: $(date)"
echo "========================================="
