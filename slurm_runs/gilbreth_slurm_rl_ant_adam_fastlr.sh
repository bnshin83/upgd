#!/bin/bash
#SBATCH --job-name=rl_ant_adam_fastlr
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_rl_ant_adam_fastlr.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_rl_ant_adam_fastlr.err

cd /scratch/gilbreth/shin283/upgd
module load cuda

export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
# Activate the UPGD conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# WandB Configuration
export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Running RL: Ant-v4 - Adam with Fast LR Decay"
echo "Seed: 0"
echo "Total timesteps: 20,000,000"
echo "LR anneal over: 5,000,000 steps (faster decay)"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_JOB_ID}_ant_adam_fastlr_seed_0"
python3 core/run/rl/ppo_continuous_action_adam.py \
    --env_id Ant-v4 \
    --seed 0 \
    --total_timesteps 20000000 \
    --lr_anneal_timesteps 5000000 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "========================================="
echo "Adam Fast LR experiment completed"
echo "End time: $(date)"
echo "========================================="
