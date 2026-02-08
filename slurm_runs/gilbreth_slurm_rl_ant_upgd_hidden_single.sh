#!/bin/bash
#SBATCH --job-name=rl_ant_upgd_hidden_single
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_rl_ant_upgd_hidden_single.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_rl_ant_upgd_hidden_single.err

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

echo "========================================="
echo "Running SINGLE RL: Ant-v4 - UPGD Hidden Only"
echo "Seeds: 1 and 3"
echo "Using 2 GPUs in parallel"
echo "Start time: $(date)"
echo "========================================="

# Run seed 1 on GPU 0
export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_JOB_ID}_ant_upgd_hidden_seed_1"
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 1 \
    --total_timesteps 20000000 \
    --optimizer upgd_hidden_only \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university" &

PID1=$!

# Run seed 3 on GPU 1
export CUDA_VISIBLE_DEVICES=1
export WANDB_RUN_NAME="${SLURM_JOB_ID}_ant_upgd_hidden_seed_3"
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 3 \
    --total_timesteps 20000000 \
    --optimizer upgd_hidden_only \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university" &

PID2=$!

# Wait for both to complete
wait $PID1 $PID2
