#!/bin/bash
#SBATCH --job-name=rl_ant_upgd_output
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_rl_ant_upgd_output.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_rl_ant_upgd_output.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Running RL: Ant-v4 - UPGD Output Only"
echo "Seed: 0"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_JOB_ID}_ant_upgd_output_only_seed_2"
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 2 \
    --total_timesteps 20000000 \
    --optimizer upgd_output_only \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "========================================="
echo "UPGD Output Only completed"
echo "End time: $(date)"
echo "========================================="
