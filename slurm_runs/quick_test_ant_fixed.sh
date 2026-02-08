#!/bin/bash
#SBATCH --job-name=qt_ant_fixed
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_qt_ant_fixed.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_qt_ant_fixed.err

# Quick test: Ant-v4 with fixed optimizer (RLLayerSelectiveUPGD for all modes)
# 2M timesteps, seed 42, 4 methods

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

METHODS=("upgd_full" "upgd_output_only" "upgd_hidden_only" "adam")
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

export WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_qt_ant_${METHOD}_seed42_fixed"

echo "=========================================="
echo "Quick test: Ant-v4 - ${METHOD} (FIXED optimizer)"
echo "Seed: 42, Timesteps: 2M"
echo "Start: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 42 \
    --total_timesteps 2000000 \
    --optimizer ${METHOD} \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "Quick test Ant ${METHOD} done at $(date)"
