#!/bin/bash
#SBATCH --job-name=qt_ant_ab
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_qt_ant_ab.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_qt_ant_ab.err

# A/B test: Ant-v4 with OLD AdaptiveUPGD (no clamp) vs fixed version
# 2M timesteps, seed 42, single job — upgd_full_old only
# Compare with qt_ant_fixed job's upgd_full (array idx 0)

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

export WANDB_RUN_NAME="${SLURM_JOB_ID}_qt_ant_upgd_full_OLD_seed42"

echo "=========================================="
echo "A/B test: Ant-v4 - upgd_full_old (AdaptiveUPGD, NO clamp)"
echo "Seed: 42, Timesteps: 2M"
echo "Compare with: qt_ant_fixed upgd_full (RLLayerSelectiveUPGD, WITH clamp)"
echo "Start: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed 42 \
    --total_timesteps 2000000 \
    --optimizer upgd_full_old \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "A/B test Ant upgd_full_old done at $(date)"
