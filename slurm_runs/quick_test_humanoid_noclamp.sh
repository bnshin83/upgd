#!/bin/bash
#SBATCH --job-name=qt_hum_nc
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --nodelist=gilbreth-k[001-031,033-050]
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%A_%a_qt_hum_noclamp.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%A_%a_qt_hum_noclamp.err

# Quick test: Humanoid-v4 — NO CLAMP on global_max_util (matches original Elsayed UPGD)
# 4 runs: full, output_only, hidden_only, adam
# 2M timesteps, seed 42

cd /scratch/gilbreth/shin283/upgd
module load cuda
export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

METHODS=("upgd_full" "upgd_output_only" "upgd_hidden_only" "adam")
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

export WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_qt_humanoid_${METHOD}_seed42_noclamp"

echo "=========================================="
echo "Quick test: Humanoid-v4 - ${METHOD} (NO CLAMP — original UPGD)"
echo "Seed: 42, Timesteps: 2M"
echo "Start: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Humanoid-v4 \
    --seed 42 \
    --total_timesteps 2000000 \
    --optimizer ${METHOD} \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "Quick test Humanoid ${METHOD} done at $(date)"
