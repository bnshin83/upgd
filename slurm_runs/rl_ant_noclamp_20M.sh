#!/bin/bash
#SBATCH --job-name=rl_ant_nc
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=14:00:00
#SBATCH --array=0-24
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_rl_ant_noclamp.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_rl_ant_noclamp.err

# Ant-v4 20M steps — NO CLAMP (matches original Elsayed UPGD)
# 5 methods x 5 seeds = 25 jobs
# Array mapping: method_idx = TASK_ID / 5, seed_idx = TASK_ID % 5

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

METHODS=("upgd_full" "upgd_output_only" "upgd_hidden_only" "upgd_full_old" "adam")
SEEDS=(1 2 3 4 5)

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))
METHOD=${METHODS[$METHOD_IDX]}
SEED=${SEEDS[$SEED_IDX]}

export WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_ant_${METHOD}_seed${SEED}_noclamp"

echo "=========================================="
echo "Ant-v4 20M - ${METHOD} seed ${SEED} (NO CLAMP)"
echo "Start: $(date)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/rl/run_ppo_upgd.py \
    --env_id Ant-v4 \
    --seed ${SEED} \
    --total_timesteps 20000000 \
    --optimizer ${METHOD} \
    --weight_decay 0.0 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "Done: Ant ${METHOD} seed ${SEED} at $(date)"
