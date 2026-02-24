#!/bin/bash
#SBATCH --job-name=rl_walker2d_all
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-79%12
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_rl_walker2d.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_rl_walker2d.err

# Walker2d-v4 PPO: 4 methods × 20 seeds = 80 tasks
#   0-19:  adam
#  20-39:  upgd_full
#  40-59:  upgd_hidden_only
#  60-79:  upgd_output_only

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

# Map array task ID to method + seed
TASK_ID=$SLURM_ARRAY_TASK_ID
METHOD_IDX=$((TASK_ID / 20))
SEED=$((TASK_ID % 20))

case $METHOD_IDX in
    0) METHOD="adam" ;;
    1) METHOD="upgd_full" ;;
    2) METHOD="upgd_hidden_only" ;;
    3) METHOD="upgd_output_only" ;;
esac

echo "========================================="
echo "Walker2d-v4 PPO — ${METHOD} — seed ${SEED}"
echo "Array Task ID: $TASK_ID (method=$METHOD_IDX, seed=$SEED)"
echo "Start time: $(date)"
echo "========================================="

python3 core/run/rl/run_ppo_upgd.py \
    --env_id Walker2d-v4 \
    --seed $SEED \
    --total_timesteps 20000000 \
    --optimizer $METHOD \
    --weight_decay 0.0 \
    --beta_utility 0.999 \
    --sigma 0.001 \
    --non_gated_scale 0.5 \
    --learning_rate 3e-4 \
    --cuda \
    --track \
    --wandb_project_name "upgd-rl" \
    --wandb_entity "shin283-purdue-university"

echo "========================================="
echo "${METHOD} seed=${SEED} completed"
echo "End time: $(date)"
echo "========================================="
