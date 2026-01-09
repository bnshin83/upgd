#!/bin/bash
#SBATCH --job-name=upgd_layer_umax
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_layer_umax.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_layer_umax.err

# Re-run a training job so the JSON logs include:
#   - layer_utility_max_per_step (raw + scaled per layer)
#
# Usage (minimal):
#   sbatch rerun_with_layer_umax.sh label_permuted_emnist_stats upgd_fo_global_outputonly 2
#
# Usage (override defaults via env vars):
#   N_SAMPLES=300000 CURVATURE_EVERY=1000000 LR=0.01 SIGMA=0.001 BETA_UTILITY=0.9 WD=0.0 \
#   sbatch rerun_with_layer_umax.sh label_permuted_emnist_stats upgd_fo_global_outputonly 2

set -euo pipefail

TASK="${1:-label_permuted_emnist_stats}"
LEARNER="${2:-upgd_fo_global_outputonly}"
SEED="${3:-2}"

LR="${LR:-0.01}"
SIGMA="${SIGMA:-0.001}"
BETA_UTILITY="${BETA_UTILITY:-0.9}"
WD="${WD:-0.0}"
NETWORK="${NETWORK:-fully_connected_relu_with_hooks}"
N_SAMPLES="${N_SAMPLES:-1000000}"
CURVATURE_EVERY="${CURVATURE_EVERY:-1000000}"

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration (optional; set WANDB_MODE=disabled to turn off)
export WANDB_PROJECT="${WANDB_PROJECT:-upgd}"
export WANDB_ENTITY="${WANDB_ENTITY:-shin283-purdue-university}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${SLURM_JOB_ID}_${LEARNER}_${TASK}_seed${SEED}}"

echo "========================================="
echo "UPGD rerun for per-layer utility max logging"
echo "Task: ${TASK}"
echo "Learner: ${LEARNER}"
echo "Seed: ${SEED}"
echo "lr=${LR}, sigma=${SIGMA}, beta_utility=${BETA_UTILITY}, wd=${WD}"
echo "n_samples=${N_SAMPLES}, compute_curvature_every=${CURVATURE_EVERY}"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
  --task "${TASK}" \
  --learner "${LEARNER}" \
  --seed "${SEED}" \
  --lr "${LR}" \
  --sigma "${SIGMA}" \
  --beta_utility "${BETA_UTILITY}" \
  --weight_decay "${WD}" \
  --network "${NETWORK}" \
  --n_samples "${N_SAMPLES}" \
  --compute_curvature_every "${CURVATURE_EVERY}" \
  --save_path logs

echo "Completed: $(date)"


