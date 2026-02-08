#!/bin/bash
#SBATCH --job-name=incr_cifar_upgd_gating
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_incr_cifar_gating.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_incr_cifar_gating.err

cd /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar

# Load modules
module load cuda python

# Display GPU information
nvidia-smi

# Activate Python 3.11 venv
source /scratch/gautschi/shin283/loss-of-plasticity/.lop_venv_compute/bin/activate
export PYTHONPATH="/scratch/gautschi/shin283/loss-of-plasticity:$PYTHONPATH"

# WandB Configuration
export WANDB_PROJECT="upgd-incremental-cifar"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

# Create results directory
mkdir -p /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results

# Gating Mode Sweep Configuration
# Array indices map to different gating modes and seeds:
# 0-2: full gating (seeds 0, 1, 2)
# 3-5: output_only gating (seeds 0, 1, 2)
# 6-8: hidden_only gating (seeds 0, 1, 2)

declare -A gating_mode_array=(
    [0]="full" [1]="full" [2]="full"
    [3]="output_only" [4]="output_only" [5]="output_only"
    [6]="hidden_only" [7]="hidden_only" [8]="hidden_only"
)
declare -A seed_array=(
    [0]=0 [1]=1 [2]=2
    [3]=0 [4]=1 [5]=2
    [6]=0 [7]=1 [8]=2
)

GATING_MODE=${gating_mode_array[$SLURM_ARRAY_TASK_ID]}
SEED=${seed_array[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "Running Incremental CIFAR-100 UPGD Gating Mode Comparison"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Gating Mode: $GATING_MODE"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_upgd_${GATING_MODE}_seed${SEED}"

# Create temporary config file with gating mode
CONFIG_FILE="/tmp/upgd_gating_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

cat > $CONFIG_FILE <<EOF
{
  "_model_description_": "UPGD optimizer gating mode comparison",
  "data_path": "",
  "results_dir": "",
  "experiment_name": "upgd_gating_${GATING_MODE}",
  "num_workers": 12,
  "stepsize": 0.1,
  "weight_decay": 0.0005,
  "momentum": 0.9,
  "noise_std": 0.0,
  "use_upgd": true,
  "upgd_beta_utility": 0.999,
  "upgd_sigma": 0.001,
  "upgd_beta1": 0.9,
  "upgd_beta2": 0.999,
  "upgd_eps": 1e-5,
  "upgd_use_adam_moments": true,
  "upgd_gating_mode": "${GATING_MODE}",
  "upgd_non_gated_scale": 0.5,
  "use_cbp": false,
  "reset_head": false,
  "reset_network": false,
  "early_stopping": true
}
EOF

python incremental_cifar_experiment.py \
    --config $CONFIG_FILE \
    --verbose \
    --experiment-index $SEED \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-run-name "${WANDB_RUN_NAME}"

# Clean up temporary config file
rm -f $CONFIG_FILE

echo "========================================="
echo "UPGD gating mode experiment completed"
echo "End time: $(date)"
echo "========================================="
