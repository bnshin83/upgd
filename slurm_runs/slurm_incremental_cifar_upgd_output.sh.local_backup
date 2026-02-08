#!/bin/bash
#SBATCH --job-name=incr_cifar_upgd_output
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_incr_cifar_upgd_output.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_incr_cifar_upgd_output.err

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

GATING_MODE="output_only"
SEED=0

echo "========================================="
echo "Running Incremental CIFAR-100 with UPGD"
echo "Gating Mode: ${GATING_MODE}"
echo "Seed: ${SEED}"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_JOB_ID}_incr_cifar_upgd_${GATING_MODE}_seed${SEED}"

# Create config file
CONFIG_FILE="/tmp/upgd_${GATING_MODE}_${SLURM_JOB_ID}.json"

cat > $CONFIG_FILE <<EOF
{
  "_model_description_": "UPGD optimizer - ${GATING_MODE} gating (only final FC layer)",
  "data_path": "",
  "results_dir": "",
  "experiment_name": "upgd_${GATING_MODE}",
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

# Clean up
rm -f $CONFIG_FILE

echo "========================================="
echo "UPGD ${GATING_MODE} gating experiment completed"
echo "End time: $(date)"
echo "========================================="
