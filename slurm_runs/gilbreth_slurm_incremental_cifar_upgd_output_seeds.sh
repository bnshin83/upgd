#!/bin/bash
#SBATCH --job-name=incr_cifar_upgd_output_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%A_%a_incr_cifar_upgd_output.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%A_%a_incr_cifar_upgd_output.err

cd /scratch/gilbreth/shin283/loss-of-plasticity/lop/incremental_cifar
module load cuda

eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/lop

# WandB Configuration
export WANDB_PROJECT="upgd-incremental-cifar"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/scratch/gilbreth/shin283/upgd:$PYTHONPATH"

# Create results directory
mkdir -p /scratch/gilbreth/shin283/loss-of-plasticity/lop/incremental_cifar/results

GATING_MODE="output_only"

# Seeds to run (excluding 0 and 2 which are already done)
# Seeds: 1, 3-10 (9 seeds total)
SEEDS=(1 3 4 5 6 7 8 9 10)

# Each array task runs up to 2 seeds
START_IDX=$((SLURM_ARRAY_TASK_ID * 2))
NUM_SEEDS=${#SEEDS[@]}

echo "========================================="
echo "Running Incremental CIFAR-100 with UPGD"
echo "Gating Mode: ${GATING_MODE}"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using 2 GPUs in parallel"
echo "Start time: $(date)"
echo "========================================="

PIDS=()

for i in 0 1; do
    SEED_IDX=$((START_IDX + i))
    if [ $SEED_IDX -lt $NUM_SEEDS ]; then
        SEED=${SEEDS[$SEED_IDX]}
        echo "Starting seed $SEED on GPU $i"
        
        # Create config file for this seed
        CONFIG_FILE="/tmp/upgd_${GATING_MODE}_${SLURM_ARRAY_JOB_ID}_${SEED}.json"
        
        cat > $CONFIG_FILE <<EOF
{
  "_model_description_": "UPGD optimizer - ${GATING_MODE} gating (only final FC layer)",
  "data_path": "",
  "results_dir": "",
  "experiment_name": "upgd_${GATING_MODE}",
  "num_workers": 6,
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
        
        CUDA_VISIBLE_DEVICES=$i \
        WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_incr_cifar_upgd_${GATING_MODE}_seed${SEED}" \
        python3 incremental_cifar_experiment.py \
            --config $CONFIG_FILE \
            --verbose \
            --experiment-index $SEED \
            --wandb \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-entity "${WANDB_ENTITY}" \
            --wandb-run-name "${SLURM_ARRAY_JOB_ID}_incr_cifar_upgd_${GATING_MODE}_seed${SEED}" &
        
        PIDS+=($!)
    fi
done

echo "Running seeds with PIDs: ${PIDS[@]}"

# Wait for all to complete
wait "${PIDS[@]}"

# Clean up config files
rm -f /tmp/upgd_${GATING_MODE}_${SLURM_ARRAY_JOB_ID}_*.json

echo "========================================="
echo "UPGD ${GATING_MODE} gating experiments completed"
echo "End time: $(date)"
echo "========================================="
