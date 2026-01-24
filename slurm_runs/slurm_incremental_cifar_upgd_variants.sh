#!/bin/bash
#SBATCH --job-name=incr_cifar_upgd_sweep
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%A_%a_incr_cifar_upgd_sweep.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%A_%a_incr_cifar_upgd_sweep.err

cd /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar
module load cuda
module load python

# Display GPU information
nvidia-smi

# Activate conda environment for loss-of-plasticity
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# WandB Configuration
export WANDB_PROJECT="upgd-incremental-cifar"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

# Create results directory
mkdir -p /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results

# Hyperparameter sweep configuration
# Array indices map to different configurations:
# 0-2: Different beta_utility values (0.99, 0.999, 0.9999) with sigma=0.001
# 3-5: Different sigma values (0.0001, 0.001, 0.01) with beta_utility=0.999
# 6-8: Different seeds (0, 1, 2) with default hyperparameters

declare -A beta_utility_array=([0]=0.99 [1]=0.999 [2]=0.9999 [3]=0.999 [4]=0.999 [5]=0.999 [6]=0.999 [7]=0.999 [8]=0.999)
declare -A sigma_array=([0]=0.001 [1]=0.001 [2]=0.001 [3]=0.0001 [4]=0.001 [5]=0.01 [6]=0.001 [7]=0.001 [8]=0.001)
declare -A seed_array=([0]=0 [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 [6]=0 [7]=1 [8]=2)

BETA_UTILITY=${beta_utility_array[$SLURM_ARRAY_TASK_ID]}
SIGMA=${sigma_array[$SLURM_ARRAY_TASK_ID]}
SEED=${seed_array[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "Running Incremental CIFAR-100 UPGD Hyperparameter Sweep"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Beta Utility: $BETA_UTILITY"
echo "Sigma: $SIGMA"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_upgd_beta${BETA_UTILITY}_sigma${SIGMA}_seed${SEED}"

# Create temporary config file with modified hyperparameters
CONFIG_FILE="/tmp/upgd_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

cat > $CONFIG_FILE <<EOF
{
  "_model_description_": "UPGD optimizer hyperparameter sweep",
  "data_path": "",
  "results_dir": "",
  "experiment_name": "upgd_sweep_beta${BETA_UTILITY}_sigma${SIGMA}",
  "num_workers": 12,
  "stepsize": 0.1,
  "weight_decay": 0.0005,
  "momentum": 0.9,
  "noise_std": 0.0,
  "use_upgd": true,
  "upgd_beta_utility": ${BETA_UTILITY},
  "upgd_sigma": ${SIGMA},
  "upgd_beta1": 0.9,
  "upgd_beta2": 0.999,
  "upgd_eps": 1e-5,
  "upgd_use_adam_moments": true,
  "use_cbp": false,
  "reset_head": false,
  "reset_network": false,
  "early_stopping": true
}
EOF

python3.8 incremental_cifar_experiment.py \
    --config $CONFIG_FILE \
    --verbose \
    --experiment-index $SEED

# Clean up temporary config file
rm -f $CONFIG_FILE

echo "========================================="
echo "UPGD sweep experiment completed"
echo "End time: $(date)"
echo "========================================="
