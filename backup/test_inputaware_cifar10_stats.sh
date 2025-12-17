#!/bin/bash
#SBATCH --job-name=upgd_input_aware_cifar10_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_input_aware_cifar10_seed_2.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_input_aware_cifar10_seed_2.err

# Change to the submission directory
cd /scratch/gautschi/shin283/upgd

# Load CUDA and Python modules
module load cuda
module load python

# Set PYTHONPATH first (before activating venv to ensure priority)
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_input_aware_cifar10_seed_2"
export WANDB_MODE="online"

# Input-aware config (override by exporting before sbatch)
# Parameter-free mappings (only τ needed):
#   'centered_linear_tau_ratio' - λ = κ/τ (linear, sharp)
#   'sigmoid_tau_ratio' - λ = 2·σ((κ-τ)/τ) (smooth)
# Legacy mappings (require lambda_scale):
#   'centered_linear', 'sigmoid', 'centered_linear_auto_scale', 'centered_linear_tau_norm'
: ${UPGD_LAMBDA_MAPPING:=centered_linear}
: ${UPGD_GATING_STRATEGY:=option_c}
: ${UPGD_OPTION_C_A:=1.0}
: ${UPGD_OPTION_C_B:=1.0}
: ${UPGD_MIN_GATING:=0.05}
export UPGD_LAMBDA_MAPPING UPGD_GATING_STRATEGY UPGD_OPTION_C_A UPGD_OPTION_C_B UPGD_MIN_GATING

# Tag the run name with gating config for traceability
export WANDB_RUN_NAME="${WANDB_RUN_NAME}_map_${UPGD_LAMBDA_MAPPING}_gate_${UPGD_GATING_STRATEGY}_a_${UPGD_OPTION_C_A}_b_${UPGD_OPTION_C_B}_ming_${UPGD_MIN_GATING}"

echo "========================================="
echo "Running Input-Aware UPGD CIFAR-10 Statistics"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Dataset: CIFAR-10 (10 classes, 32x32x3 RGB images)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.001"
echo "Beta Utility: 0.999"
echo "Weight Decay: 0.0"
echo "Curvature threshold: 0.01 (τ)"
echo "Lambda max: 2.0"
if [[ "$UPGD_LAMBDA_MAPPING" == "centered_linear_tau_ratio" ]]; then
  echo "Lambda mapping: ${UPGD_LAMBDA_MAPPING} (parameter-free: λ = κ / τ)"
elif [[ "$UPGD_LAMBDA_MAPPING" == "sigmoid_tau_ratio" ]]; then
  echo "Lambda mapping: ${UPGD_LAMBDA_MAPPING} (parameter-free: λ = 2·σ((κ-τ)/τ))"
else
  echo "Lambda mapping: ${UPGD_LAMBDA_MAPPING}"
fi
echo "Lambda scale: 0.1"
echo "Hutchinson samples: 5"
echo "Compute curvature every: 1 step(s)"
echo "Total samples: 1000000"
echo "Gating strategy: ${UPGD_GATING_STRATEGY}"
echo "Option C params: a=${UPGD_OPTION_C_A}, b=${UPGD_OPTION_C_B}, min_g=${UPGD_MIN_GATING}"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "========================================="

# Run input-aware UPGD
export CUDA_VISIBLE_DEVICES=0

# Enable debug logging for lambda computation (first 100 steps only)
export UPGD_DEBUG_LAMBDA=1

python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_cifar10_stats \
    --learner upgd_input_aware_fo_global \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.001 \
    --beta_utility 0.999 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --curvature_threshold 0.01 \
    --lambda_max 2.0 \
    --lambda_scale 0.1 \
    --hutchinson_samples 5 \
    --compute_curvature_every 1 \
    --disable_regularization False \
    --disable_gating False \
    --save_path logs

echo "========================================="
echo "Input-Aware UPGD experiment completed"
echo "End time: $(date)"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}_upgd_input_aware_cifar10_seed_2.out"
echo "Check the log directory and WandB dashboard for detailed results"
echo "========================================="
