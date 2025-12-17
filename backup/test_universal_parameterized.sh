#!/bin/bash
#SBATCH --job-name=upgd-universal
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_slurm-%j.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_slurm-%j.err

# Configuration parameters - Edit these to change the experiment
TASK="input_permuted_mnist_stats"
LEARNER="adam"  # Can be: adam, ewc, mas, pgd, sgd, synaptic_intelligence, upgd_fo_global, upgd_input_aware_fo_global, etc.
NETWORK="fully_connected_relu_with_hooks"
SEED=0
LR=0.01
SIGMA=0.001
N_SAMPLES=500000
# These parameters are only used for input-aware learners, ignored by others
CURVATURE_THRESHOLD=0.05
LAMBDA_MAX=1.0
HUTCHINSON_SAMPLES=5
COMPUTE_CURVATURE_EVERY=1

# Remap deprecated learner names to current registry keys
case "$LEARNER" in
    input_aware_upgd_fo_global)
        LEARNER="upgd_input_aware_fo_global"
        ;;
    input_aware_upgd_so_global)
        LEARNER="upgd_input_aware_so_global"
        ;;
esac

# Generate experiment identifier from parameters
if [[ "$LEARNER" == *"input_aware"* ]]; then
    EXPERIMENT_ID="${TASK}_${LEARNER}_curv_${CURVATURE_THRESHOLD}_samples_${N_SAMPLES}_seed_${SEED}"
else
    EXPERIMENT_ID="${TASK}_${LEARNER}_samples_${N_SAMPLES}_seed_${SEED}"
fi

# SBATCH directives moved above to ensure Slurm applies them
# Change to the submission directory
cd /scratch/gautschi/shin283/upgd

# Load CUDA and Python modules
module load cuda
module load python

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# Create log directory
mkdir -p /scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}
# Live tee of stdout/stderr into experiment log directory
STDOUT_PATH="/scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}/${SLURM_JOB_ID:-local}.out"
STDERR_PATH="/scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}/${SLURM_JOB_ID:-local}.err"
exec > >(tee -a "$STDOUT_PATH") 2> >(tee -a "$STDERR_PATH" >&2)

# Install wandb if not already installed
pip install wandb

# Initialize wandb configuration
export WANDB_PROJECT="continual-learning-algorithms"
export WANDB_RUN_NAME="${EXPERIMENT_ID}_job_${SLURM_JOB_ID}"
export WANDB_MODE="online"

echo "========================================="
echo "Running Continual Learning Experiment"
echo "Experiment ID: ${EXPERIMENT_ID}"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo ""
echo "Configuration:"
echo "  Task: ${TASK}"
echo "  Learner: ${LEARNER}"
echo "  Network: ${NETWORK}"
echo "  Seed: ${SEED}"
echo "  Learning Rate: ${LR}"
echo "  Sigma: ${SIGMA}"
echo "  N Samples: ${N_SAMPLES}"
if [[ "$LEARNER" == *"input_aware"* ]]; then
    echo "  Curvature Threshold: ${CURVATURE_THRESHOLD}"
    echo "  Lambda Max: ${LAMBDA_MAX}"
    echo "  Hutchinson Samples: ${HUTCHINSON_SAMPLES}"
    echo "  Compute Curvature Every: ${COMPUTE_CURVATURE_EVERY} step(s)"
fi
echo ""
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "SLURM Log Directory: /scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}/"
echo "========================================="

# Run with enhanced stats runner (works for all algorithms)
export CUDA_VISIBLE_DEVICES=0
echo "Starting experiment with command:"
CMD="python3 core/run/run_stats_with_curvature.py --task ${TASK} --learner ${LEARNER} --seed ${SEED} --lr ${LR} --sigma ${SIGMA} --network ${NETWORK} --n_samples ${N_SAMPLES}"
if [[ "$LEARNER" == *"input_aware"* ]]; then
    CMD="$CMD --curvature_threshold ${CURVATURE_THRESHOLD} --lambda_max ${LAMBDA_MAX} --hutchinson_samples ${HUTCHINSON_SAMPLES} --compute_curvature_every ${COMPUTE_CURVATURE_EVERY}"
fi
echo "$CMD"
echo ""

$CMD

# Move SLURM output files to logs directory after completion
if [ ! -z "$SLURM_JOB_ID" ]; then
    SLURM_OUT_FILE="/scratch/gautschi/shin283/upgd/slurm-${SLURM_JOB_ID}.out"
    SLURM_ERR_FILE="/scratch/gautschi/shin283/upgd/slurm-${SLURM_JOB_ID}.err"
    
    if [ -f "$SLURM_OUT_FILE" ]; then
        mv "$SLURM_OUT_FILE" "/scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}/${SLURM_JOB_ID}.out"
        echo "Moved SLURM output to: /scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}/${SLURM_JOB_ID}.out"
    fi
    
    if [ -f "$SLURM_ERR_FILE" ]; then
        mv "$SLURM_ERR_FILE" "/scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}/${SLURM_JOB_ID}.err"  
        echo "Moved SLURM error to: /scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}/${SLURM_JOB_ID}.err"
    fi
fi

echo "========================================="
echo "Experiment ${EXPERIMENT_ID} completed"
echo "End time: $(date)"
echo "JSON results will be saved under: /scratch/gautschi/shin283/upgd/logs/${TASK}/${LEARNER}/${NETWORK}/..."
echo "WandB Run: $WANDB_RUN_NAME"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${EXPERIMENT_ID}/${SLURM_JOB_ID}.out"
echo "Check the log directory and WandB dashboard for detailed results"
echo "========================================="