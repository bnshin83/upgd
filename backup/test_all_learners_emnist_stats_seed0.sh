#!/bin/bash
#SBATCH --job-name=test_all_learners_emnist_stats_seed0
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=7
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_all_learners_emnist_stats_seed0.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_all_learners_emnist_stats_seed0.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

echo "========================================="
echo "Running All Learners EMNIST Statistics (Seed 0)"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "========================================="

# Common parameters
TASK="label_permuted_emnist_stats"
SEED=0
LR=0.01
SIGMA=0.001
NETWORK="fully_connected_relu_with_hooks"
N_SAMPLES=1000000

# Array of learners to test
LEARNERS=(
    "sgd"
    "adam"
    "pgd"
    "upgd_fo_global"
    "ewc"
    "mas"
    "si"
)

# Function to run a learner on a specific GPU
run_learner() {
    local learner=$1
    local gpu_id=$2
    local log_prefix="gpu${gpu_id}_${learner}"
    
    echo "========================================="
    echo "Starting learner: $learner on GPU $gpu_id"
    echo "Time: $(date)"
    echo "========================================="
    
    # Set GPU for this process
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Construct the command
    CMD="python3 core/run/run_stats.py --task $TASK --learner $learner --seed $SEED --lr $LR --sigma $SIGMA --network $NETWORK --n_samples $N_SAMPLES"
    
    # Add weight_decay for non-continual learning methods
    if [[ "$learner" == "sgd" || "$learner" == "adam" || "$learner" == "pgd" || "$learner" == "upgd_fo_global" ]]; then
        CMD="$CMD --weight_decay 0.0"
    fi
    
    # Add beta_utility for UPGD
    if [[ "$learner" == "upgd_fo_global" ]]; then
        CMD="$CMD --beta_utility 0.9"
    fi
    
    echo "GPU $gpu_id - Running: $CMD"
    
    # Run the command and capture output to separate log files
    eval $CMD > "logs/${log_prefix}.out" 2> "logs/${log_prefix}.err"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ GPU $gpu_id - $learner completed successfully at $(date)"
    else
        echo "✗ GPU $gpu_id - $learner failed with exit code $exit_code at $(date)"
    fi
    
    return $exit_code
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Start all learners in parallel, one per GPU
pids=()
for i in "${!LEARNERS[@]}"; do
    learner="${LEARNERS[i]}"
    gpu_id=$i
    
    echo "Launching $learner on GPU $gpu_id..."
    run_learner "$learner" "$gpu_id" &
    pids+=($!)
    
    # Small delay to avoid race conditions
    sleep 2
done

echo "========================================="
echo "All learners launched in parallel"
echo "PIDs: ${pids[@]}"
echo "Waiting for completion..."
echo "========================================="

# Wait for all background processes to complete
exit_codes=()
for i in "${!pids[@]}"; do
    pid=${pids[i]}
    learner="${LEARNERS[i]}"
    
    echo "Waiting for $learner (PID: $pid)..."
    wait $pid
    exit_code=$?
    exit_codes+=($exit_code)
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $learner finished successfully"
    else
        echo "✗ $learner failed with exit code $exit_code"
    fi
done

echo "========================================="
echo "All learner tests completed"
echo "End time: $(date)"
echo "========================================="

# Summary
echo "Summary of learners tested:"
for learner in "${LEARNERS[@]}"; do
    echo "  - $learner"
done