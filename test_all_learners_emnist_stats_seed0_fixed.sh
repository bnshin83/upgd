#!/bin/bash
#SBATCH --job-name=test_all_learners_emnist_stats_seed0_fixed
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=7
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_all_learners_emnist_stats_seed0_fixed.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_all_learners_emnist_stats_seed0_fixed.err

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
echo "Running All Learners EMNIST Statistics (Seed 0) - FIXED"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "========================================="

# Common parameters
TASK="label_permuted_emnist_stats"
SEED=0
LR=0.01
NETWORK="fully_connected_relu_with_hooks"
N_SAMPLES=1000000

# Function to run a learner on a specific GPU with correct parameters
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
    
    # Construct base command
    CMD="python3 core/run/run_stats.py --task $TASK --learner $learner --seed $SEED --lr $LR --network $NETWORK --n_samples $N_SAMPLES"
    
    # Add learner-specific parameters based on the generated command files
    case $learner in
        "sgd")
            CMD="$CMD --weight_decay 0.0001"
            ;;
        "adam")
            CMD="$CMD --weight_decay 0.0001"
            ;;
        "pgd")
            CMD="$CMD --sigma 0.005"
            ;;
        "upgd_fo_global")
            CMD="$CMD --beta_utility 0.9 --sigma 0.001 --weight_decay 0.0"
            ;;
        "ewc")
            CMD="$CMD --weight_decay 0.0001 --ewc_lambda 1000.0"
            ;;
        "mas")
            CMD="$CMD --weight_decay 0.0001 --mas_lambda 1.0"
            ;;
        "si")
            CMD="$CMD --weight_decay 0.0001 --si_lambda 1.0 --si_xi 1.0"
            ;;
        *)
            echo "Unknown learner: $learner"
            return 1
            ;;
    esac
    
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

# Define learners and their GPU assignments
declare -A LEARNER_GPU_MAP=(
    ["sgd"]=0
    ["adam"]=1
    ["pgd"]=2
    ["upgd_fo_global"]=3
    ["ewc"]=4
    ["mas"]=5
    ["si"]=6
)

# Start all learners in parallel, one per GPU
pids=()
learner_list=()

for learner in "${!LEARNER_GPU_MAP[@]}"; do
    gpu_id=${LEARNER_GPU_MAP[$learner]}
    
    echo "Launching $learner on GPU $gpu_id..."
    run_learner "$learner" "$gpu_id" &
    pids+=($!)
    learner_list+=("$learner")
    
    # Small delay to avoid race conditions
    sleep 2
done

echo "========================================="
echo "All learners launched in parallel"
echo "PIDs: ${pids[@]}"
echo "Learners: ${learner_list[@]}"
echo "Waiting for completion..."
echo "========================================="

# Wait for all background processes to complete
exit_codes=()
for i in "${!pids[@]}"; do
    pid=${pids[i]}
    learner="${learner_list[i]}"
    
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

# Final summary
echo "Summary of results:"
for i in "${!learner_list[@]}"; do
    learner="${learner_list[i]}"
    exit_code="${exit_codes[i]}"
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ $learner: SUCCESS"
    else
        echo "  ✗ $learner: FAILED (exit code $exit_code)"
    fi
done

# Check if any experiments created output files
echo ""
echo "Checking for output files in logs/:"
if ls logs/label_permuted_emnist_stats/**/0.json >/dev/null 2>&1; then
    echo "✓ JSON output files found:"
    ls logs/label_permuted_emnist_stats/**/0.json
else
    echo "✗ No JSON output files found"
fi