#!/bin/bash
#SBATCH --job-name=upgd_parallel
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_parallel.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_parallel.err

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
echo "Running UPGD Experiments in Parallel"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "========================================="

# Create a function to run experiments on specific GPU
run_on_gpu() {
    local gpu_id=$1
    local cmd_file=$2
    local exp_name=$3
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "[GPU $gpu_id] Starting $exp_name"
    
    while IFS= read -r cmd; do
        eval $cmd &
        # Limit parallel jobs per GPU
        while [ $(jobs -r | wc -l) -ge 2 ]; do
            sleep 1
        done
    done < "$cmd_file"
    
    wait
    echo "[GPU $gpu_id] Completed $exp_name"
}

# Run experiments in parallel across GPUs
# Assign different experiment types to different GPUs

# GPU 0-1: Input-permuted MNIST UPGD experiments
run_on_gpu 0 "generated_cmds/input_permuted_mnist/upgd_fo_global.txt" "MNIST UPGD FO Global" &
run_on_gpu 1 "generated_cmds/input_permuted_mnist/upgd_nonprotecting_fo_global.txt" "MNIST UPGD Non-protecting" &

# GPU 2-3: Input-permuted MNIST other methods
run_on_gpu 2 "generated_cmds/input_permuted_mnist/shrink_and_perturb.txt" "MNIST Shrink & Perturb" &
run_on_gpu 3 "generated_cmds/input_permuted_mnist/pgd.txt" "MNIST PGD" &

# GPU 4-5: Label-permuted CIFAR-10 UPGD experiments
run_on_gpu 4 "generated_cmds/label_permuted_cifar10/upgd_fo_global.txt" "CIFAR-10 UPGD FO Global" &
run_on_gpu 5 "generated_cmds/label_permuted_cifar10/upgd_nonprotecting_fo_global.txt" "CIFAR-10 UPGD Non-protecting" &

# GPU 6-7: Label-permuted CIFAR-10 other methods and baselines
run_on_gpu 6 "generated_cmds/label_permuted_cifar10/shrink_and_perturb.txt" "CIFAR-10 Shrink & Perturb" &
run_on_gpu 7 "generated_cmds/label_permuted_cifar10/pgd.txt" "CIFAR-10 PGD" &

# Wait for all background jobs to complete
wait

# Run SGD baselines sequentially (they're smaller)
echo "Running SGD baselines..."
export CUDA_VISIBLE_DEVICES=0
while IFS= read -r cmd; do eval $cmd; done < "generated_cmds/input_permuted_mnist/sgd.txt"
while IFS= read -r cmd; do eval $cmd; done < "generated_cmds/label_permuted_cifar10/sgd.txt"

echo "========================================="
echo "All experiments completed"
echo "End time: $(date)"
echo "========================================="

echo "Plotting results..."
python core/plot/plotter.py
python core/plot/plotter_utility.py

echo "Done!"
