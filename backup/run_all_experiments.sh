#!/bin/bash
#SBATCH --job-name=upgd_run_all
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_run_all.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_run_all.err

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
echo "Running ALL UPGD Experiments"
echo "Start time: $(date)"
echo "========================================="

# Function to run experiments from a command file
run_experiment_file() {
    local cmd_file=$1
    local exp_name=$2
    
    if [ -f "$cmd_file" ]; then
        echo ""
        echo "Running $exp_name from $cmd_file"
        echo "Number of commands: $(wc -l < $cmd_file)"
        echo "---"
        
        # Run each command in the file
        while IFS= read -r cmd; do
            echo "Executing: $cmd"
            eval $cmd
            if [ $? -eq 0 ]; then
                echo "✓ Command completed"
            else
                echo "✗ Command failed"
            fi
        done < "$cmd_file"
        
        echo "✓ Finished $exp_name"
    else
        echo "✗ File not found: $cmd_file"
    fi
}

# Run all generated experiments

echo "=== INPUT-PERMUTED MNIST EXPERIMENTS ==="
run_experiment_file "generated_cmds/input_permuted_mnist/sgd.txt" "SGD baseline"
run_experiment_file "generated_cmds/input_permuted_mnist/pgd.txt" "PGD"
run_experiment_file "generated_cmds/input_permuted_mnist/shrink_and_perturb.txt" "Shrink & Perturb"
run_experiment_file "generated_cmds/input_permuted_mnist/upgd_fo_global.txt" "UPGD First-Order Global"
run_experiment_file "generated_cmds/input_permuted_mnist/upgd_nonprotecting_fo_global.txt" "UPGD Non-protecting First-Order Global"

echo ""
echo "=== LABEL-PERMUTED CIFAR-10 EXPERIMENTS ==="
run_experiment_file "generated_cmds/label_permuted_cifar10/sgd.txt" "SGD baseline"
run_experiment_file "generated_cmds/label_permuted_cifar10/pgd.txt" "PGD"
run_experiment_file "generated_cmds/label_permuted_cifar10/shrink_and_perturb.txt" "Shrink & Perturb"
run_experiment_file "generated_cmds/label_permuted_cifar10/upgd_fo_global.txt" "UPGD First-Order Global"
run_experiment_file "generated_cmds/label_permuted_cifar10/upgd_nonprotecting_fo_global.txt" "UPGD Non-protecting First-Order Global"

echo ""
echo "========================================="
echo "All experiments completed"
echo "End time: $(date)"
echo "========================================="

echo ""
echo "Next steps:"
echo "1. Check results in logs/ directory"
echo "2. Plot results with:"
echo "   python core/plot/plotter.py"
echo "   python core/plot/plotter_utility.py"
