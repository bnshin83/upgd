#!/bin/bash
#SBATCH --job-name=upgd_statistics
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_statistics.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_statistics.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Display GPU information
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p /scratch/gautschi/shin283/upgd/logs

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Verify environment setup
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set PYTHONPATH to include the current directory
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

echo "========================================="
echo "Starting Diagnostic Statistics Experiments (Figure 5)"
echo "Time: $(date)"
echo "========================================="

# Choose which statistics experiment to run
# Uncomment the one you want to execute:

# Option 1: Input-permuted MNIST statistics
# python experiments/statistics_input_permuted_mnist.py

# Option 2: Output-permuted CIFAR-10 statistics
# python experiments/statistics_output_permuted_cifar10.py

# Option 3: Output-permuted EMNIST statistics
# python experiments/statistics_output_permuted_emnist.py

# Option 4: Output-permuted ImageNet statistics
# python experiments/statistics_output_permuted_imagenet.py

echo "NOTE: Uncomment one of the statistics experiments in this script before running"
echo ""
echo "After experiments complete, results will be saved in logs/ in JSON format"

echo ""
echo "Job completed at: $(date)"
