#!/bin/bash
#SBATCH --job-name=test_upgd_fo_global_cifar10
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_upgd_fo_global_cifar10.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_upgd_fo_global_cifar10.err

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
echo "Testing UPGD FO Global baseline on Label Permuted CIFAR-10"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "========================================="

# Run UPGD FO Global baseline
export CUDA_VISIBLE_DEVICES=0
echo "Running: python3 core/run/run.py --task label_permuted_cifar10 --learner upgd_fo_global --seed 0 --lr 0.01 --sigma 0.001 --network convolutional_network_relu --n_samples 1000000"
python3 core/run/run.py --task label_permuted_cifar10 --learner upgd_fo_global --seed 0 --lr 0.01 --sigma 0.001 --network convolutional_network_relu --n_samples 1000000

echo "========================================="
echo "Test completed"
echo "End time: $(date)"
echo "========================================="