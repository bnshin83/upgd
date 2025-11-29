#!/bin/bash
#SBATCH --job-name=test_input_aware_emnist_max
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_input_aware_emnist_max.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_input_aware_emnist_max.err

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
echo "Testing Input-Aware UPGD on Label Permuted EMNIST with Max Configuration"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "========================================="

# Run input-aware UPGD with maximum protection configuration on label permuted EMNIST
export CUDA_VISIBLE_DEVICES=0
echo "Running: python3 core/run/input_aware_run.py --task label_permuted_emnist --learner input_aware_upgd_fo_global --seed 0 --lr 0.01 --sigma 0.001 --network fully_connected_relu --n_samples 1000000 --curvature_threshold 10.0 --lambda_max 1.0 --hutchinson_samples 3 --compute_curvature_every 1"
python3 core/run/input_aware_run.py --task label_permuted_emnist --learner input_aware_upgd_fo_global --seed 0 --lr 0.01 --sigma 0.001 --network fully_connected_relu --n_samples 1000000 --curvature_threshold 10.0 --lambda_max 1.0 --hutchinson_samples 3 --compute_curvature_every 1

echo "========================================="
echo "Test completed"
echo "End time: $(date)"
echo "========================================="