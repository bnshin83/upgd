#!/bin/bash
#SBATCH --job-name=debug_input_aware
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_debug_input_aware.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_debug_input_aware.err

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
echo "Debug Input-Aware UPGD vs PGD baseline"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0

echo "Testing PGD baseline with 10K samples..."
time python3 core/run/run.py --task label_permuted_emnist --learner pgd --seed 0 --lr 0.01 --sigma 0.001 --network fully_connected_relu --n_samples 10000

echo ""
echo "Testing input-aware UPGD with 10K samples..."
time python3 core/run/run.py --task label_permuted_emnist --learner input_aware_upgd_fo_global --seed 0 --lr 0.01 --sigma 0.001 --network fully_connected_relu --n_samples 10000 --curvature_threshold 10.0 --lambda_max 1.0 --hutchinson_samples 3 --compute_curvature_every 1

echo "========================================="
echo "Debug completed"
echo "End time: $(date)"
echo "========================================="