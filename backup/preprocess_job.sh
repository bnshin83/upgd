#!/bin/bash
#SBATCH --job-name=preprocess_imagenet
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/preprocess_%j.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/preprocess_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --account=jhaddock
#SBATCH --partition=ai

echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "============================================"

# Load modules
module load cuda
module load python

# Go to project directory
cd /scratch/gautschi/shin283/upgd

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run preprocessing
echo ""
echo "Starting preprocessing..."
python3 preprocess_imagenet.py

echo ""
echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"

