#!/bin/bash
#SBATCH --job-name=preprocess_imagenet
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb

echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "============================================"

# Load modules
module purge
module load external
module load anaconda/2024.10-py312
module load cuda/12.6.0

# Activate conda environment
conda activate upgd

# Go to project directory
cd /scratch/gilbreth/shin283/upgd

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run preprocessing
echo ""
echo "Starting preprocessing..."
python preprocess_imagenet.py

echo ""
echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"

