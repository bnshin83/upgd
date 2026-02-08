#!/bin/bash
#SBATCH --job-name=upgd_mini_imagenet
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_mini_imagenet.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_mini_imagenet.err

echo "============================================"
echo "Mini-ImageNet Experiment"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "============================================"

# Change to the project directory
cd /scratch/gilbreth/shin283/upgd

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module purge
module load external
module load anaconda/2024.10-py312
module load cuda/12.6.0

# Activate conda environment
conda activate upgd

# Display GPU information
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Verify environment
echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set PYTHONPATH
export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH

echo ""
echo "========================================="
echo "Running Label-Permuted Mini-ImageNet"
echo "========================================="

# Run the experiment
python -c "
from core.task.label_permuted_mini_imagenet import LabelPermutedMiniImageNet
import torch

print('Loading task...')
task = LabelPermutedMiniImageNet(batch_size=32)
print(f'Dataset size: {len(task.dataset)} samples')
print(f'Feature dimension: {task.n_inputs}')
print(f'Number of classes: {task.n_outputs}')

# Test a few batches
print('\nTesting batches...')
for i, (x, y) in enumerate(task):
    print(f'Batch {i}: x.shape={x.shape}, y.shape={y.shape}')
    if i >= 5:
        break

print('\n✓ Mini-ImageNet task loaded successfully!')
"

echo ""
echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"
