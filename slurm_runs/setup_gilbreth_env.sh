#!/bin/bash
# Setup UPGD environment on Gilbreth cluster

echo "🔧 Setting up UPGD environment on Gilbreth..."

# Set environment path
ENV_PATH="/scratch/gilbreth/shin283/conda_envs/upgd"

# Create conda environment in /scratch to avoid filling home directory
conda create --prefix $ENV_PATH python=3.8 -y

# Use environment's pip directly (no activation needed)
PIP="$ENV_PATH/bin/pip"

# Install PyTorch (compatible with backpack 1.3.0)
$PIP install torch==1.10.0 torchvision==0.11.1

# Install backpack (required version for HesScale)
$PIP install backpack-for-pytorch==1.3.0

# Install other requirements
$PIP install matplotlib==3.5.3 numpy==1.21.0

# Install HesScale
cd /scratch/gilbreth/shin283/upgd
$PIP install HesScale/.

# Install UPGD package
$PIP install -e .

# Install wandb for logging
$PIP install wandb

echo "✅ Environment setup complete!"
echo "To activate: conda activate /scratch/gilbreth/shin283/conda_envs/upgd"

