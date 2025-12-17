#!/bin/bash
#SBATCH --job-name=incremental_cifar_git
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=13-1:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_incremental_cifar.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_incremental_cifar.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Display GPU information
nvidia-smi

# Set environment variable for output directory
export OUTPUT_DIR=/scratch/gautschi/shin283/lop

# Create results2 directory for checkpoints
mkdir -p /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results


# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Run the Python script with specified arguments
python3.8 incremental_cifar_experiment.py --config ./cfg/base_deep_learning_system.json --verbose --experiment-index 0