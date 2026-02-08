#!/bin/bash
#SBATCH --job-name=upgd_hiddenonly_imnist_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_hiddenonly_imnist_seed_2.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_hiddenonly_imnist_seed_2.err

cd /scratch/gilbreth/shin283/upgd
module load cuda

export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
# Activate the UPGD conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_hiddenonly_imnist_seed_2"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD Hidden-Only Gating (Input-Permuted MNIST)"
echo "Gating applied ONLY to linear_1, linear_2 (hidden layers)"
echo "linear_3 uses fixed scaling at 0.5 (like SGD)"
echo "Start time: $(date)"
echo "Dataset: Input-Permuted MNIST (10 classes)"
echo "lr=0.01, sigma=0.1, beta=0.9999, wd=0.01"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global_hiddenonly \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.1 \
    --beta_utility 0.9999 \
    --weight_decay 0.01 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "========================================="
echo "Hidden-only gating experiment completed"
echo "End time: $(date)"
echo "========================================"
