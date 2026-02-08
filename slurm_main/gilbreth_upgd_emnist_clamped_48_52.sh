#!/bin/bash
#SBATCH --job-name=upgd_clamped_48_52_emnist_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_clamped_48_52_emnist_seed_2.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_clamped_48_52_emnist_seed_2.err

cd /scratch/gilbreth/shin283/upgd
module load cuda

export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
# Activate the UPGD conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_clamped_48_52_emnist_seed_2"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD Symmetric Clamping [0.48, 0.52] (EMNIST)"
echo "Very narrow range (±0.02 from 0.5, width=0.04)"
echo "Tests: Can UPGD work with just core 99% of utilities?"
echo "Start time: $(date)"
echo "Dataset: EMNIST (47 classes)"
echo "lr=0.01, sigma=0.001, beta=0.9, wd=0.0"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_emnist_stats \
    --learner upgd_fo_global_clamped_48_52 \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.001 \
    --beta_utility 0.9 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "========================================="
echo "Symmetric clamping [0.48, 0.52] completed"
echo "End time: $(date)"
echo "========================================"
