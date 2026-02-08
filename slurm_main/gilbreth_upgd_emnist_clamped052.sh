#!/bin/bash
#SBATCH --job-name=upgd_clamped052_emnist_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_clamped052_emnist_seed_2.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_clamped052_emnist_seed_2.err

cd /scratch/gilbreth/shin283/upgd
module load cuda

export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
# Activate the UPGD conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_clamped052_emnist_seed_2"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD Clamped-0.52 (EMNIST)"
echo "Testing importance of utilities > 0.52"
echo "Start time: $(date)"
echo "Dataset: EMNIST (47 classes)"
echo "lr=0.01, sigma=0.001, beta=0.9, wd=0.0"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_emnist_stats \
    --learner upgd_fo_global_clamped052 \
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
echo "Clamped UPGD experiment completed"
echo "End time: $(date)"
echo "========================================"
