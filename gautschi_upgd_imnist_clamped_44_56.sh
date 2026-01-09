#!/bin/bash
#SBATCH --job-name=upgd_clamped_44_56_imnist_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_clamped_44_56_imnist_seed_2.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_clamped_44_56_imnist_seed_2.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_clamped_44_56_imnist_seed_2"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD Symmetric Clamping [0.44, 0.56] (Input-Permuted MNIST)"
echo "Medium range (±0.06 from 0.5, width=0.12)"
echo "Start time: $(date)"
echo "Dataset: Input-Permuted MNIST (10 classes)"
echo "lr=0.01, sigma=0.1, beta=0.9999, wd=0.01"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global_clamped_44_56 \
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
echo "Symmetric clamping [0.44, 0.56] completed"
echo "End time: $(date)"
echo "========================================"
