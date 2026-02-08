#!/bin/bash
#SBATCH --job-name=upgd_clamped_44_56_mini_imagenet_seed_2
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_clamped_44_56_mini_imagenet_seed_2.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_upgd_clamped_44_56_mini_imagenet_seed_2.err

cd /scratch/gilbreth/shin283/upgd
module load cuda

export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
# Activate the UPGD conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_clamped_44_56_mini_imagenet_seed_2_lr_0.01_sigma_0.001_beta_0.9_wd_0.0"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD Symmetric Clamping [0.44, 0.56] (Mini-ImageNet)"
echo "Narrow range (±0.06 from 0.5, width=0.12)"
echo "Tests: More utility range than [0.48,0.52]"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_mini_imagenet_stats \
    --learner upgd_fo_global_clamped_44_56 \
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
echo "Symmetric clamping [0.44, 0.56] completed"
echo "End time: $(date)"
echo "Hypothesis: Should perform better than [0.48,0.52]!"
echo "========================================"
