#!/bin/bash
#SBATCH --job-name=upgd_scale0_emnist
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_scale0_emnist.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_scale0_emnist.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputonly_scale0_emnist"

echo "========================================="
echo "UPGD Ablation: scale=0.0 (hidden frozen)"
echo "Output layer: utility gated"
echo "Hidden layers: frozen (no updates)"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_emnist_stats \
    --learner upgd_fo_global_outputonly_scale0 \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --beta_utility 0.9 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "Completed: $(date)"
