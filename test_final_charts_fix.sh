#!/bin/bash
#SBATCH --job-name=final_charts_fix
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_final_charts_fix.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_final_charts_fix.err

cd /scratch/gautschi/shin283/upgd
module load cuda python
source .upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_FINAL_CHARTS_FIX_upgd_fo_global_samples_30"
export WANDB_MODE="online"

echo "========================================="
echo "FINAL Charts Tab Fix - WandB Treats All As Input-Aware"
echo "Expected: Charts tab should now appear!"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 30 \
    --compute_curvature_every 1 \
    --save_path logs

echo "Final test completed - Check WandB for Charts tab!"