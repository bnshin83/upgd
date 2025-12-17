#!/bin/bash
#SBATCH --job-name=test_charts_fix
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_charts_fix.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_charts_fix.err

cd /scratch/gautschi/shin283/upgd
module load cuda python
source .upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# WandB Configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_CHARTS_FIX_TEST_upgd_fo_global_samples_50_seed_0"
export WANDB_MODE="online"

echo "========================================="
echo "Testing Charts Tab Fix with Minimal Run"
echo "Start time: $(date)"
echo "Expected: Charts tab should appear with comprehensive metrics"
echo "========================================="

# Very short run to test Charts tab creation
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 50 \
    --compute_curvature_every 5 \
    --save_path logs

echo "========================================="
echo "Charts fix test completed"
echo "End time: $(date)"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "Check WandB dashboard for Charts tab"
echo "========================================"