#!/bin/bash
#SBATCH --job-name=test_wandb_curvature_fix
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_wandb_curvature_fix.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_wandb_curvature_fix.err

cd /scratch/gautschi/shin283/upgd
module load cuda python
source .upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# WandB Configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_FINAL_wandb_curvature_fix_upgd_fo_global_samples_30_seed_0"
export WANDB_MODE="online"

echo "========================================="
echo "FINAL FIX TEST: Real Curvature in WandB Charts"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.001"
echo "Total samples: 30"
echo "Compute curvature every: 3 steps"
echo "Expected: REAL curvature values in WandB curvature/* charts"
echo "========================================="

# Run with frequent curvature computation to verify WandB charts
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 30 \
    --compute_curvature_every 3 \
    --save_path logs

echo "========================================="
echo "WandB curvature fix verification completed"
echo "End time: $(date)"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo ""
echo "EXPECTED RESULTS:"
echo "✅ WandB curvature/current: REAL values (not zeros)"
echo "✅ WandB curvature/avg: REAL values (not zeros)"
echo "✅ WandB curvature/max: REAL values (not zeros)"
echo "✅ JSON input_curvature_per_step: REAL values"
echo "✅ Standard UPGD with curvature analysis capability"
echo "========================================"