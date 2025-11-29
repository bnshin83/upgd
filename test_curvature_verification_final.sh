#!/bin/bash
#SBATCH --job-name=test_curvature_final
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_curvature_final.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_curvature_final.err

cd /scratch/gautschi/shin283/upgd
module load cuda python
source .upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# WandB Configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_final_curvature_test_upgd_fo_global_samples_50_seed_0"
export WANDB_MODE="online"

echo "========================================="
echo "FINAL TEST: Real Curvature Computation for Standard UPGD"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.001"
echo "Total samples: 50"
echo "Compute curvature every: 5 steps"
echo "Expected: REAL curvature values in both WandB and JSON"
echo "========================================="

# Run with frequent curvature computation for verification
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
echo "Final verification completed"
echo "End time: $(date)"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "Check:"
echo "1. WandB curvature charts should show REAL values (not zeros)"
echo "2. JSON file should contain input_curvature_per_step with real values"
echo "3. Standard UPGD computes curvature but doesn't use it for updates"
echo "========================================"