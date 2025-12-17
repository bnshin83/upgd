#!/bin/bash
#SBATCH --job-name=test_real_curvature_upgd_fo_global
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_real_curvature_upgd_fo_global.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_real_curvature_upgd_fo_global.err

cd /scratch/gautschi/shin283/upgd
module load cuda python
source .upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# WandB Configuration
export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_test_real_curvature_upgd_fo_global_samples_200_seed_0"
export WANDB_MODE="online"

echo "========================================="
echo "Testing REAL Input Curvature for Standard UPGD"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Learning Rate: 0.01"
echo "Sigma: 0.001"
echo "Total samples: 200"
echo "Compute curvature every: 10 steps"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "========================================="

# Run UPGD FO Global with REAL curvature computation (but not used for updates)
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 200 \
    --compute_curvature_every 10 \
    --save_path logs

echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "Check WandB dashboard - curvature values should now be REAL (not zero)!"
echo "Standard UPGD should show actual input curvature for analysis"
echo "========================================"