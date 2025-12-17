#!/bin/bash
#SBATCH --job-name=test_revert_fix
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_revert_fix.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_revert_fix.err

cd /scratch/gautschi/shin283/upgd
module load cuda python
source .upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_REVERT_FIX_TEST_input_aware"
export WANDB_MODE="online"

echo "========================================="
echo "Testing if input-aware Charts tab is restored after reverting WandB config change"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task input_permuted_mnist_stats \
    --learner upgd_input_aware_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 20 \
    --compute_curvature_every 1 \
    --save_path logs

echo "Revert test completed - check if Charts tab is back!"