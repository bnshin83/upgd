#!/bin/bash
#SBATCH --job-name=test_learner_swap
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_test_learner_swap.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_test_learner_swap.err

cd /scratch/gautschi/shin283/upgd
module load cuda python
source .upgd/bin/activate
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

export WANDB_PROJECT="upgd-input-aware"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_LEARNER_SWAP_TEST_input_aware_upgd"
export WANDB_MODE="online"

echo "========================================="
echo "Testing with INPUT-AWARE learner (should have Charts tab)"
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

echo "Input-aware test completed - should have Charts tab!"