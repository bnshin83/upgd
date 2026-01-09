#!/bin/bash
#SBATCH --job-name=layer_umax_imnist_hiddenandoutput_s2
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_layer_umax_imnist_hiddenandoutput_s2.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_layer_umax_imnist_hiddenandoutput_s2.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_RUN_NAME="${SLURM_JOB_ID}_layer_umax_imnist_hiddenandoutput_seed_2"
export WANDB_MODE="online"

echo "========================================="
echo "Re-run for per-layer max logging (Input-Permuted MNIST)"
echo "Mode: hidden+output gating (all layers gated via layer-selective optimizer)"
echo "Seed: 2"
echo "lr=0.01, sigma=0.1, beta=0.9999, wd=0.01"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
  --task input_permuted_mnist_stats \
  --learner upgd_fo_global_hiddenandoutput \
  --seed 2 \
  --lr 0.01 \
  --sigma 0.1 \
  --beta_utility 0.9999 \
  --weight_decay 0.01 \
  --network fully_connected_relu_with_hooks \
  --n_samples 1000000 \
  --compute_curvature_every 1000000 \
  --save_path logs

echo "Completed: $(date)"


