#!/bin/bash
#SBATCH --job-name=upgd_scale_ablation_emnist
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_scale_ablation_emnist.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_scale_ablation_emnist.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"

echo "========================================="
echo "UPGD Ablation Study"
echo ""
echo "1. Non-gated scale ablation (output-only gating):"
echo "   scale=0.0  (hidden frozen)"
echo "   scale=0.27 (hidden max protection)"
echo "   scale=0.5  (hidden neutral, default)"
echo "   scale=0.73 (hidden min protection)"
echo "   scale=1.0  (hidden full SGD)"
echo ""
echo "2. Output frozen (output layer completely frozen)"
echo ""
echo "3. Freeze high utility (params with s >= 0.52 frozen)"
echo ""
echo "Start time: $(date)"
echo "Dataset: EMNIST (47 classes)"
echo "lr=0.01, sigma=0.001, beta=0.9, wd=0.0"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0

# Common parameters
TASK="label_permuted_emnist_stats"
SEED=0
LR=0.01
SIGMA=0.001
BETA=0.9
WD=0.0
NETWORK="fully_connected_relu_with_hooks"
N_SAMPLES=1000000
CURVATURE_EVERY=1000000

# Scale 0.0 (frozen hidden layers)
echo ""
echo ">>> Running scale=0.0 (frozen) ..."
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputonly_scale0_emnist"
python3 core/run/run_stats_with_curvature.py \
    --task $TASK \
    --learner upgd_fo_global_outputonly_scale0 \
    --seed $SEED \
    --lr $LR \
    --sigma $SIGMA \
    --beta_utility $BETA \
    --weight_decay $WD \
    --network $NETWORK \
    --n_samples $N_SAMPLES \
    --compute_curvature_every $CURVATURE_EVERY \
    --save_path logs

# Scale 0.27 (max protection)
echo ""
echo ">>> Running scale=0.27 (max protection) ..."
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputonly_scale027_emnist"
python3 core/run/run_stats_with_curvature.py \
    --task $TASK \
    --learner upgd_fo_global_outputonly_scale27 \
    --seed $SEED \
    --lr $LR \
    --sigma $SIGMA \
    --beta_utility $BETA \
    --weight_decay $WD \
    --network $NETWORK \
    --n_samples $N_SAMPLES \
    --compute_curvature_every $CURVATURE_EVERY \
    --save_path logs

# Scale 0.5 (neutral, default)
echo ""
echo ">>> Running scale=0.5 (neutral) ..."
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputonly_scale50_emnist"
python3 core/run/run_stats_with_curvature.py \
    --task $TASK \
    --learner upgd_fo_global_outputonly_scale50 \
    --seed $SEED \
    --lr $LR \
    --sigma $SIGMA \
    --beta_utility $BETA \
    --weight_decay $WD \
    --network $NETWORK \
    --n_samples $N_SAMPLES \
    --compute_curvature_every $CURVATURE_EVERY \
    --save_path logs

# Scale 0.73 (min protection)
echo ""
echo ">>> Running scale=0.73 (min protection) ..."
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputonly_scale73_emnist"
python3 core/run/run_stats_with_curvature.py \
    --task $TASK \
    --learner upgd_fo_global_outputonly_scale73 \
    --seed $SEED \
    --lr $LR \
    --sigma $SIGMA \
    --beta_utility $BETA \
    --weight_decay $WD \
    --network $NETWORK \
    --n_samples $N_SAMPLES \
    --compute_curvature_every $CURVATURE_EVERY \
    --save_path logs

# Scale 1.0 (full SGD)
echo ""
echo ">>> Running scale=1.0 (full SGD) ..."
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputonly_scale1_emnist"
python3 core/run/run_stats_with_curvature.py \
    --task $TASK \
    --learner upgd_fo_global_outputonly_scale1 \
    --seed $SEED \
    --lr $LR \
    --sigma $SIGMA \
    --beta_utility $BETA \
    --weight_decay $WD \
    --network $NETWORK \
    --n_samples $N_SAMPLES \
    --compute_curvature_every $CURVATURE_EVERY \
    --save_path logs

# Output frozen (output layer completely frozen, hidden layers full SGD)
echo ""
echo ">>> Running output frozen (output=frozen, hidden=full SGD) ..."
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_outputfrozen_emnist"
python3 core/run/run_stats_with_curvature.py \
    --task $TASK \
    --learner upgd_fo_global_outputfrozen \
    --seed $SEED \
    --lr $LR \
    --sigma $SIGMA \
    --beta_utility $BETA \
    --weight_decay $WD \
    --network $NETWORK \
    --n_samples $N_SAMPLES \
    --compute_curvature_every $CURVATURE_EVERY \
    --save_path logs

# Freeze high-utility params (scaled_utility >= 0.52)
echo ""
echo ">>> Running freeze high utility (s >= 0.52 frozen) ..."
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_freezehigh52_emnist"
python3 core/run/run_stats_with_curvature.py \
    --task $TASK \
    --learner upgd_fo_global_freezehigh52 \
    --seed $SEED \
    --lr $LR \
    --sigma $SIGMA \
    --beta_utility $BETA \
    --weight_decay $WD \
    --network $NETWORK \
    --n_samples $N_SAMPLES \
    --compute_curvature_every $CURVATURE_EVERY \
    --save_path logs

echo "========================================="
echo "Scale ablation experiment completed"
echo "End time: $(date)"
echo "========================================="
