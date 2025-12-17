#!/bin/bash
#SBATCH --job-name=upgd_clamped052_mini_imagenet_seed_2_test_tail_importance
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_upgd_clamped052_mini_imagenet_seed_2.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_upgd_clamped052_mini_imagenet_seed_2.err

# Change to the submission directory
cd /scratch/gautschi/shin283/upgd

# Load CUDA and Python modules
module load cuda
module load python

# Set PYTHONPATH first (before activating venv to ensure priority)
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd"
export WANDB_ENTITY="shin283-purdue-university"  # Set entity to avoid team conflict
export WANDB_RUN_NAME="${SLURM_JOB_ID}_upgd_clamped052_mini_imagenet_seed_2_lr_0.01_sigma_0.001_beta_0.9_wd_0.0"
export WANDB_MODE="online"

echo "========================================="
echo "Running UPGD FO Global CLAMPED-0.52 Mini-ImageNet (Ablation Study)"
echo "Testing importance of utilities > 0.52 (top ~0.1% tail)"
echo "Start time: $(date)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Dataset: Mini-ImageNet (100 classes, ResNet50 bottleneck features, 2048-dim)"
echo "Learning Rate: 0.01 (same as standard UPGD)"
echo "Sigma: 0.001 (same as standard UPGD)"
echo "Beta Utility: 0.9 (same as standard UPGD)"
echo "Weight Decay: 0.0 (same as standard UPGD)"
echo "Clamping: Utilities clamped to max 0.52"
echo "Seed: 2"
echo "Total samples: 1000000"
echo "Network: fully_connected_relu_with_hooks"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Entity: $WANDB_ENTITY"
echo "WandB Run: $WANDB_RUN_NAME"
echo "Save path: logs"
echo "NOTE: This requires Mini-ImageNet dataset files in dataset/ directory"
echo "      (mini-imagenet_targets.pkl and processed_imagenet.pkl)"
echo "========================================="

# Run UPGD FO Global with utilities clamped to 0.52
export CUDA_VISIBLE_DEVICES=0
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_mini_imagenet_stats \
    --learner upgd_fo_global_clamped052 \
    --seed 2 \
    --lr 0.01 \
    --sigma 0.001 \
    --beta_utility 0.9 \
    --weight_decay 0.0 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --compute_curvature_every 1000000 \
    --save_path logs

echo "========================================="
echo "Clamped UPGD experiment completed"
echo "End time: $(date)"
echo "JSON results saved to: logs/label_permuted_mini_imagenet_stats/upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000/2.json"
echo "WandB Run: ${WANDB_RUN_NAME}"
echo "SLURM logs: /scratch/gautschi/shin283/upgd/logs/${SLURM_JOB_ID}.out"
echo "========================================="
echo "Experiment Design:"
echo "  Standard UPGD: utilities ∈ [0, 1] with ~99% in [0.48, 0.52]"
echo "  Clamped UPGD:  utilities ∈ [0, 0.52] (removes top ~0.1% tail)"
echo ""
echo "Hypothesis Test:"
echo "  If clamped < standard: The >0.52 tail is critical"
echo "  If clamped ≈ standard: The [0.48, 0.52] core is sufficient"
echo ""
echo "WandB Tracking:"
echo "  - clamping/percentage: % of params hitting 0.52 ceiling"
echo "  - clamping/mean_clamped vs mean_unclamped"
echo "  - histograms/scaled_utility_clamped vs unclamped"
echo "========================================"
