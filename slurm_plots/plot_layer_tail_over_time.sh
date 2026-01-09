#!/bin/bash
#SBATCH --job-name=upgd_plot_layer_tail
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_plot_layer_tail.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_plot_layer_tail.err

# Usage:
#   sbatch plot_layer_tail_over_time.sh emnist
#
# Outputs:
#   /scratch/gautschi/shin283/upgd/upgd_plots/figures/<dataset>/
#     - layer_tail_over_time.png
#     - layer_tail_over_time_log.png
#     - layer_raw_umax_over_time.png (if present in logs)
#     - layer_raw_umax_over_time_log.png (if present in logs)

set -euo pipefail

DATASET="${1:-emnist}"

cd /scratch/gautschi/shin283/upgd
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Plot jobs should never try to log to WandB
export WANDB_MODE="disabled"

echo "========================================="
echo "Plotting per-layer tail + raw umax over time"
echo "Dataset: ${DATASET}"
echo "Start time: $(date)"
echo "========================================="

python3 upgd_plots/scripts/plot_layer_tail_over_time.py "${DATASET}"

echo "Completed: $(date)"


