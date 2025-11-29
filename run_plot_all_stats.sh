#!/bin/bash
#SBATCH --job-name=plot_all_stats
#SBATCH --account=jhaddock
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_plot_all_stats.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_plot_all_stats.err

# Script to generate plots for all statistics directories

# Set up environment
echo "Starting plot generation at $(date)"
echo "Working directory: $(pwd)"

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load modules
module load python

# Activate conda environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Create plots directory
PLOT_DIR="/scratch/gautschi/shin283/upgd/plots_all_stats"
mkdir -p "$PLOT_DIR"

echo "Output directory: $PLOT_DIR"

# Run the plotting script with all statistics directories
python plot_all_stats.py \
    --base-dirs \
        "/scratch/gautschi/shin283/upgd/logs/input_permuted_mnist_stats" \
        "/scratch/gautschi/shin283/upgd/logs/label_permuted_cifar10_stats" \
        "/scratch/gautschi/shin283/upgd/logs/label_permuted_emnist_stats" \
        "/scratch/gautschi/shin283/upgd/logs/label_permuted_mini_imagenet_stats" \
    --output-dir "$PLOT_DIR"

# Check if plotting was successful
if [ $? -eq 0 ]; then
    echo "Plotting completed successfully!"
    echo "Plots saved to: $PLOT_DIR"
    ls -la "$PLOT_DIR"
else
    echo "Error: Plotting failed"
    exit 1
fi

echo "Plot generation completed at $(date)"