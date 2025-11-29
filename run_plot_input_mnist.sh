#!/bin/bash
#SBATCH --job-name=plot_input_mnist
#SBATCH --output=logs/plot_input_mnist_%j.out
#SBATCH --error=logs/plot_input_mnist_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Script to generate plots for input_permuted_mnist experiments

# Set up environment
echo "Starting plot generation at $(date)"
echo "Working directory: $(pwd)"

# Create plots directory if it doesn't exist
PLOT_DIR="/scratch/gautschi/shin283/upgd/plots/input_permuted_mnist"
mkdir -p "$PLOT_DIR"

# Path to the experimental data
DATA_PATH="/scratch/gautschi/shin283/upgd/logs/input_permuted_mnist/pgd/fully_connected_relu/lr_0.01_sigma_0.001/0.json"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    exit 1
fi

echo "Data file found: $DATA_PATH"
echo "Output directory: $PLOT_DIR"

# Run the plotting script
python /scratch/gautschi/shin283/upgd/plot_input_permuted_mnist.py \
    --data-path "$DATA_PATH" \
    --output-dir "$PLOT_DIR" \
    --tasks-per-plot 5000 \
    --window-size 1000

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
