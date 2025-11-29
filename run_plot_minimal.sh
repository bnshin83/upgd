#!/bin/bash
#SBATCH --job-name=plot_minimal
#SBATCH --account=jhaddock
#SBATCH --partition=cpu
#SBATCH --time=30:00
#SBATCH --mem=8G
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_plot_minimal.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_plot_minimal.err

# Minimal plotting script
cd /scratch/gautschi/shin283/upgd

# Activate environment
source /scratch/gautschi/shin283/conda_envs/plasticity/bin/activate

# Run plotting
python plot_all_stats.py --output-dir plots_all_stats