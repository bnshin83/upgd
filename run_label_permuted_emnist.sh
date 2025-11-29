#!/bin/bash
#SBATCH --job-name=upgd_label_emnist
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_label_emnist.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_label_emnist.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Display GPU information
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p /scratch/gautschi/shin283/upgd/logs

# Activate the UPGD virtual environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Verify environment setup
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set PYTHONPATH to include the current directory
export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH

echo "========================================="
echo "Starting Label-Permuted EMNIST Experiment (Figure 6)"
echo "Time: $(date)"
echo "========================================="

# Run label-permuted EMNIST experiment
python experiments/label_permuted_emnist.py

if [ $? -eq 0 ]; then
    echo "✓ Label-permuted EMNIST experiment setup completed successfully"
    echo ""
    echo "Generated commands are saved in logs/"
    echo "Run the generated commands to execute the actual experiments"
    echo ""
    echo "After experiments complete, plot results with:"
    echo "python core/plot/plotter.py"
else
    echo "✗ Label-permuted EMNIST experiment failed with exit code $?"
fi

echo ""
echo "Job completed at: $(date)"
