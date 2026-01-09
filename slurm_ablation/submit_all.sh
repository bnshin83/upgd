#!/bin/bash
# Submit all 7 ablation study SLURM jobs in parallel

echo "Submitting 7 ablation study jobs..."

sbatch slurm_ablation/ablation_scale0.sh
sbatch slurm_ablation/ablation_scale27.sh
sbatch slurm_ablation/ablation_scale50.sh
sbatch slurm_ablation/ablation_scale73.sh
sbatch slurm_ablation/ablation_scale1.sh
sbatch slurm_ablation/ablation_outputfrozen.sh
sbatch slurm_ablation/ablation_freezehigh52.sh

echo "All jobs submitted. Use 'squeue -u \$USER' to check status."
