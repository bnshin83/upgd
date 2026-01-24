#!/bin/bash
#SBATCH --job-name=rl_ant_adam_seeds
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=5
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/upgd/logs/%j_rl_ant_adam_seeds.out
#SBATCH --error=/scratch/gautschi/shin283/upgd/logs/%j_rl_ant_adam_seeds.err

cd /scratch/gautschi/shin283/upgd
module load cuda
module load python

export PYTHONPATH=/scratch/gautschi/shin283/upgd:$PYTHONPATH
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# WandB Configuration
export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Running RL: Ant-v4 - Adam"
echo "Seeds: 0-4 (5 seeds in parallel)"
echo "Total timesteps: 20,000,000 (matching LR schedule)"
echo "Start time: $(date)"
echo "========================================="

# Run 5 seeds in parallel, each on a different GPU
for seed in 0 1 2 3 4; do
    export CUDA_VISIBLE_DEVICES=$seed
    export WANDB_RUN_NAME="${SLURM_JOB_ID}_ant_adam_5M_seed_${seed}"
    python3 core/run/rl/ppo_continuous_action_adam.py \
        --env_id Ant-v4 \
        --seed $seed \
        --total_timesteps 20000000 \
        --cuda \
        --track \
        --wandb_project_name "upgd-rl" \
        --wandb_entity "shin283-purdue-university" &
done

# Wait for all background jobs to complete
wait

echo "========================================="
echo "All Adam seed experiments completed"
echo "End time: $(date)"
echo "========================================="
