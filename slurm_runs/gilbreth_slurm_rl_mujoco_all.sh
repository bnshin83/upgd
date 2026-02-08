#!/bin/bash
#SBATCH --job-name=rl_mujoco_all
#SBATCH --account=jhaddock
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gilbreth/shin283/upgd/logs/%j_rl_mujoco_all.out
#SBATCH --error=/scratch/gilbreth/shin283/upgd/logs/%j_rl_mujoco_all.err

cd /scratch/gilbreth/shin283/upgd
module load cuda

export PYTHONPATH=/scratch/gilbreth/shin283/upgd:$PYTHONPATH
# Activate the UPGD conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/gilbreth/shin283/conda_envs/upgd

# WandB Configuration
export WANDB_PROJECT="upgd-rl"
export WANDB_ENTITY="shin283-purdue-university"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1  # Ensure print statements appear immediately

echo "========================================="
echo "Running RL Experiments: All MuJoCo Environments"
echo "Environments: Ant-v4, HalfCheetah-v4, Hopper-v4, Walker2d-v4"
echo "Optimizers: Adam (baseline), UPGD"
echo "Seeds: 0-4"
echo "Start time: $(date)"
echo "========================================="

export CUDA_VISIBLE_DEVICES=0

ENVS="Ant-v4 HalfCheetah-v4 Hopper-v4 Walker2d-v4"
SEEDS="0 1 2 3 4"

for env in $ENVS; do
    env_short=$(echo $env | tr '[:upper:]' '[:lower:]' | sed 's/-v4//')
    
    # Run Adam baseline
    for seed in $SEEDS; do
        echo "Running Adam $env seed $seed..."
        export WANDB_RUN_NAME="${SLURM_JOB_ID}_${env_short}_adam_seed_${seed}"
        python3 core/run/rl/ppo_continuous_action_adam.py \
            --env_id $env \
            --seed $seed \
            --total_timesteps 1000000 \
            --cuda \
            --track \
            --wandb_project_name "upgd-rl" \
            --wandb_entity "shin283-purdue-university"
    done
    
    # Run UPGD
    for seed in $SEEDS; do
        echo "Running UPGD $env seed $seed..."
        export WANDB_RUN_NAME="${SLURM_JOB_ID}_${env_short}_upgd_seed_${seed}"
        python3 core/run/rl/ppo_continuous_action_upgd.py \
            --env_id $env \
            --seed $seed \
            --total_timesteps 1000000 \
            --cuda \
            --track \
            --wandb_project_name "upgd-rl" \
            --wandb_entity "shin283-purdue-university"
    done
done

echo "========================================="
echo "All RL experiments completed"
echo "End time: $(date)"
echo "========================================="
