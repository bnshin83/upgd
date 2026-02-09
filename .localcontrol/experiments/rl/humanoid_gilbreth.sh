EXP_NAME="hum_20m"
EXP_TYPE="custom"  # Use custom type to avoid standard RL mapping
JOB_NAME="rl_hum_20m_gb"

# Gilbreth: 10 array tasks, each runs 2 adam seeds in parallel
# Tasks 0-9, each using 2 GPUs (full node)
CUSTOM_ARRAY="0-9%3"

# Longer time for running 2 seeds in parallel
TIME_OVERRIDE="20:00:00"

# Override to use full node (2 GPUs)
SBATCH_GPUS_OVERRIDE="2"
SBATCH_CPUS_OVERRIDE="14"
SBATCH_MEM_OVERRIDE="64G"

WANDB_PROJECT="upgd-rl"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    # Each array task runs 2 consecutive seeds in parallel
    # Task 0: seeds 0,1  Task 1: seeds 2,3  ...  Task 9: seeds 18,19
    SEED1=$((SLURM_ARRAY_TASK_ID * 2))
    SEED2=$((SLURM_ARRAY_TASK_ID * 2 + 1))

    echo "=========================================="
    echo "Running adam seeds $SEED1 and $SEED2 in parallel"
    echo "=========================================="

    # Run seed 1 on GPU 0 in background
    CUDA_VISIBLE_DEVICES=0 \
    WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_hum_20m_adam_seed${SEED1}" \
    python3 core/run/rl/run_ppo_upgd.py \
        --env_id Humanoid-v4 \
        --seed $SEED1 \
        --total_timesteps 20000000 \
        --optimizer adam \
        --weight_decay 0.0 \
        --cuda \
        --track \
        --wandb_project_name "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}" &

    PID1=$!

    # Run seed 2 on GPU 1 in background
    CUDA_VISIBLE_DEVICES=1 \
    WANDB_RUN_NAME="${SLURM_ARRAY_JOB_ID}_hum_20m_adam_seed${SEED2}" \
    python3 core/run/rl/run_ppo_upgd.py \
        --env_id Humanoid-v4 \
        --seed $SEED2 \
        --total_timesteps 20000000 \
        --optimizer adam \
        --weight_decay 0.0 \
        --cuda \
        --track \
        --wandb_project_name "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}" &

    PID2=$!

    # Wait for both to complete
    echo "Waiting for seed $SEED1 (PID $PID1) and seed $SEED2 (PID $PID2)..."
    wait $PID1
    EXIT1=$?
    wait $PID2
    EXIT2=$?

    echo "Seed $SEED1 exit code: $EXIT1"
    echo "Seed $SEED2 exit code: $EXIT2"

    if [ $EXIT1 -ne 0 ] || [ $EXIT2 -ne 0 ]; then
        echo "ERROR: One or both seeds failed"
        exit 1
    fi

    echo "Both seeds completed successfully at $(date)"
}
