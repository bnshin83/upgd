EXP_NAME="sant_val_shrink_head"
EXP_TYPE="rl"
JOB_NAME="rl_sant_sh"
TIME_OVERRIDE="24:00:00"

# 1 method x 3 seeds = 3 tasks
ARRAY_RANGE="0-2"

METHODS=("shrink_head")
SEEDS=(1 2 3)

WANDB_PROJECT="upgd-rl"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    python3 core/run/rl/run_ppo_replicate.py \
        --method "${METHOD}" \
        --lr 1e-4 \
        --seed ${SEED} \
        --total-timesteps 20000000 \
        --cuda \
        --track \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}"
}
