EXP_NAME="sant_val_upgd_wd4"
EXP_TYPE="rl"
JOB_NAME="rl_sant_wd4"
TIME_OVERRIDE="24:00:00"

# 1 method x 3 seeds = 3 tasks
ARRAY_RANGE="0-2"

METHODS=("upgd_full")
SEEDS=(1 2 3)

WANDB_PROJECT="upgd-rl"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    python3 core/run/rl/run_ppo_replicate.py \
        --method "${METHOD}" \
        --lr 5e-5 \
        --upgd-wd 1e-4 \
        --beta-utility 0.999 \
        --sigma 0.001 \
        --non-gated-scale 1.0 \
        --seed ${SEED} \
        --total-timesteps 20000000 \
        --cuda \
        --track \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}"
}
