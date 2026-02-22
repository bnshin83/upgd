EXP_NAME="sant_val_cbp_layers"
EXP_TYPE="rl"
JOB_NAME="rl_sant_cbpl"
TIME_OVERRIDE="24:00:00"

# 3 methods x 3 seeds = 9 tasks
ARRAY_RANGE="0-8"

METHODS=("cbp_h1" "cbp_h2" "cbp_no_gnt")
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
