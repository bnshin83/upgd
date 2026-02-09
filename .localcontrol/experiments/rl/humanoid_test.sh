EXP_NAME="hum_test"
EXP_TYPE="rl"
JOB_NAME="rl_hum_test"
ARRAY_RANGE="0-1"
TIME_OVERRIDE="2:00:00"

METHODS=("upgd_full" "upgd_output_only" "upgd_hidden_only" "adam")
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

WANDB_PROJECT="upgd-rl"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    python3 core/run/rl/run_ppo_upgd.py \
        --env_id Humanoid-v4 \
        --seed ${SEED} \
        --total_timesteps 200000 \
        --optimizer ${METHOD} \
        --weight_decay 0.0 \
        --cuda \
        --track \
        --wandb_project_name "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}"
}
