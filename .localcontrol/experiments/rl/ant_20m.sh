EXP_NAME="ant_20m"
EXP_TYPE="rl"
JOB_NAME="rl_ant_nc"
TIME_OVERRIDE="14:00:00"

METHODS=("upgd_full" "upgd_output_only" "upgd_hidden_only" "upgd_full_old" "adam")
SEEDS=(1 2 3 4 5)

WANDB_PROJECT="upgd-rl"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    python3 core/run/rl/run_ppo_upgd.py \
        --env_id Ant-v4 \
        --seed ${SEED} \
        --total_timesteps 20000000 \
        --optimizer ${METHOD} \
        --weight_decay 0.0 \
        --cuda \
        --track \
        --wandb_project_name "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}"
}
