EXP_NAME="qt_humanoid"
EXP_TYPE="rl"
JOB_NAME="qt_hum"
TIME_OVERRIDE="12:00:00"

METHODS=("upgd_full" "upgd_output_only" "upgd_hidden_only" "adam")

WANDB_PROJECT="upgd-rl"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    python3 core/run/rl/run_ppo_upgd.py \
        --env_id Humanoid-v4 \
        --seed 42 \
        --total_timesteps 2000000 \
        --optimizer ${METHOD} \
        --weight_decay 0.0 \
        --cuda \
        --track \
        --wandb_project_name "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}"
}
