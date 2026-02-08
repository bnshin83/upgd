EXP_NAME="hum_20m"
EXP_TYPE="rl"
JOB_NAME="rl_hum_nc"

METHODS=("upgd_full" "upgd_output_only" "upgd_hidden_only" "adam")
SEEDS=(1 2 3 4 5)

WANDB_PROJECT="upgd-rl"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    python3 core/run/rl/run_ppo_upgd.py \
        --env_id Humanoid-v4 \
        --seed ${SEED} \
        --total_timesteps 20000000 \
        --optimizer ${METHOD} \
        --weight_decay 0.0 \
        --cuda \
        --track \
        --wandb_project_name "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}"
}
