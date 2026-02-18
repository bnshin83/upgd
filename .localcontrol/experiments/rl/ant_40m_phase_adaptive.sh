EXP_NAME="ant_40m_phase"
EXP_TYPE="rl"
JOB_NAME="rl_ant_ph"
TIME_OVERRIDE="30:00:00"

# 10 schedules x 5 seeds = 50 tasks, max 24 concurrent
ARRAY_RANGE="0-49%24"

# Format: OPTIMIZER:PHASE1:PHASE2:PHASE3 (or just OPTIMIZER for baselines)
METHODS=(
    "upgd_full:h:h:h"    # 1. Uniform hidden (best known)
    "upgd_full:f:f:f"    # 2. Uniform full (runner-up)
    "upgd_full:o:o:o"    # 3. Uniform output (weak)
    "adam"                # 4. No-gating baseline
    "upgd_full:h:h:f"    # 5. Theory-predicted best
    "upgd_full:h:f:f"    # 6. Earlier switch to full
    "upgd_full:h:f:h"    # 7. Full mid, back to hidden
    "upgd_full:f:h:h"    # 8. Start broad, narrow to hidden
    "upgd_full:h:h:o"    # 9. Hidden then output late
    "upgd_full:f:f:h"    # 10. Full then hidden late (control)
)
SEEDS=(1 2 3 4 5)

WANDB_PROJECT="upgd-rl"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    if [[ "${METHOD}" == "adam" ]]; then
        python3 core/run/rl/run_ppo_upgd.py \
            --env_id Ant-v4 \
            --seed ${SEED} \
            --total_timesteps 40000000 \
            --optimizer adam \
            --weight_decay 0.0 \
            --cuda \
            --track \
            --wandb_project_name "${WANDB_PROJECT}" \
            --wandb_entity "${WANDB_ENTITY}"
    else
        IFS=':' read -r OPT P1 P2 P3 <<< "${METHOD}"
        SCHEDULE="${P1}:${P2}:${P3}"

        python3 core/run/rl/run_ppo_upgd.py \
            --env_id Ant-v4 \
            --seed ${SEED} \
            --total_timesteps 40000000 \
            --optimizer ${OPT} \
            --gating_schedule "${SCHEDULE}" \
            --weight_decay 0.0 \
            --beta_utility 0.999 \
            --sigma 0.001 \
            --non_gated_scale 0.5 \
            --learning_rate 3e-4 \
            --cuda \
            --track \
            --wandb_project_name "${WANDB_PROJECT}" \
            --wandb_entity "${WANDB_ENTITY}"
    fi
}
