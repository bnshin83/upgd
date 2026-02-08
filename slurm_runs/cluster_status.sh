#!/bin/bash
# cluster_status.sh — Run on cluster to report experiment status
# Deployed to /scratch/{cluster}/shin283/upgd/slurm_runs/
# Called by local monitor.sh or run manually: bash cluster_status.sh

CLUSTER_NAME="${1:-$(hostname -s)}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGS_DIR="${BASE_DIR}/logs"

TASKS=("label_permuted_emnist_stats" "label_permuted_cifar10_stats" "label_permuted_mini_imagenet_stats" "input_permuted_mnist_stats")
TASK_SHORT=("emnist" "cifar10" "mini_imagenet" "imnist")
LEARNERS=("adam" "ewc" "mas" "si" "shrink_and_perturb" "sgd" "upgd_fo_global" "upgd_fo_global_outputonly" "upgd_fo_global_hiddenonly")
LEARNER_SHORT=("Adam" "EWC" "MAS" "SI" "S&P" "SGD" "UPGD-Full" "UPGD-Out" "UPGD-Hid")
NETWORK="fully_connected_relu_with_hooks"

echo "CLUSTER:${CLUSTER_NAME}"
echo "TIMESTAMP:$(date '+%Y-%m-%d %H:%M:%S')"

# --- Queue status ---
echo ""
echo "SECTION:QUEUE"
RUNNING=$(squeue -u shin283 -t RUNNING -h 2>/dev/null | wc -l)
PENDING=$(squeue -u shin283 -t PENDING -h 2>/dev/null | wc -l)
echo "RUNNING:${RUNNING}"
echo "PENDING:${PENDING}"

# Per-job-name breakdown
squeue -u shin283 --format="%.20j %.8T" -h 2>/dev/null | sort | uniq -c | while read count name state; do
    echo "JOB:${name}:${state}:${count}"
done

# --- Failures since Feb 5 ---
echo ""
echo "SECTION:FAILURES"
FAILED_COUNT=$(sacct -u shin283 --starttime=2026-02-05 --state=FAILED -n 2>/dev/null | wc -l)
echo "FAILED_TOTAL:${FAILED_COUNT}"
if [ "$FAILED_COUNT" -gt 0 ]; then
    sacct -u shin283 --starttime=2026-02-05 --state=FAILED --format=JobID%-20,JobName%-25,State,ExitCode,Elapsed -n 2>/dev/null | head -20
fi

# --- OOM / timeout kills ---
OOM_COUNT=$(sacct -u shin283 --starttime=2026-02-05 --state=OUT_OF_MEMORY -n 2>/dev/null | wc -l)
TIMEOUT_COUNT=$(sacct -u shin283 --starttime=2026-02-05 --state=TIMEOUT -n 2>/dev/null | wc -l)
echo "OOM:${OOM_COUNT}"
echo "TIMEOUT:${TIMEOUT_COUNT}"

# --- Seed completion matrix ---
echo ""
echo "SECTION:SEEDS"
for t_idx in "${!TASKS[@]}"; do
    task=${TASKS[$t_idx]}
    task_short=${TASK_SHORT[$t_idx]}
    for l_idx in "${!LEARNERS[@]}"; do
        learner=${LEARNERS[$l_idx]}
        learner_short=${LEARNER_SHORT[$l_idx]}

        seeds_done=""
        count=0
        for seed in 0 1 2 3 4; do
            json="${LOGS_DIR}/${task}/${learner}/${NETWORK}/seed_${seed}/stats_curvature.json"
            if [ -f "$json" ]; then
                seeds_done="${seeds_done}${seed},"
                count=$((count + 1))
            fi
        done
        # Only print if at least one seed exists
        if [ "$count" -gt 0 ]; then
            echo "SEED:${task_short}:${learner_short}:${count}:${seeds_done%,}"
        fi
    done
done

# --- Recent log errors (last 30 min of .err files) ---
echo ""
echo "SECTION:RECENT_ERRORS"
err_files=$(find "${BASE_DIR}/logs/" -name "*.err" -newer <(date -d '30 minutes ago' '+%Y%m%d%H%M' 2>/dev/null || echo "/dev/null") -size +0 2>/dev/null | head -5)
if [ -n "$err_files" ]; then
    for f in $err_files; do
        echo "ERRFILE:${f}"
        tail -3 "$f" 2>/dev/null
    done
else
    echo "NO_RECENT_ERRORS"
fi

echo ""
echo "SECTION:DONE"
