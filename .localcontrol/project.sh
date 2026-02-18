PROJECT_NAME="upgd"
# Remote dir: ${SCRATCH}/${PROJECT_NAME}  (e.g., /scratch/gautschi/shin283/upgd)

# Exclude when pushing code TO cluster
RSYNC_EXCLUDE=(
    ".git" "__pycache__" "*.pyc" "*.pyo"
    "*.pt" "*.pth" "*.ckpt"
    "wandb" "logs" "data" ".DS_Store"
    "plots_all_stats" "pgd_plots" "upgd_plots"
    "slurm_status_logs" "results"
    ".upgd" "runs_upgd"
)

# Pull config: what to pull FROM cluster
PULL_REMOTE_DIR="logs"
PULL_LOCAL_DIR="results"
PULL_INCLUDE=("*.json" "*.csv" "*.out" "*.err" "*.txt" "*.log")
PULL_EXCLUDE=("*.pt" "*.pth" "*.ckpt")

# Archive destinations
ARCHIVE_DEPOT="/depot/jhaddock/data/shin283/${PROJECT_NAME}"
ARCHIVE_NAS=""
