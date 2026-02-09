#!/bin/bash
# Monitoring script for Humanoid-v4 RL experiments
# Usage: ./monitor_humanoid.sh [job_id]

set -e

LC_BIN="$HOME/projects/localcontrol/bin"
CLUSTER="gautschi"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Humanoid-v4 Experiment Monitor${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check job status
echo -e "${GREEN}[1] Job Status on Gautschi:${NC}"
PATH="$LC_BIN:$PATH" lc-status "$CLUSTER"
echo ""

# If job ID provided, show detailed info
if [ -n "$1" ]; then
    JOB_ID="$1"
    echo -e "${GREEN}[2] Detailed Status for Job $JOB_ID:${NC}"
    ssh gautschi "sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode -X"
    echo ""

    echo -e "${GREEN}[3] Latest Log Output (last 30 lines):${NC}"
    PATH="$LC_BIN:$PATH" lc-logs "$CLUSTER" "$JOB_ID" 2>/dev/null | tail -30 || echo "  No logs yet or job not found"
    echo ""
fi

# Check WandB
echo -e "${GREEN}[4] WandB Dashboard:${NC}"
echo -e "  ${BLUE}https://wandb.ai/shin283-purdue-university/upgd-rl${NC}"
echo -e "  Filter: ${YELLOW}*humanoid* OR *hum_*${NC}"
echo ""

# Show useful commands
echo -e "${YELLOW}Useful Commands:${NC}"
echo "  View logs:       PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-logs gautschi [job_id]"
echo "  Follow logs:     PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-logs gautschi [job_id] --follow"
echo "  View errors:     PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-logs gautschi [job_id] --err"
echo "  Live monitoring: PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-status gautschi --watch"
echo ""

# Show test job info
echo -e "${YELLOW}Current Test Job:${NC}"
echo "  Job ID: 7608913 (with race condition fix)"
echo "  Tasks: 2 (upgd_full seeds 0-1)"
echo "  Expected runtime: ~30-60 minutes"
echo ""

# Show what's next
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Wait for test job to complete (~1 hour)"
echo "  2. Verify both tasks succeeded"
echo "  3. If test passes, submit full job:"
echo "     cd /Users/boonam/projects/upgd"
echo "     PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-submit --cluster gautschi --exp rl/humanoid_completion --sync"
echo ""
