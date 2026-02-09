#!/bin/bash
# Dual-cluster monitoring for Humanoid-v4 RL experiments
# Usage: ./monitor_humanoid_dual.sh [gautschi_job_id] [gilbreth_job_id]

set -e

LC_BIN="$HOME/projects/localcontrol/bin"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Humanoid-v4 Dual-Cluster Monitor${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Gautschi status
echo -e "${GREEN}[1] GAUTSCHI Status (H100, 8 concurrent):${NC}"
echo -e "${CYAN}    Methods: upgd_full, upgd_output_only (40 tasks)${NC}"
PATH="$LC_BIN:$PATH" lc-status gautschi
echo ""

# Gilbreth status
echo -e "${GREEN}[2] GILBRETH Status (A100, 6 concurrent):${NC}"
echo -e "${CYAN}    Methods: upgd_hidden_only, adam (40 tasks)${NC}"
PATH="$LC_BIN:$PATH" lc-status gilbreth
echo ""

# Detailed status if job IDs provided
if [ -n "$1" ]; then
    GAUTSCHI_JOB="$1"
    echo -e "${GREEN}[3] Gautschi Job $GAUTSCHI_JOB Details:${NC}"
    ssh gautschi "sacct -j $GAUTSCHI_JOB --format=JobID%-15,JobName,State,Elapsed,ExitCode -X | head -20"
    echo ""
fi

if [ -n "$2" ]; then
    GILBRETH_JOB="$2"
    echo -e "${GREEN}[4] Gilbreth Job $GILBRETH_JOB Details:${NC}"
    ssh gilbreth "sacct -j $GILBRETH_JOB --format=JobID%-15,JobName,State,Elapsed,ExitCode -X | head -20"
    echo ""
fi

# Progress summary
if [ -n "$1" ] && [ -n "$2" ]; then
    echo -e "${GREEN}[5] Combined Progress:${NC}"
    GAUTSCHI_DONE=$(ssh gautschi "sacct -j $1 --format=State -X | grep COMPLETED | wc -l" 2>/dev/null || echo "0")
    GILBRETH_DONE=$(ssh gilbreth "sacct -j $2 --format=State -X | grep COMPLETED | wc -l" 2>/dev/null || echo "0")
    TOTAL_DONE=$((GAUTSCHI_DONE + GILBRETH_DONE))
    echo -e "  Gautschi: ${CYAN}${GAUTSCHI_DONE}/40${NC} tasks complete"
    echo -e "  Gilbreth: ${CYAN}${GILBRETH_DONE}/40${NC} tasks complete"
    echo -e "  ${YELLOW}Total: ${TOTAL_DONE}/80 tasks complete ($(( TOTAL_DONE * 100 / 80 ))%)${NC}"
    echo ""
fi

# WandB
echo -e "${GREEN}[6] WandB Dashboard:${NC}"
echo -e "  ${BLUE}https://wandb.ai/shin283-purdue-university/upgd-rl${NC}"
echo -e "  Filter: ${YELLOW}*humanoid* OR *hum_*${NC}"
echo ""

# Useful commands
echo -e "${YELLOW}Useful Commands:${NC}"
echo "  Gautschi logs:   PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-logs gautschi [job_id]"
echo "  Gilbreth logs:   PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-logs gilbreth [job_id]"
echo "  Live monitoring: watch -n 60 ./monitor_humanoid_dual.sh [g_job] [gb_job]"
echo ""

# Current test
echo -e "${YELLOW}Current Test Job (Gautschi):${NC}"
echo "  Job ID: 7608913"
echo "  Status: Should be completing soon (~30 min runtime)"
echo ""

# Next steps
echo -e "${YELLOW}Next Steps (After Test Passes):${NC}"
echo "  ${GREEN}1. Submit to GAUTSCHI:${NC}"
echo "     cd /Users/boonam/projects/upgd"
echo "     PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-submit --cluster gautschi --exp rl/humanoid_gautschi --sync"
echo ""
echo "  ${GREEN}2. Submit to GILBRETH (after SGD jobs finish ~12:30 PM):${NC}"
echo "     cd /Users/boonam/projects/upgd"
echo "     PATH=\"\$HOME/projects/localcontrol/bin:\$PATH\" lc-submit --cluster gilbreth --exp rl/humanoid_gilbreth --sync"
echo ""
