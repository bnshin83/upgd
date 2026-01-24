#!/bin/bash
# Verification script for Layer-Selective UPGD implementation
# Run this to check that all files were created correctly

echo "============================================================"
echo "Layer-Selective UPGD Implementation Verification"
echo "============================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

# Function to check file exists
check_file() {
    local file=$1
    local description=$2

    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        echo "  Missing: $file"
        ((FAILED++))
        return 1
    fi
}

# Function to check directory exists
check_dir() {
    local dir=$1
    local description=$2

    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        echo "  Missing: $dir"
        ((FAILED++))
        return 1
    fi
}

echo "1. Modified Files"
echo "-----------------------------------------------------------"
check_file "/scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py" \
    "UPGD optimizer with layer-selective gating"
check_file "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/incremental_cifar_experiment.py" \
    "Incremental CIFAR experiment with gating support"
echo ""

echo "2. Configuration Files"
echo "-----------------------------------------------------------"
check_file "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_baseline.json" \
    "UPGD baseline config (full gating)"
check_file "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_with_cbp.json" \
    "UPGD with CBP config (full gating)"
check_file "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_output_only.json" \
    "UPGD output-only gating config"
check_file "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_hidden_only.json" \
    "UPGD hidden-only gating config"
check_file "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_output_only_cbp.json" \
    "UPGD output-only + CBP config"
check_file "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_hidden_only_cbp.json" \
    "UPGD hidden-only + CBP config"
echo ""

echo "3. SLURM Scripts Directory"
echo "-----------------------------------------------------------"
check_dir "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts" \
    "SLURM scripts directory"
check_file "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/README.md" \
    "SLURM scripts README"
check_file "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_sgd_baseline.sh" \
    "SGD baseline script"
check_file "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_full.sh" \
    "UPGD full gating script"
check_file "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_output_only.sh" \
    "UPGD output-only script"
check_file "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_hidden_only.sh" \
    "UPGD hidden-only script"
check_file "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_output_only_cbp.sh" \
    "UPGD output-only + CBP script"
check_file "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_upgd_hidden_only_cbp.sh" \
    "UPGD hidden-only + CBP script"
check_file "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_all_variants.sh" \
    "Array job for all variants"
echo ""

echo "4. Test and Documentation"
echo "-----------------------------------------------------------"
check_file "/scratch/gautschi/shin283/upgd/test_layer_selective_gating.py" \
    "Layer-selective gating test script"
check_file "/scratch/gautschi/shin283/upgd/LAYER_SELECTIVE_UPGD_IMPLEMENTATION.md" \
    "Implementation documentation"
check_file "/scratch/gautschi/shin283/upgd/verify_implementation.sh" \
    "This verification script"
echo ""

echo "5. Script Permissions"
echo "-----------------------------------------------------------"
if [ -x "/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/slurm_incr_cifar_all_variants.sh" ]; then
    echo -e "${GREEN}✓${NC} SLURM scripts are executable"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} SLURM scripts are not executable"
    echo "  Run: chmod +x /scratch/gautschi/shin283/upgd/upgd_aurel_scripts/*.sh"
    ((FAILED++))
fi

if [ -x "/scratch/gautschi/shin283/upgd/test_layer_selective_gating.py" ]; then
    echo -e "${GREEN}✓${NC} Test script is executable"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Test script is not executable"
    echo "  Run: chmod +x /scratch/gautschi/shin283/upgd/test_layer_selective_gating.py"
    ((FAILED++))
fi
echo ""

echo "6. Configuration File Contents"
echo "-----------------------------------------------------------"

# Check if gating_mode is present in config files
if grep -q "upgd_gating_mode" "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_baseline.json" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} upgd_baseline.json contains gating_mode parameter"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} upgd_baseline.json missing gating_mode parameter"
    ((FAILED++))
fi

if grep -q '"output_only"' "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_output_only.json" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} upgd_output_only.json contains correct gating mode"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} upgd_output_only.json missing or incorrect gating mode"
    ((FAILED++))
fi

if grep -q '"hidden_only"' "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_hidden_only.json" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} upgd_hidden_only.json contains correct gating mode"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} upgd_hidden_only.json missing or incorrect gating mode"
    ((FAILED++))
fi
echo ""

echo "7. Code Modifications"
echo "-----------------------------------------------------------"

# Check if UPGD optimizer has gating_mode parameter
if grep -q "gating_mode" "/scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} UPGD optimizer has gating_mode parameter"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} UPGD optimizer missing gating_mode parameter"
    ((FAILED++))
fi

# Check if UPGD has _should_apply_gating method
if grep -q "_should_apply_gating" "/scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} UPGD optimizer has _should_apply_gating method"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} UPGD optimizer missing _should_apply_gating method"
    ((FAILED++))
fi

# Check if experiment file has upgd_gating_mode parameter
if grep -q "upgd_gating_mode" "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/incremental_cifar_experiment.py" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Experiment file has upgd_gating_mode parameter"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Experiment file missing upgd_gating_mode parameter"
    ((FAILED++))
fi
echo ""

echo "============================================================"
echo "Summary"
echo "============================================================"
TOTAL=$((PASSED + FAILED))
echo "Total checks: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo -e "${GREEN}Failed: $FAILED${NC}"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED - Implementation is complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run test script:"
    echo "   source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh"
    echo "   conda activate /scratch/gautschi/shin283/conda_envs/lop"
    echo "   python3.8 /scratch/gautschi/shin283/upgd/test_layer_selective_gating.py"
    echo ""
    echo "2. Submit experiments:"
    echo "   cd /scratch/gautschi/shin283/upgd/upgd_aurel_scripts"
    echo "   sbatch slurm_incr_cifar_all_variants.sh"
    exit 0
else
    echo -e "${RED}✗ SOME CHECKS FAILED - Please review errors above${NC}"
    exit 1
fi
