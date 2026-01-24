# Layer-Selective UPGD Implementation - Documentation Hub

**Status**: ✅ Implementation Complete (28/28 verification checks passed)
**Date**: January 2026
**Author**: Shin Lee (shin283@purdue.edu)

---

## 📁 Documentation Files in This Folder

### 1. **IMPLEMENTATION_SUMMARY.md** ⭐ START HERE
Quick reference guide with:
- What was implemented (summary)
- Quick start commands
- File checklist
- Verification status
- Next steps

**Use this for**: Quick overview and getting started

---

### 2. **LAYER_SELECTIVE_UPGD_IMPLEMENTATION.md** 📘 FULL DETAILS
Complete technical documentation (324 lines):
- Detailed implementation for all components
- Code examples and rationale
- ResNet18 architecture breakdown
- Expected results and hypotheses
- Verification steps
- Usage examples
- WandB integration

**Use this for**: In-depth understanding and troubleshooting

---

### 3. **SLURM_SCRIPTS_README.md** 🚀 CLUSTER GUIDE
SLURM scripts documentation:
- All 6 variants explained
- Individual vs array job usage
- Expected runtimes
- Log file locations
- WandB project details
- Cluster configuration

**Use this for**: Running experiments on Gautschi cluster

---

### 4. **verify_implementation.sh** ✓ VERIFICATION SCRIPT
Automated verification script (28 checks):
- Validates all files created
- Checks code modifications
- Verifies config file contents
- Tests script permissions

**Usage**:
```bash
cd /scratch/gautschi/shin283/upgd/docs
./verify_implementation.sh
```

**Expected output**: "✓ ALL CHECKS PASSED - Implementation is complete!"

---

### 5. **test_layer_selective_gating.py** 🧪 TEST SUITE
Comprehensive test script with 5 tests:
- Test 1: UPGD initialization with different gating modes
- Test 2: ResNet18 layer names
- Test 3: Gating logic for each mode
- Test 4: Parameter counts (gated vs non-gated)
- Test 5: Forward/backward pass with optimizer

**Usage**:
```bash
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop
cd /scratch/gautschi/shin283/upgd/docs
python3.8 test_layer_selective_gating.py
```

**Expected output**: All 5 tests PASS

---

## 🎯 Quick Start (3 Steps)

### Step 1: Verify Implementation
```bash
cd /scratch/gautschi/shin283/upgd/docs
./verify_implementation.sh
```

### Step 2: Run Tests (requires cluster/conda)
```bash
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop
python3.8 test_layer_selective_gating.py
```

### Step 3: Submit Experiments
```bash
cd /scratch/gautschi/shin283/upgd/upgd_aurel_scripts
sbatch slurm_incr_cifar_all_variants.sh
```

---

## 📊 What Was Implemented

### Core Changes
1. **UPGD Optimizer** (`/scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py`)
   - Added layer-selective gating: `'full'`, `'output_only'`, `'hidden_only'`
   - Added `non_gated_scale` parameter (default: 0.5)
   - Enhanced statistics tracking

2. **Experiment File** (`/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/incremental_cifar_experiment.py`)
   - Added `upgd_gating_mode` and `upgd_non_gated_scale` parameters
   - Added logging for gating statistics

### Configuration Files (6 total)
- `upgd_baseline.json` - Full gating
- `upgd_with_cbp.json` - Full + CBP
- `upgd_output_only.json` - Output-only gating ⭐ NEW
- `upgd_hidden_only.json` - Hidden-only gating ⭐ NEW
- `upgd_output_only_cbp.json` - Output + CBP ⭐ NEW
- `upgd_hidden_only_cbp.json` - Hidden + CBP ⭐ NEW

### SLURM Scripts (8 total)
Located in `/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/`:
- 6 individual variant scripts
- 1 array job script (runs all in parallel)
- 1 README with full documentation

---

## 🧬 Layer-Selective Gating Modes

### Full Gating (Baseline)
- **Gated**: All 11.2M parameters
- **Non-gated**: None
- **Use case**: Standard UPGD

### Output-Only Gating
- **Gated**: fc.* only (51K params, 0.5%)
- **Non-gated**: All conv/bn layers (11.1M, 99.5%)
- **Hypothesis**: High plasticity for new classes, stable features

### Hidden-Only Gating
- **Gated**: All conv/bn (11.1M, 99.5%)
- **Non-gated**: fc.* only (51K, 0.5%)
- **Hypothesis**: Adaptive features, stable output mapping

---

## 📈 Expected Runtime

- **Per experiment**: 4-6 days (4000 epochs × 20 tasks)
- **Array job**: 4-6 days total (6 variants run in parallel)
- **Requirements**: 1 GPU per job, Gautschi ai partition

---

## 🔍 WandB Tracking

- **Project**: `upgd-incremental-cifar`
- **Entity**: `shin283-purdue-university`
- **URL**: https://wandb.ai/shin283-purdue-university/upgd-incremental-cifar

---

## 📂 File Locations

### Documentation (this folder)
```
/scratch/gautschi/shin283/upgd/docs/
├── README.md (this file)
├── IMPLEMENTATION_SUMMARY.md
├── LAYER_SELECTIVE_UPGD_IMPLEMENTATION.md
├── SLURM_SCRIPTS_README.md
├── test_layer_selective_gating.py
└── verify_implementation.sh
```

### Code (modified files)
```
/scratch/gautschi/shin283/loss-of-plasticity/lop/
├── algos/upgd.py (modified)
└── incremental_cifar/
    ├── incremental_cifar_experiment.py (modified)
    └── cfg/
        ├── upgd_baseline.json (updated)
        ├── upgd_with_cbp.json (updated)
        ├── upgd_output_only.json (new)
        ├── upgd_hidden_only.json (new)
        ├── upgd_output_only_cbp.json (new)
        └── upgd_hidden_only_cbp.json (new)
```

### SLURM Scripts
```
/scratch/gautschi/shin283/upgd/upgd_aurel_scripts/
├── README.md
├── slurm_incr_cifar_sgd_baseline.sh
├── slurm_incr_cifar_upgd_full.sh
├── slurm_incr_cifar_upgd_output_only.sh
├── slurm_incr_cifar_upgd_hidden_only.sh
├── slurm_incr_cifar_upgd_output_only_cbp.sh
├── slurm_incr_cifar_upgd_hidden_only_cbp.sh
└── slurm_incr_cifar_all_variants.sh
```

---

## ✅ Verification Status

```
Total checks: 28
✓ Passed: 28
✗ Failed: 0

✓ ALL CHECKS PASSED - Implementation is complete!
```

---

## 🎓 Reading Order (Recommended)

1. **This file** (README.md) - Overview
2. **IMPLEMENTATION_SUMMARY.md** - Quick reference
3. **verify_implementation.sh** - Run verification
4. **test_layer_selective_gating.py** - Run tests
5. **SLURM_SCRIPTS_README.md** - Before submitting jobs
6. **LAYER_SELECTIVE_UPGD_IMPLEMENTATION.md** - Deep dive when needed

---

## 🆘 Support

If you encounter issues:
1. Run verification: `./verify_implementation.sh`
2. Check logs: `/scratch/gautschi/shin283/upgd/logs/*.out`
3. Verify environment: `conda list | grep torch`
4. Review documentation in this folder

---

## 📞 Contact

- **Author**: Shin Lee
- **Email**: shin283@purdue.edu
- **Institution**: Purdue University
- **Location**: `/scratch/gautschi/shin283/upgd/docs/`

---

**Last Updated**: January 24, 2026
**Implementation Status**: ✅ READY FOR EXPERIMENTS
