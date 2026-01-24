 Implementation Plan: UPGD for Incremental CIFAR-100

 Overview

 Implement UPGD (Utility-based Perturbed Gradient Descent) optimizer for the incremental CIFAR-100 continual learning benchmark. This involves creating a UPGD algorithm wrapper compatible with the
 loss-of-plasticity framework and integrating it into the existing experiment infrastructure.

 Context

 - RL Implementation (Phase 2): UPGD variants for RL are already implemented in /scratch/gautschi/shin283/upgd/core/run/rl/
   - adaupgd.py: Adaptive UPGD with Adam-style moments
   - rl_upgd_layerselective.py: Layer-selective gating for actor-critic networks
   - plasticity_metrics.py: Dead neurons, stable rank, weight statistics
   - run_ppo_upgd.py: Unified PPO script supporting multiple optimizer variants
 - Incremental CIFAR-100 Setup: Already exists at /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/
   - Uses ResNet18 architecture
   - Current optimizers: SGD (via incremental_cifar_experiment.py)
   - Algorithm wrappers in /scratch/gautschi/shin283/loss-of-plasticity/lop/algos/ (bp.py, cbp.py, res_gnt.py)
   - Configuration-based experiment management

 Task Breakdown

 1. Create UPGD Algorithm Wrapper

 File: /scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py

 Requirements:
 - Create a wrapper class similar to Backprop class in bp.py
 - Support both basic UPGD and adaptive UPGD (with Adam moments)
 - Include utility tracking and histogram logging
 - Compatible with ResNet18 architecture

 Implementation approach:
 - Adapt AdaptiveUPGD from /scratch/gautschi/shin283/upgd/core/run/rl/adaupgd.py
 - Follow the interface pattern from lop/algos/bp.py:
   - Constructor takes net, step_size, loss, opt, weight_decay, device, momentum
   - learn(x, target) method for training step
   - Return loss and optionally output
 - Add get_utility_stats() method for logging utility metrics
 - Support for noise perturbation (shrink-and-perturb style)

 Key differences from RL implementation:
 - Use named_parameters() to track parameter names
 - Integrate with ResNet architecture (Conv2d + Linear layers)
 - Match the loss-of-plasticity framework's API

 2. Modify Incremental CIFAR Experiment

 File: /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/incremental_cifar_experiment.py

 Changes needed:
 1. Add UPGD optimizer selection logic (around line 106-108 where optimizer is created)
 2. Add config parameters for UPGD:
   - use_upgd: boolean flag
   - upgd_beta_utility: utility decay rate (default 0.999)
   - upgd_sigma: noise scale (default 0.001)
   - upgd_beta1, upgd_beta2: Adam moments (defaults 0.9, 0.999)
   - upgd_eps: epsilon for numerical stability (default 1e-5)
 3. Conditional optimizer initialization:
 if self.use_upgd:
     from lop.algos.upgd import UPGD
     self.upgd_wrapper = UPGD(
         net=self.net,
         step_size=self.stepsize,
         loss='nll',
         weight_decay=self.weight_decay,
         beta_utility=self.upgd_beta_utility,
         sigma=self.upgd_sigma,
         device=self.device
     )
 else:
     self.optim = torch.optim.SGD(...)

 4. Modify training loop to use UPGD wrapper when enabled
 5. Add utility logging to results dict (histograms, norms, global max)

 3. Create Configuration Files

 Location: /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/

 Create two new config files:

 File 1: upgd_baseline.json
 {
   "_model_description_": "UPGD optimizer baseline",
   "data_path": "",
   "results_dir": "",
   "experiment_name": "upgd_baseline",
   "num_workers": 12,

   "stepsize": 0.1,
   "weight_decay": 0.0005,
   "momentum": 0.9,

   "noise_std": 0.0,

   "use_cbp": false,
   "use_upgd": true,
   "upgd_beta_utility": 0.999,
   "upgd_sigma": 0.001,
   "upgd_beta1": 0.9,
   "upgd_beta2": 0.999,
   "upgd_eps": 1e-5,

   "reset_head": false,
   "reset_network": false,
   "early_stopping": true
 }

 File 2: upgd_with_cbp.json
 - Same as above but with use_cbp: true and CBP parameters

 4. Create SLURM Scripts

 Location: /scratch/gautschi/shin283/upgd/slurm_runs/

 Create scripts following the pattern of slurm_rl_ant_upgd.sh:

 File 1: slurm_incremental_cifar_upgd.sh
 - Run UPGD baseline on incremental CIFAR-100
 - 4000 epochs, seed 0 for initial test
 - Point to correct config file
 - Set WandB project name: "upgd-incremental-cifar"

 File 2: slurm_incremental_cifar_sgd_baseline.sh
 - Run SGD baseline for comparison
 - Same hyperparameters except optimizer

 File 3: slurm_incremental_cifar_upgd_variants.sh
 - Array job for multiple seeds or hyperparameter sweeps
 - Test different beta_utility values (0.99, 0.999, 0.9999)
 - Test different sigma values (0.0001, 0.001, 0.01)

 5. Integration Testing Plan

 Before full runs, verify:
 1. Unit test: UPGD wrapper can process a single batch
   - Create minimal test with random data
   - Check utility computation is non-zero
   - Verify gradient updates work
 2. Short run test: Run 10 epochs on incremental CIFAR
   - Check logging works (utility histograms, loss, accuracy)
   - Verify checkpoint saving/loading
   - Ensure no NaN/Inf values
 3. Baseline comparison: Run 200 epochs each
   - UPGD vs SGD vs Adam
   - Compare training curves
   - Check utility distributions

 Critical Files Summary

 New files to create:
 1. /scratch/gautschi/shin283/loss-of-plasticity/lop/algos/upgd.py
 2. /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_baseline.json
 3. /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_with_cbp.json
 4. /scratch/gautschi/shin283/upgd/slurm_runs/slurm_incremental_cifar_upgd.sh
 5. /scratch/gautschi/shin283/upgd/slurm_runs/slurm_incremental_cifar_sgd_baseline.sh

 Files to modify:
 1. /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/incremental_cifar_experiment.py
   - Lines ~64-73: Add UPGD config parameters
   - Lines ~106-108: Add conditional UPGD optimizer initialization
   - Training loop: Add UPGD wrapper call when enabled
   - Logging: Add utility statistics to results dict

 Verification Steps

 1. Code verification:
   - Run Python import test: python -c "from lop.algos.upgd import UPGD"
   - Check config loading: python -c "import json; print(json.load(open('cfg/upgd_baseline.json')))"
 2. Dry run:
   - Run 1 epoch with UPGD on small batch
   - Verify utility histograms are populated
   - Check all metrics are logged
 3. Full experiment:
   - Submit SLURM job for UPGD baseline
   - Submit SLURM job for SGD baseline
   - Monitor WandB for curves
   - Compare final test accuracy after 4000 epochs
 4. Analysis:
   - Plot utility distributions over time
   - Compare learning curves (UPGD vs SGD vs CBP)
   - Analyze loss of plasticity metrics
   - Check for catastrophic forgetting

 Expected Outcomes

 1. UPGD should maintain plasticity better than SGD baseline
 2. Utility histograms should show active gating (not all 0 or all 1)
 3. Learning curves should be stable (no divergence)
 4. Final test accuracy comparable or better than SGD
 5. Lower catastrophic forgetting on earlier tasks

 Notes

 - The RL implementation uses Haiku model for efficiency, but incremental CIFAR uses full Sonnet
 - UPGD may require learning rate tuning (start with SGD's 0.1, may need to adjust)
 - Weight decay interaction with utility gating needs monitoring
 - Consider adding plasticity metrics from plasticity_metrics.py to incremental CIFAR