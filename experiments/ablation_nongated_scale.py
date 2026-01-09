"""
Ablation study for non-gated layer scaling in Output-Only UPGD.

This experiment tests different scaling factors for hidden layers when only
the output layer receives utility-based gating:
  - scale=0.0:  Hidden layers frozen (no updates)
  - scale=0.27: Hidden layers at max protection level (1 - sigmoid(1))
  - scale=0.5:  Hidden layers at neutral level (1 - sigmoid(0)) [default]
  - scale=0.73: Hidden layers at min protection level (sigmoid(1))
  - scale=1.0:  Hidden layers at full SGD (no scaling)

The hypothesis is that different tasks may benefit from different levels of
hidden layer plasticity.
"""

from core.grid_search import GridSearch
from core.learner.weight_upgd import (
    UPGDOutputOnlyScale0Learner,
    UPGDOutputOnlyScale027Learner,
    UPGDOutputOnlyScale05Learner,
    UPGDOutputOnlyScale073Learner,
    UPGDOutputOnlyScale1Learner,
    UPGDLayerSelectiveOutputOnlyLearner,  # Original for comparison
)
from core.learner.sgd import SGDLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.network.fcn_relu import FullyConnectedReLU
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

# Choose task: can be changed to other label-permuted tasks
exp_name = "label_permuted_emnist"
task = tasks[exp_name]()
n_steps = 1000000
n_seeds = 5  # Fewer seeds for ablation

# Grid for UPGD output-only variants
upgd_grid = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[0.01],  # Use best LR from main experiments
    beta_utility=[0.9, 0.999],
    sigma=[0.001, 0.01],
    weight_decay=[0.0],
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

# Baseline grids
sgd_grid = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[0.01],
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

sp_grid = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[0.01],
    sigma=[0.01],
    decay=[0.001],
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

# All learners to test
learners = [
    # Scale ablation variants
    UPGDOutputOnlyScale0Learner(),    # scale=0.0 (frozen)
    UPGDOutputOnlyScale027Learner(),  # scale=0.27 (max protection)
    UPGDOutputOnlyScale05Learner(),   # scale=0.5 (neutral) [default]
    UPGDOutputOnlyScale073Learner(),  # scale=0.73 (min protection)
    UPGDOutputOnlyScale1Learner(),    # scale=1.0 (full SGD)
    # Baselines
    SGDLearner(),
    ShrinkandPerturbLearner(),
]

grids = [upgd_grid] * 5 + [sgd_grid, sp_grid]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, f"{exp_name}_scale_ablation", learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}_scale_ablation", f"{exp_name}_scale_ablation")
    create_script_runner(f"generated_cmds/{exp_name}_scale_ablation")
