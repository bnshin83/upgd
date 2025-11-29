from core.grid_search import GridSearch
from core.learner.weight_upgd import FirstOrderGlobalUPGDLearner, FirstOrderNonprotectingGlobalUPGDLearner
from core.learner.input_aware_upgd import InputAwareFirstOrderGlobalUPGDLearner, InputAwareSecondOrderGlobalUPGDLearner
from core.learner.sgd import SGDLearner
from core.learner.pgd import PGDLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.network.fcn_relu import FullyConnectedReLU
from core.runner import Runner
from core.run.input_aware_run import InputAwareRun
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "label_permuted_emnist"
task = tasks[exp_name]()
n_steps = 1000000
n_seeds = 20

# Standard UPGD grids for comparison
up_grids = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[10 ** -i for i in range(2, 6)],
    beta_utility=[0.99, 0.999],
    sigma=[0.01, 0.1, 1.0],
    weight_decay=[0.0, 0.1, 0.01, 0.001, 0.0001],
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

# Input-aware first-order UPGD grids
input_aware_fo_grids = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[10 ** -i for i in range(2, 6)],
    beta_utility=[0.99, 0.999],
    sigma=[0.01, 0.1, 1.0],
    weight_decay=[0.0, 0.01, 0.001],
    # Input-aware specific parameters
    curvature_threshold=[0.1, 1.0, 10.0],  # Threshold for considering input as high curvature
    lambda_max=[0.1, 0.5, 1.0],  # Maximum regularization strength
    lambda_scale=[0.1, 1.0],  # Scaling factor for curvature-to-lambda mapping
    beta_curvature=[0.9, 0.99],  # Momentum for curvature tracking
    hutchinson_samples=[1, 3],  # Number of samples for Hutchinson estimator
    compute_curvature_every=[1, 10],  # Compute curvature every N steps
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

# Input-aware second-order UPGD grids (with HesScale)
input_aware_so_grids = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[10 ** -i for i in range(2, 6)],
    beta_utility=[0.99, 0.999],
    sigma=[0.01, 0.1, 1.0],
    weight_decay=[0.0, 0.01, 0.001],
    # Input-aware specific parameters
    curvature_threshold=[0.1, 1.0, 10.0],
    lambda_max=[0.1, 0.5, 1.0],
    lambda_scale=[0.1, 1.0],
    beta_curvature=[0.9, 0.99],
    hutchinson_samples=[1, 3],
    compute_curvature_every=[1, 10],
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

# Baseline methods
pgd_grids = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[10 ** -i for i in range(2, 6)],
    sigma=[0.005, 0.05, 0.5],
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

sgd_grid = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[10 ** -i for i in range(2, 6)],
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

sp_grid = GridSearch(
    seed=[i for i in range(0, n_seeds)],
    lr=[10 ** -i for i in range(2, 6)],
    sigma=[0.005, 0.05, 0.5],
    decay=[0.1, 0.01, 0.001, 0.0001],
    network=[FullyConnectedReLU()],
    n_samples=[n_steps],
)

# Combine all grids
grids = [
    up_grids, up_grids,  # Standard UPGD methods
    input_aware_fo_grids, input_aware_so_grids,  # Input-aware methods
    sgd_grid, pgd_grids, sp_grid  # Baseline methods
]

learners = [
    FirstOrderGlobalUPGDLearner(),
    FirstOrderNonprotectingGlobalUPGDLearner(),
    InputAwareFirstOrderGlobalUPGDLearner(),  # New input-aware learners
    InputAwareSecondOrderGlobalUPGDLearner(),
    SGDLearner(),
    PGDLearner(),
    ShrinkandPerturbLearner(),
]

# Generate command scripts for each learner
for learner, grid in zip(learners, grids):
    # Use InputAwareRun for input-aware learners, regular Run for others
    if isinstance(learner, (InputAwareFirstOrderGlobalUPGDLearner, InputAwareSecondOrderGlobalUPGDLearner)):
        run_class = InputAwareRun
    else:
        from core.run.run import Run
        run_class = Run
    
    runner = Runner(run_class, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds_input_aware")
    create_script_generator(f"generated_cmds_input_aware/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds_input_aware/{exp_name}")

print(f"Generated experiment scripts for {exp_name} with input-aware UPGD methods")
print(f"Total configurations:")
for learner, grid in zip(learners, grids):
    print(f"  {learner.name}: {len(list(grid))} configurations")
print(f"\nScripts saved to: generated_cmds_input_aware/{exp_name}/")
print(f"\nTo run experiments:")
print(f"  1. cd generated_cmds_input_aware/{exp_name}/")
print(f"  2. ./run_{exp_name}.sh  # Run all experiments")
print(f"  3. Or run specific learner: ./generate_{exp_name}_{learner.name}.sh")
