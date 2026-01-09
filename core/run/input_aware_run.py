import torch
import sys
import os
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
from core.network.gate import GateLayer, GateLayerGrad
from core.learner.input_aware_upgd import (
    InputAwareFirstOrderGlobalUPGDLearner,
    InputAwareSecondOrderGlobalUPGDLearner
)
import signal
import traceback
import time
from functools import partial


def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'slurm_status_logs/timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)


class InputAwareRun:
    """
    Modified Run class that handles input-aware optimization with curvature computation.
    """
    name = 'input_aware_run'
    
    def __init__(self, n_samples=10000, task=None, learner=None, save_path="logs", 
                 seed=0, network=None, compute_curvature_every=1, **kwargs):
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = tasks[task]()
        self.task_name = task
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)
        self.compute_curvature_every = int(compute_curvature_every)
        
        # Check if learner is input-aware
        self.is_input_aware = isinstance(self.learner, (
            InputAwareFirstOrderGlobalUPGDLearner,
            InputAwareSecondOrderGlobalUPGDLearner
        ))

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_step_size = []
        curvatures_per_step = []  # Track input curvatures

        if self.task.criterion == 'cross_entropy':
            accuracy_per_step_size = []
            
        self.learner.set_task(self.task)
        
        if self.learner.extend:    
            extension = HesScale()
            extension.set_module_extension(GateLayer, GateLayerGrad())
            
        criterion = extend(criterions[self.task.criterion]()) if self.learner.extend else criterions[self.task.criterion]()
        
        optimizer = self.learner.optimizer(
            self.learner.parameters, **self.learner.optim_kwargs
        )
        
        # Store optimizer instance in learner for curvature updates
        if self.is_input_aware:
            self.learner.optimizer_instance = optimizer

        for step in range(self.n_samples):
            input, target = next(self.task)
            input, target = input.to(self.device), target.to(self.device)
            
            # Enable gradient computation for input if using input-aware learner
            if self.is_input_aware and step % self.compute_curvature_every == 0:
                input.requires_grad_(True)
            
            optimizer.zero_grad()
            output = self.learner.predict(input)
            loss = criterion(output, target)
            
            # Compute and update input curvature for input-aware learners
            if self.is_input_aware and step % self.compute_curvature_every == 0:
                with torch.enable_grad():
                    # Compute input curvature before backward pass
                    curvature = self.learner.compute_input_curvature(
                        self.learner.network, input, target, criterion
                    )
                    self.learner.update_optimizer_curvature(curvature)
                    curvatures_per_step.append(curvature)
                    
                    # Detach input to avoid double backward
                    input = input.detach()
                    
                    # Recompute forward pass with detached input
                    optimizer.zero_grad()
                    output = self.learner.predict(input)
                    loss = criterion(output, target)
            
            # Standard backward pass
            if self.learner.extend:
                with backpack(extension):
                    loss.backward()
            else:
                loss.backward()
                
            optimizer.step()
            losses_per_step_size.append(loss.item())
            
            if self.task.criterion == 'cross_entropy':
                accuracy_per_step_size.append((output.argmax(dim=1) == target).float().mean().item())

        # Log results
        log_data = {
            'losses': losses_per_step_size,
            'task': self.task_name,
            'learner': self.learner.name,
            'network': self.learner.network.name,
            'optimizer_hps': self.learner.optim_kwargs,
            'n_samples': self.n_samples,
            'seed': self.seed,
        }
        
        if self.task.criterion == 'cross_entropy':
            log_data['accuracies'] = accuracy_per_step_size
            
        if self.is_input_aware and curvatures_per_step:
            log_data['input_curvatures'] = curvatures_per_step
            log_data['avg_input_curvature'] = sum(curvatures_per_step) / len(curvatures_per_step)
            
        self.logger.log(**log_data)


if __name__ == "__main__":
    # Start the run using command line arguments
    ll = sys.argv[1:]
    args = {k[2:]: v for k, v in zip(ll[::2], ll[1::2])}
    run = InputAwareRun(**args)
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    
    try:
        run.start()
        with open(f"slurm_status_logs/finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        with open(f"slurm_status_logs/failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"slurm_status_logs/failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")
