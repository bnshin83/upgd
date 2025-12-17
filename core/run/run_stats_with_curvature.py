import torch, sys, os
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())

# HesScale is optional - only needed for second-order methods
try:
    from HesScale.hesscale import HesScale
    HESSCALE_AVAILABLE = True
except ImportError:
    HesScale = None
    HESSCALE_AVAILABLE = False

# GateLayer imports may also depend on HesScale
try:
    from core.network.gate import GateLayer, GateLayerGrad
except ImportError:
    GateLayer = None
    GateLayerGrad = None
from core.optim.weight_upgd.input_aware import compute_input_curvature_finite_diff
import signal
import traceback
import time
from functools import partial
import numpy as np  # Move numpy import to top

import wandb

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

class RunStatsWithCurvature:
    name = 'run_stats_with_curvature'
    def __init__(self, n_samples=10000, task=None, learner=None, save_path="logs", seed=0, network=None, 
                 compute_curvature_every=1, disable_regularization=False, disable_gating=False, **kwargs):
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = tasks[task]()
        self.task_name = task

        # Check if this is an input-aware learner or curvature-gating learner before adding disable flags
        is_input_aware_learner = 'input_aware' in learner or 'curvature_gating' in learner

        # Add disable flags to kwargs only for input-aware learners
        if is_input_aware_learner:
            kwargs.update({
                'disable_regularization': disable_regularization,
                'disable_gating': disable_gating
            })

        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)
        self.compute_curvature_every = int(compute_curvature_every)

        # Use the previously computed input-aware check
        self.is_input_aware = is_input_aware_learner
        
        # Initialize wandb if environment variables are set
        self.wandb_enabled = False
        if (
            os.environ.get('WANDB_PROJECT')
            and os.environ.get('WANDB_MODE', 'online') != 'disabled'
        ):
            try:
                wandb.init(
                    project=os.environ.get('WANDB_PROJECT', 'upgd-experiments'),
                    entity=os.environ.get('WANDB_ENTITY'),  # Set entity to avoid team conflict
                    name=os.environ.get('WANDB_RUN_NAME', f'{task}_{learner}_{seed}'),
                    config={
                        'task': task,
                        'learner': learner, 
                        'network': network,
                        'n_samples': n_samples,
                        'seed': seed,
                        'compute_curvature_every': compute_curvature_every,
                        'is_input_aware': self.is_input_aware,
                        'disable_regularization': disable_regularization,
                        'disable_gating': disable_gating,
                        **kwargs
                    },
                    settings=wandb.Settings(
                        console='off',         # Reduce console output
                    )
                )
                self.wandb_enabled = True
                print(f"WandB initialized: {wandb.run.name}")
                # Note: Removed initial seed log to avoid wandb sync issues
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                self.wandb_enabled = False
        
    def start(self):
        torch.manual_seed(self.seed)
        losses_per_task = []
        plasticity_per_task = []
        n_dead_units_per_task = []
        weight_rank_per_task = []
        weight_l2_per_task = []
        weight_l1_per_task = []
        grad_l2_per_task = []
        grad_l1_per_task = []
        grad_l0_per_task = []
        
        # Curvature tracking lists (only for input-aware learners)
        if self.is_input_aware:
            input_curvature_per_task = []
            lambda_values_per_task = []
            avg_curvature_per_task = []
            # Store curvature statistics per task
            curvature_max_per_task = []
            curvature_min_per_task = []
            curvature_std_per_task = []

        if self.task.criterion == 'cross_entropy':
            accuracy_per_task = []
        self.learner.set_task(self.task)
        if self.learner.extend:    
            extension = HesScale()
            extension.set_module_extension(GateLayer, GateLayerGrad())
        criterion = extend(criterions[self.task.criterion]()) if self.learner.extend else criterions[self.task.criterion]()
        optimizer = self.learner.optimizer(
            self.learner.parameters, **self.learner.optim_kwargs
        )

        # Ensure input-aware learners can update optimizer curvature (for lambda computation)
        try:
            if self.is_input_aware:
                setattr(self.learner, 'optimizer_instance', optimizer)
        except Exception:
            pass

        # Initialize log directory early for visibility during long runs
        try:
            self.logger.initialize_log_path(
                task=self.task_name,
                learner=self.learner.name,
                network=self.learner.network.name,
                optimizer_hps=self.learner.optim_kwargs,
                n_samples=self.n_samples,
                seed=self.seed,
            )
        except Exception as e:
            print(f"Warning: could not initialize log path early: {e}")

        # Step-level tracking for ALL steps (saved to JSON, never reset)
        all_losses_per_step = []
        all_plasticity_per_step = []
        all_n_dead_units_per_step = []
        all_weight_rank_per_step = []
        all_weight_l2_per_step = []
        all_weight_l1_per_step = []
        all_grad_l2_per_step = []
        all_grad_l1_per_step = []
        all_grad_l0_per_step = []

        # Utility histogram tracking (9 bins) - logged every 10 steps
        all_utility_hist_per_step = {
            'steps': [],  # Which steps have utility data
            'hist_0_20_pct': [],
            'hist_20_40_pct': [],
            'hist_40_44_pct': [],
            'hist_44_48_pct': [],
            'hist_48_52_pct': [],
            'hist_52_56_pct': [],
            'hist_56_60_pct': [],
            'hist_60_80_pct': [],
            'hist_80_100_pct': [],
            'global_max': [],
            'total_params': None,  # Will be set once from optimizer
        }
        # Per-layer utility histograms
        all_layer_utility_hist_per_step = {
            'linear_1': {'steps': [], 'hist_48_52_pct': [], 'hist_52_56_pct': []},
            'linear_2': {'steps': [], 'hist_48_52_pct': [], 'hist_52_56_pct': []},
            'linear_3': {'steps': [], 'hist_48_52_pct': [], 'hist_52_56_pct': []},
        }

        # Current task step tracking (reset after each task for per-task averaging)
        losses_per_step = []
        plasticity_per_step = []
        n_dead_units_per_step = []
        weight_rank_per_step = []
        weight_l2_per_step = []
        weight_l1_per_step = []
        grad_l2_per_step = []
        grad_l1_per_step = []
        grad_l0_per_step = []
        
        # Curvature tracking per step
        if self.is_input_aware:
            input_curvature_per_step = []
            lambda_values_per_step = []
            # Track all curvature values within current task for statistics
            current_task_curvatures = []
        else:
            # Standard learners also track curvature for analysis
            input_curvature_per_step = []
            lambda_values_per_step = []

        if self.task.criterion == 'cross_entropy':
            all_accuracy_per_step = []
            accuracy_per_step = []

        print(f"Starting training for {self.n_samples} samples...", flush=True)
        for i in range(self.n_samples):
            # Progress logging (every 1000 steps)
            if i % 1000 == 0:
                print(f"Step {i}/{self.n_samples} ({100*i/self.n_samples:.1f}%)", flush=True)
            
            input, target = next(self.task)
            input, target = input.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            # For input-aware learners, we need to track curvature
            if self.is_input_aware and i % self.compute_curvature_every == 0:
                # Make sure input requires grad for curvature computation
                input.requires_grad_(True)
            
            output = self.learner.predict(input)
            loss = criterion(output, target)
            
            # Compute input curvature for input-aware learners
            current_curvature = 0.0
            current_lambda = 0.0
            if self.is_input_aware and i % self.compute_curvature_every == 0:
                try:
                    # Compute input curvature using the new finite differences method
                    current_curvature = self.learner.compute_input_curvature(
                        model=self.learner.network, 
                        input_batch=input, 
                        targets=target, 
                        criterion=criterion
                    )
                    
                    # Update optimizer with curvature
                    self.learner.update_optimizer_curvature(current_curvature)
                    
                    # Get current lambda value from optimizer
                    if hasattr(optimizer, 'compute_lambda'):
                        current_lambda = optimizer.compute_lambda()
                        
                    input_curvature_per_step.append(current_curvature)
                    lambda_values_per_step.append(current_lambda)
                    # Store individual curvature for task-level statistics
                    current_task_curvatures.append(current_curvature)
                    
                    # Remove console logging for curvature
                    # print(f"Step {i}: Input curvature = {current_curvature:.6f}, Lambda = {current_lambda:.6f}")
                    
                    # Log to wandb every curvature computation
                    if self.wandb_enabled:
                        wandb.log({
                            'curvature/current': input_curvature_per_step[-1],
                            'curvature/avg': sum(input_curvature_per_step) / len(input_curvature_per_step),
                            'curvature/max': max(input_curvature_per_step),
                            'curvature/lambda': lambda_values_per_step[-1] if lambda_values_per_step else 0,
                        }, step=i)
                    
                except Exception as e:
                    print(f"Warning: Could not compute curvature at step {i}: {e}")
                    if self.is_input_aware:
                        input_curvature_per_step.append(0.0)
                        lambda_values_per_step.append(0.0)
                        current_task_curvatures.append(0.0)
            elif self.is_input_aware:
                # Use previous curvature values when not computing
                prev_curvature = input_curvature_per_step[-1] if input_curvature_per_step else 0.0
                prev_lambda = lambda_values_per_step[-1] if lambda_values_per_step else 0.0
                input_curvature_per_step.append(prev_curvature)
                lambda_values_per_step.append(prev_lambda)
                # Also add to task curvatures for statistics (using previous value)
                current_task_curvatures.append(prev_curvature)
            else:
                # Standard learners: analysis-only curvature
                if i % self.compute_curvature_every == 0:
                    try:
                        curvatures = compute_input_curvature_finite_diff(
                            model=self.learner.network,
                            inputs=input,
                            targets=target,
                            criterion=criterion,
                            h=1e-3,
                            niter=int(os.environ.get('HUTCHINSON_SAMPLES', 5)),
                            temp=1.0,
                        )
                        current_curvature = float(np.mean(curvatures))
                        current_lambda = 0.0
                        input_curvature_per_step.append(current_curvature)
                        lambda_values_per_step.append(current_lambda)
                        if self.wandb_enabled:
                            wandb.log({
                                'curvature/current': current_curvature,
                                'curvature/avg': sum(input_curvature_per_step) / len(input_curvature_per_step),
                                'curvature/max': max(input_curvature_per_step),
                                'curvature/lambda': 0.0,
                            }, step=i)
                    except Exception as e:
                        print(f"Warning: Could not compute curvature at step {i}: {e}")
                        input_curvature_per_step.append(0.0)
                        lambda_values_per_step.append(0.0)
                else:
                    prev_curvature = input_curvature_per_step[-1] if input_curvature_per_step else 0.0
                    prev_lambda = lambda_values_per_step[-1] if lambda_values_per_step else 0.0
                    input_curvature_per_step.append(prev_curvature)
                    lambda_values_per_step.append(prev_lambda)
            
            if self.learner.extend:
                with backpack(extension):
                    loss.backward()
            else:
                loss.backward()
            optimizer.step()
            loss_val = loss.item()
            losses_per_step.append(loss_val)
            all_losses_per_step.append(loss_val)

            # Compute accuracy immediately after getting output
            current_accuracy = None
            if self.task.criterion == 'cross_entropy':
                current_accuracy = (output.argmax(dim=1) == target).float().mean().item()
                accuracy_per_step.append(current_accuracy)
                all_accuracy_per_step.append(current_accuracy)

            # compute some statistics after each task change
            with torch.no_grad():
                output_new = self.learner.predict(input)
                loss_after = criterion(output_new, target)
                loss_before = torch.clamp(loss, min=1e-8)
                plasticity_val = torch.clamp((1-loss_after/loss_before), min=0.0, max=1.0).item()
                plasticity_per_step.append(plasticity_val)
                all_plasticity_per_step.append(plasticity_val)
            n_dead_units = 0
            for _, value in self.learner.network.activations.items():
                n_dead_units += value
            dead_units_val = n_dead_units / self.learner.network.n_units
            n_dead_units_per_step.append(dead_units_val)
            all_n_dead_units_per_step.append(dead_units_val)

            sample_weight_rank = 0.0
            sample_max_rank = 0.0
            sample_weight_l2 = 0.0
            sample_grad_l2 = 0.0
            sample_weight_l1 = 0.0
            sample_grad_l1 = 0.0
            sample_grad_l0 = 0.0
            sample_n_weights = 0.0

            for name, param in self.learner.network.named_parameters():
                if 'weight' in name:
                    if 'conv' in name:
                        sample_weight_rank += torch.linalg.matrix_rank(param.data).float().mean()
                        sample_max_rank += torch.min(torch.tensor(param.data.shape)[-2:])
                    else:
                        sample_weight_rank += torch.linalg.matrix_rank(param.data)
                        sample_max_rank += torch.min(torch.tensor(param.data.shape))
                    sample_weight_l2 += torch.norm(param.data, p=2) ** 2
                    sample_weight_l1 += torch.norm(param.data, p=1)

                    sample_grad_l2 += torch.norm(param.grad.data, p=2) ** 2
                    sample_grad_l1 += torch.norm(param.grad.data, p=1)

                    sample_grad_l0 += torch.norm(param.grad.data, p=0)
                    sample_n_weights += torch.numel(param.data)

            weight_l2_val = sample_weight_l2.sqrt().item()
            weight_l1_val = sample_weight_l1.item()
            grad_l2_val = sample_grad_l2.sqrt().item()
            grad_l1_val = sample_grad_l1.item()
            grad_l0_val = sample_grad_l0.item()/sample_n_weights
            weight_rank_val = sample_weight_rank.item() / sample_max_rank.item()

            weight_l2_per_step.append(weight_l2_val)
            weight_l1_per_step.append(weight_l1_val)
            grad_l2_per_step.append(grad_l2_val)
            grad_l1_per_step.append(grad_l1_val)
            grad_l0_per_step.append(grad_l0_val)
            weight_rank_per_step.append(weight_rank_val)

            all_weight_l2_per_step.append(weight_l2_val)
            all_weight_l1_per_step.append(weight_l1_val)
            all_grad_l2_per_step.append(grad_l2_val)
            all_grad_l1_per_step.append(grad_l1_val)
            all_grad_l0_per_step.append(grad_l0_val)
            all_weight_rank_per_step.append(weight_rank_val)
            
            # Incremental JSON updates for long runs
            if i % 1000 == 0 and i > 0:
                try:
                    self.logger.log_incremental(
                        {
                            'status': 'in_progress',
                            'current_step': i,
                            'progress_percent': i / self.n_samples,
                        },
                        task=self.task_name,
                        learner=self.learner.name,
                        network=self.learner.network.name,
                        optimizer_hps=self.learner.optim_kwargs,
                        n_samples=self.n_samples,
                        seed=self.seed,
                    )
                except Exception as e:
                    print(f"Warning: incremental logging failed at step {i}: {e}")

            # Log comprehensive step metrics every 10 steps to wandb
            if self.wandb_enabled and i % 10 == 0:
                # Basic training metrics
                step_metrics = {
                    'training/loss': loss.item(),
                    'training/step': i,
                    'training/task_progress': (i % self.task.change_freq) / self.task.change_freq,
                    'training/global_progress': i / self.n_samples,
                    'training/current_task': len(losses_per_task),
                }
                
                # Accuracy for classification tasks
                if self.task.criterion == 'cross_entropy' and current_accuracy is not None:
                    step_metrics['training/accuracy'] = current_accuracy
                    if accuracy_per_step:
                        step_metrics['training/avg_accuracy'] = sum(accuracy_per_step) / len(accuracy_per_step)
                
                # Plasticity metrics - ensure we have data before logging
                if plasticity_per_step:
                    step_metrics['plasticity/current'] = plasticity_per_step[-1]
                    step_metrics['plasticity/avg'] = sum(plasticity_per_step) / len(plasticity_per_step)
                
                # Network health metrics - ensure we have data
                if n_dead_units_per_step:
                    step_metrics['network/dead_units_ratio'] = n_dead_units_per_step[-1]
                    step_metrics['network/avg_dead_units'] = sum(n_dead_units_per_step) / len(n_dead_units_per_step)
                
                # Weight statistics - ensure we have data
                if weight_l2_per_step:
                    step_metrics['weights/l2_norm'] = weight_l2_per_step[-1]
                    step_metrics['weights/avg_l2'] = sum(weight_l2_per_step) / len(weight_l2_per_step)
                if weight_l1_per_step:
                    step_metrics['weights/l1_norm'] = weight_l1_per_step[-1]
                if weight_rank_per_step:
                    step_metrics['weights/rank_ratio'] = weight_rank_per_step[-1]
                
                # Gradient statistics - ensure we have data
                if grad_l2_per_step:
                    step_metrics['gradients/l2_norm'] = grad_l2_per_step[-1]
                    step_metrics['gradients/avg_l2'] = sum(grad_l2_per_step) / len(grad_l2_per_step)
                if grad_l1_per_step:
                    step_metrics['gradients/l1_norm'] = grad_l1_per_step[-1]
                if grad_l0_per_step:
                    step_metrics['gradients/l0_ratio'] = grad_l0_per_step[-1]
                
                # Add curvature metrics for all learners
                if input_curvature_per_step:
                    step_metrics['curvature/current'] = input_curvature_per_step[-1] if input_curvature_per_step else 0
                    step_metrics['curvature/avg'] = sum(input_curvature_per_step) / len(input_curvature_per_step)
                    step_metrics['curvature/max'] = max(input_curvature_per_step)
                    step_metrics['curvature/lambda'] = lambda_values_per_step[-1] if lambda_values_per_step else 0
                
                # Add utility metrics if available (use optimizer instance, not class)
                if hasattr(optimizer, 'get_utility_stats'):
                    utility_stats = optimizer.get_utility_stats()
                    step_metrics.update(utility_stats)

                    # Save utility histogram data to local tracking (for JSON export)
                    if 'utility/hist_48_52_pct' in utility_stats:
                        all_utility_hist_per_step['steps'].append(i)
                        all_utility_hist_per_step['hist_0_20_pct'].append(utility_stats.get('utility/hist_0_20_pct', 0))
                        all_utility_hist_per_step['hist_20_40_pct'].append(utility_stats.get('utility/hist_20_40_pct', 0))
                        all_utility_hist_per_step['hist_40_44_pct'].append(utility_stats.get('utility/hist_40_44_pct', 0))
                        all_utility_hist_per_step['hist_44_48_pct'].append(utility_stats.get('utility/hist_44_48_pct', 0))
                        all_utility_hist_per_step['hist_48_52_pct'].append(utility_stats.get('utility/hist_48_52_pct', 0))
                        all_utility_hist_per_step['hist_52_56_pct'].append(utility_stats.get('utility/hist_52_56_pct', 0))
                        all_utility_hist_per_step['hist_56_60_pct'].append(utility_stats.get('utility/hist_56_60_pct', 0))
                        all_utility_hist_per_step['hist_60_80_pct'].append(utility_stats.get('utility/hist_60_80_pct', 0))
                        all_utility_hist_per_step['hist_80_100_pct'].append(utility_stats.get('utility/hist_80_100_pct', 0))
                        all_utility_hist_per_step['global_max'].append(utility_stats.get('utility/global_max', 0))
                        # Save total_params (only need to set once, it's constant)
                        if all_utility_hist_per_step['total_params'] is None:
                            all_utility_hist_per_step['total_params'] = utility_stats.get('utility/total_params', 0)

                        # Per-layer utility histograms
                        for layer in ['linear_1', 'linear_2', 'linear_3']:
                            key_48_52 = f'layer/{layer}/hist_48_52_pct'
                            key_52_56 = f'layer/{layer}/hist_52_56_pct'
                            if key_48_52 in utility_stats:
                                all_layer_utility_hist_per_step[layer]['steps'].append(i)
                                all_layer_utility_hist_per_step[layer]['hist_48_52_pct'].append(utility_stats[key_48_52])
                                all_layer_utility_hist_per_step[layer]['hist_52_56_pct'].append(utility_stats.get(key_52_56, 0))
                
                # Log histograms every 100 steps using wandb.Histogram() for proper visualization
                if i % 100 == 0 and hasattr(optimizer, 'get_histogram_tensors'):
                    histogram_tensors = optimizer.get_histogram_tensors()
                    if histogram_tensors:
                        # Convert to wandb.Histogram for proper visualization
                        histogram_metrics = {}
                        for key, tensor in histogram_tensors.items():
                            if tensor is not None and tensor.numel() > 0:
                                histogram_metrics[key] = wandb.Histogram(tensor.numpy())
                        if histogram_metrics:
                            step_metrics.update(histogram_metrics)
                            if i % 1000 == 0:
                                print(f"Step {i}: Logged {len(histogram_metrics)} histograms: {list(histogram_metrics.keys())}")
                
                wandb.log(step_metrics, step=i, commit=True)  # commit=True forces immediate sync
                
                # Force sync every 1000 steps to ensure data appears in dashboard
                if i % 1000 == 0 and self.wandb_enabled:
                    try:
                        wandb.run.summary.update({'last_synced_step': i})
                        wandb.run.summary.update({'progress_percent': 100.0 * i / self.n_samples})
                    except:
                        pass

            if i % self.task.change_freq == 0:
                losses_per_task.append(sum(losses_per_step) / len(losses_per_step))
                if self.task.criterion == 'cross_entropy':
                    accuracy_per_task.append(sum(accuracy_per_step) / len(accuracy_per_step))
                plasticity_per_task.append(sum(plasticity_per_step) / len(plasticity_per_step))
                n_dead_units_per_task.append(sum(n_dead_units_per_step) / len(n_dead_units_per_step))
                weight_rank_per_task.append(sum(weight_rank_per_step) / len(weight_rank_per_step))
                weight_l2_per_task.append(sum(weight_l2_per_step) / len(weight_l2_per_step))
                weight_l1_per_task.append(sum(weight_l1_per_step) / len(weight_l1_per_step))
                grad_l2_per_task.append(sum(grad_l2_per_step) / len(grad_l2_per_step))
                grad_l1_per_task.append(sum(grad_l1_per_step) / len(grad_l1_per_step))
                grad_l0_per_task.append(sum(grad_l0_per_step) / len(grad_l0_per_step))
                
                # Add curvature statistics for input-aware learners
                if self.is_input_aware:
                    if input_curvature_per_step and current_task_curvatures:
                        # Compute statistics for this task
                        task_curvatures = np.array(current_task_curvatures)
                        
                        input_curvature_per_task.append(task_curvatures.mean())
                        lambda_values_per_task.append(sum(lambda_values_per_step) / len(lambda_values_per_step))
                        
                        # Store curvature statistics
                        curvature_max_per_task.append(task_curvatures.max())
                        curvature_min_per_task.append(task_curvatures.min())
                        curvature_std_per_task.append(task_curvatures.std())
                        
                        # Track current average curvature from optimizer
                        if hasattr(optimizer, 'avg_input_curvature'):
                            avg_curvature_per_task.append(optimizer.avg_input_curvature)
                        else:
                            avg_curvature_per_task.append(0.0)
                            
                        # Remove console logging for task curvature stats
                        # print(f"Task {len(input_curvature_per_task)}: Curvature stats - Mean: {task_curvatures.mean():.6f}, Max: {task_curvatures.max():.6f}, Min: {task_curvatures.min():.6f}, Std: {task_curvatures.std():.6f}")
                        
                        # Log comprehensive task-level statistics to wandb
                        if self.wandb_enabled:
                            task_metrics = {
                                # Task identification
                                'task_level/task_number': len(input_curvature_per_task),
                                'task_level/step': i,
                                
                                # Core training metrics
                                'task_level/loss': sum(losses_per_step) / len(losses_per_step),
                                'task_level/plasticity': sum(plasticity_per_step) / len(plasticity_per_step),
                                
                                # Network health
                                'task_level/dead_units_ratio': sum(n_dead_units_per_step) / len(n_dead_units_per_step),
                                
                                # Weight statistics
                                'task_level/weight_l2': sum(weight_l2_per_step) / len(weight_l2_per_step),
                                'task_level/weight_l1': sum(weight_l1_per_step) / len(weight_l1_per_step),
                                'task_level/weight_rank': sum(weight_rank_per_step) / len(weight_rank_per_step),
                                
                                # Gradient statistics
                                'task_level/grad_l2': sum(grad_l2_per_step) / len(grad_l2_per_step),
                                'task_level/grad_l1': sum(grad_l1_per_step) / len(grad_l1_per_step),
                                'task_level/grad_l0': sum(grad_l0_per_step) / len(grad_l0_per_step),
                                
                                # Curvature statistics
                                'task_level/curvature_mean': task_curvatures.mean(),
                                'task_level/curvature_max': task_curvatures.max(),
                                'task_level/curvature_min': task_curvatures.min(),
                                'task_level/curvature_std': task_curvatures.std(),
                                'task_level/avg_lambda': sum(lambda_values_per_step) / len(lambda_values_per_step),
                            }
                            
                            # Add accuracy for classification tasks
                            if self.task.criterion == 'cross_entropy' and accuracy_per_step:
                                task_metrics['task_level/accuracy'] = sum(accuracy_per_step) / len(accuracy_per_step)
                            
                            wandb.log(task_metrics, step=i)
                        
                    else:
                        input_curvature_per_task.append(0.0)
                        lambda_values_per_task.append(0.0)
                        avg_curvature_per_task.append(0.0)
                        curvature_max_per_task.append(0.0)
                        curvature_min_per_task.append(0.0)
                        curvature_std_per_task.append(0.0)

                # Reset per-step arrays after each task (matches original UPGD behavior)
                # This ensures per-task metrics are window averages, not cumulative
                losses_per_step = []
                if self.task.criterion == 'cross_entropy':
                    accuracy_per_step = []
                plasticity_per_step = []
                n_dead_units_per_step = []
                weight_rank_per_step = []
                weight_l2_per_step = []
                weight_l1_per_step = []
                grad_l2_per_step = []
                grad_l1_per_step = []
                grad_l0_per_step = []

                # Reset task-level curvature tracking
                if self.is_input_aware:
                    current_task_curvatures = []  # Reset for next task

        # Prepare logging data with BOTH task-level and step-level data
        log_data = {
            # Task-level summaries
            'losses': losses_per_task,
            'plasticity_per_task': plasticity_per_task,
            'n_dead_units_per_task': n_dead_units_per_task,
            'weight_rank_per_task': weight_rank_per_task,
            'weight_l2_per_task': weight_l2_per_task,
            'weight_l1_per_task': weight_l1_per_task,
            'grad_l2_per_task': grad_l2_per_task,
            'grad_l0_per_task': grad_l0_per_task,
            'grad_l1_per_task': grad_l1_per_task,
            
            # Step-level data (ALL steps)
            'losses_per_step': all_losses_per_step,
            'plasticity_per_step': all_plasticity_per_step,
            'n_dead_units_per_step': all_n_dead_units_per_step,
            'weight_rank_per_step': all_weight_rank_per_step,
            'weight_l2_per_step': all_weight_l2_per_step,
            'weight_l1_per_step': all_weight_l1_per_step,
            'grad_l2_per_step': all_grad_l2_per_step,
            'grad_l0_per_step': all_grad_l0_per_step,
            'grad_l1_per_step': all_grad_l1_per_step,
            
            # Metadata
            'task': self.task_name, 
            'learner': self.learner.name,
            'network': self.learner.network.name,
            'optimizer_hps': self.learner.optim_kwargs,
            'n_samples': self.n_samples,
            'seed': self.seed,
            'status': 'completed',
            'current_step': self.n_samples,
            'progress_percent': 1.0,
        }
        
        # Add accuracy for classification tasks
        if self.task.criterion == 'cross_entropy':
            log_data['accuracies'] = accuracy_per_task
            log_data['accuracy_per_step'] = all_accuracy_per_step
            
        # Add curvature data for input-aware learners (task-level), and include step-level for all
        if self.is_input_aware:
            log_data.update({
                # Task-level curvature summaries
                'input_curvature_per_task': input_curvature_per_task,
                'lambda_values_per_task': lambda_values_per_task,
                'avg_curvature_per_task': avg_curvature_per_task,
                'curvature_max_per_task': curvature_max_per_task,
                'curvature_min_per_task': curvature_min_per_task,
                'curvature_std_per_task': curvature_std_per_task,
                
                # Step-level curvature data (ALL steps)
                'input_curvature_per_step': input_curvature_per_step,
                'lambda_values_per_step': lambda_values_per_step,
                
                # Configuration
                'compute_curvature_every': self.compute_curvature_every,
            })
        else:
            log_data.update({
                'input_curvature_per_step': input_curvature_per_step,
                'lambda_values_per_step': lambda_values_per_step,
                'compute_curvature_every': self.compute_curvature_every,
            })

        # Add utility histogram data (9 bins) if collected
        if all_utility_hist_per_step['steps']:
            log_data['utility_histogram_per_step'] = all_utility_hist_per_step
            log_data['layer_utility_histogram_per_step'] = all_layer_utility_hist_per_step

        # Log comprehensive final summary to wandb
        if self.wandb_enabled:
            final_summary = {
                # Training overview
                'summary/final_loss': all_losses_per_step[-1] if all_losses_per_step else 0.0,
                'summary/avg_loss': sum(all_losses_per_step) / len(all_losses_per_step) if all_losses_per_step else 0.0,
                'summary/min_loss': min(all_losses_per_step) if all_losses_per_step else 0.0,
                'summary/total_steps': len(all_losses_per_step),
                'summary/total_tasks': len(losses_per_task),

                # Plasticity summary
                'summary/final_plasticity': all_plasticity_per_step[-1] if all_plasticity_per_step else 0.0,
                'summary/avg_plasticity': sum(all_plasticity_per_step) / len(all_plasticity_per_step) if all_plasticity_per_step else 0.0,
                'summary/min_plasticity': min(all_plasticity_per_step) if all_plasticity_per_step else 0.0,

                # Network health summary
                'summary/final_dead_units': all_n_dead_units_per_step[-1] if all_n_dead_units_per_step else 0.0,
                'summary/avg_dead_units': sum(all_n_dead_units_per_step) / len(all_n_dead_units_per_step) if all_n_dead_units_per_step else 0.0,
                'summary/max_dead_units': max(all_n_dead_units_per_step) if all_n_dead_units_per_step else 0.0,

                # Weight statistics summary
                'summary/final_weight_l2': all_weight_l2_per_step[-1] if all_weight_l2_per_step else 0.0,
                'summary/avg_weight_l2': sum(all_weight_l2_per_step) / len(all_weight_l2_per_step) if all_weight_l2_per_step else 0.0,
                'summary/final_weight_rank': all_weight_rank_per_step[-1] if all_weight_rank_per_step else 0.0,

                # Gradient statistics summary
                'summary/final_grad_l2': all_grad_l2_per_step[-1] if all_grad_l2_per_step else 0.0,
                'summary/avg_grad_l2': sum(all_grad_l2_per_step) / len(all_grad_l2_per_step) if all_grad_l2_per_step else 0.0,
                'summary/max_grad_l2': max(all_grad_l2_per_step) if all_grad_l2_per_step else 0.0,
            }
            
            # Add curvature summary for all learners (standard = analysis-only)
            if input_curvature_per_step:
                final_summary.update({
                    'summary/avg_curvature': sum(input_curvature_per_step) / len(input_curvature_per_step),
                    'summary/max_curvature': max(input_curvature_per_step),
                    'summary/min_curvature': min(input_curvature_per_step),
                    'summary/final_curvature': input_curvature_per_step[-1],
                    'summary/final_lambda': lambda_values_per_step[-1] if lambda_values_per_step else 0.0,
                    'summary/avg_lambda': sum(lambda_values_per_step) / len(lambda_values_per_step) if lambda_values_per_step else 0.0,
                    'summary/max_lambda': max(lambda_values_per_step) if lambda_values_per_step else 0.0,
                })
            
            # Add accuracy summary for classification tasks
            if self.task.criterion == 'cross_entropy' and all_accuracy_per_step:
                final_summary.update({
                    'summary/final_accuracy': all_accuracy_per_step[-1],
                    'summary/avg_accuracy': sum(all_accuracy_per_step) / len(all_accuracy_per_step),
                    'summary/max_accuracy': max(all_accuracy_per_step),
                    'summary/min_accuracy': min(all_accuracy_per_step),
                })
            
            wandb.log(final_summary)
            wandb.finish()
            print(f"WandB logging completed")
        
        self.logger.log(**log_data)


if __name__ == "__main__":
    # start the run using the command line arguments
    ll = sys.argv[1:]
    args = {k[2:]:v for k,v in zip(ll[::2], ll[1::2])}
    run = RunStatsWithCurvature(**args)
    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")