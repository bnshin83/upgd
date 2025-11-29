from core.learner.learner import Learner
from core.optim.weight_upgd.input_aware import (
    InputAwareFirstOrderGlobalUPGD, 
    InputAwareSecondOrderGlobalUPGD,
    hutchinson_trace_estimator,
    compute_input_curvature_finite_diff
)
import torch


class InputAwareFirstOrderGlobalUPGDLearner(Learner):
    """
    Input-aware first-order UPGD learner that computes input curvature
    and uses it to modulate parameter protection.
    """
    
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = InputAwareFirstOrderGlobalUPGD
        name = "input_aware_upgd_fo_global"
        
        # Filter out run-specific parameters that shouldn't go to optimizer
        run_specific_params = {'compute_curvature_every', 'n_samples', 'task', 'learner', 'save_path', 'seed', 'network'}
        filtered_kwargs = {k: v for k, v in optim_kwargs.items() if k not in run_specific_params}
        
        # Convert string parameters to appropriate types for optimizer
        if 'lr' in filtered_kwargs:
            filtered_kwargs['lr'] = float(filtered_kwargs['lr'])
        if 'weight_decay' in filtered_kwargs:
            filtered_kwargs['weight_decay'] = float(filtered_kwargs['weight_decay'])
        if 'beta_utility' in filtered_kwargs:
            filtered_kwargs['beta_utility'] = float(filtered_kwargs['beta_utility'])
        if 'sigma' in filtered_kwargs:
            filtered_kwargs['sigma'] = float(filtered_kwargs['sigma'])
        if 'curvature_threshold' in filtered_kwargs:
            filtered_kwargs['curvature_threshold'] = float(filtered_kwargs['curvature_threshold'])
        if 'lambda_max' in filtered_kwargs:
            filtered_kwargs['lambda_max'] = float(filtered_kwargs['lambda_max'])
        if 'lambda_scale' in filtered_kwargs:
            filtered_kwargs['lambda_scale'] = float(filtered_kwargs['lambda_scale'])
        if 'beta_curvature' in filtered_kwargs:
            filtered_kwargs['beta_curvature'] = float(filtered_kwargs['beta_curvature'])
        if 'hutchinson_samples' in filtered_kwargs:
            filtered_kwargs['hutchinson_samples'] = int(filtered_kwargs['hutchinson_samples'])
        # Option C numeric parameters (safe to pass through)
        if 'gating_option_c_a' in filtered_kwargs:
            filtered_kwargs['gating_option_c_a'] = float(filtered_kwargs['gating_option_c_a'])
        if 'gating_option_c_b' in filtered_kwargs:
            filtered_kwargs['gating_option_c_b'] = float(filtered_kwargs['gating_option_c_b'])
        if 'gating_min_g' in filtered_kwargs:
            filtered_kwargs['gating_min_g'] = float(filtered_kwargs['gating_min_g'])
        if 'disable_regularization' in filtered_kwargs:
            val = filtered_kwargs['disable_regularization']
            filtered_kwargs['disable_regularization'] = val if isinstance(val, bool) else val.lower() == 'true'
        if 'disable_gating' in filtered_kwargs:
            val = filtered_kwargs['disable_gating']
            filtered_kwargs['disable_gating'] = val if isinstance(val, bool) else val.lower() == 'true'
        
        super().__init__(name, network, optimizer, filtered_kwargs)
        self.hutchinson_samples = int(optim_kwargs.get('hutchinson_samples', 1))
        
    def compute_input_curvature(self, model, input_batch, targets, criterion, return_per_sample=False):
        """
        Compute input curvature for current batch using finite differences method.
        
        This follows the methodology from post_run_analysis_modified2.py.
        
        Args:
            model: The neural network model
            input_batch: input tensor
            targets: target tensor
            criterion: loss function
            return_per_sample: If True, return per-sample curvatures, else return average
            
        Returns:
            Per-sample curvatures (numpy array) if return_per_sample=True, else average curvature (float)
        """
        # Use finite differences method like in the existing codebase
        per_sample_curvatures = compute_input_curvature_finite_diff(
            model=model,
            inputs=input_batch, 
            targets=targets,
            criterion=criterion,
            h=1e-3,  # Perturbation size
            niter=self.hutchinson_samples,  # Use hutchinson_samples as niter
            temp=1.0  # Temperature scaling
        )
        
        if return_per_sample:
            return per_sample_curvatures  # Return numpy array of per-sample values
        else:
            return per_sample_curvatures.mean()  # Return average for backward compatibility
    
    def update_optimizer_curvature(self, curvature):
        """Update the optimizer with current input curvature."""
        if hasattr(self, 'optimizer_instance'):
            self.optimizer_instance.set_input_curvature(curvature)


class InputAwareSecondOrderGlobalUPGDLearner(Learner):
    """
    Input-aware second-order UPGD learner using HesScale for parameter curvature
    and input curvature for protection gating.
    """
    
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = InputAwareSecondOrderGlobalUPGD
        name = "input_aware_upgd_so_global"
        
        # Filter out run-specific parameters that shouldn't go to optimizer
        run_specific_params = {'compute_curvature_every', 'n_samples', 'task', 'learner', 'save_path', 'seed', 'network'}
        filtered_kwargs = {k: v for k, v in optim_kwargs.items() if k not in run_specific_params}
        
        # Convert string parameters to appropriate types for optimizer
        if 'lr' in filtered_kwargs:
            filtered_kwargs['lr'] = float(filtered_kwargs['lr'])
        if 'weight_decay' in filtered_kwargs:
            filtered_kwargs['weight_decay'] = float(filtered_kwargs['weight_decay'])
        if 'beta_utility' in filtered_kwargs:
            filtered_kwargs['beta_utility'] = float(filtered_kwargs['beta_utility'])
        if 'sigma' in filtered_kwargs:
            filtered_kwargs['sigma'] = float(filtered_kwargs['sigma'])
        if 'curvature_threshold' in filtered_kwargs:
            filtered_kwargs['curvature_threshold'] = float(filtered_kwargs['curvature_threshold'])
        if 'lambda_max' in filtered_kwargs:
            filtered_kwargs['lambda_max'] = float(filtered_kwargs['lambda_max'])
        if 'lambda_scale' in filtered_kwargs:
            filtered_kwargs['lambda_scale'] = float(filtered_kwargs['lambda_scale'])
        if 'beta_curvature' in filtered_kwargs:
            filtered_kwargs['beta_curvature'] = float(filtered_kwargs['beta_curvature'])
        if 'hutchinson_samples' in filtered_kwargs:
            filtered_kwargs['hutchinson_samples'] = int(filtered_kwargs['hutchinson_samples'])
        # Option C numeric parameters (safe to pass through)
        if 'gating_option_c_a' in filtered_kwargs:
            filtered_kwargs['gating_option_c_a'] = float(filtered_kwargs['gating_option_c_a'])
        if 'gating_option_c_b' in filtered_kwargs:
            filtered_kwargs['gating_option_c_b'] = float(filtered_kwargs['gating_option_c_b'])
        if 'gating_min_g' in filtered_kwargs:
            filtered_kwargs['gating_min_g'] = float(filtered_kwargs['gating_min_g'])
        if 'disable_regularization' in filtered_kwargs:
            val = filtered_kwargs['disable_regularization']
            filtered_kwargs['disable_regularization'] = val if isinstance(val, bool) else val.lower() == 'true'
        if 'disable_gating' in filtered_kwargs:
            val = filtered_kwargs['disable_gating']
            filtered_kwargs['disable_gating'] = val if isinstance(val, bool) else val.lower() == 'true'
        
        super().__init__(name, network, optimizer, filtered_kwargs, extend=True)
        self.hutchinson_samples = int(optim_kwargs.get('hutchinson_samples', 1))
        
    def compute_input_curvature(self, model, input_batch, targets, criterion):
        """
        Compute input curvature for current batch using finite differences method.
        
        This follows the methodology from post_run_analysis_modified2.py.
        
        Args:
            model: The neural network model
            input_batch: input tensor
            targets: target tensor
            criterion: loss function
            
        Returns:
            Estimated input curvature
        """
        # Use finite differences method like in the existing codebase
        curvature = compute_input_curvature_finite_diff(
            model=model,
            inputs=input_batch, 
            targets=targets,
            criterion=criterion,
            h=1e-3,  # Perturbation size
            niter=self.hutchinson_samples,  # Use hutchinson_samples as niter
            temp=1.0  # Temperature scaling
        )
        
        return curvature
    
    def update_optimizer_curvature(self, curvature):
        """Update the optimizer with current input curvature."""
        if hasattr(self, 'optimizer_instance'):
            self.optimizer_instance.set_input_curvature(curvature)
