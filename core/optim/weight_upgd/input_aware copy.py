import torch
from torch.nn import functional as F
import sys
import os
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale


def compute_input_curvature_finite_diff(model, inputs, targets, criterion, h=1e-3, niter=10, temp=1.0):
    """
    Compute input curvature using finite differences method from post_run_analysis_modified2.py.
    
    This method follows the approach used in the existing codebase, using Rademacher random vectors
    and finite differences to estimate input-space curvature.
    
    Args:
        model: The neural network model
        inputs: Input tensor batch [batch_size, ...]
        targets: Target tensor batch [batch_size, ...]
        criterion: Loss function (e.g., CrossEntropyLoss)
        h: Perturbation size for finite differences (default: 1e-3)
        niter: Number of random directions to sample (default: 10)
        temp: Temperature scaling for softmax (default: 1.0)
    
    Returns:
        Average curvature estimate for the batch
    """
    device = inputs.device
    model.eval()
    
    num_samples = inputs.shape[0]
    regr = torch.zeros(num_samples, device=device)
    
    # Ensure inputs requires gradients
    inputs = inputs.detach().requires_grad_(True)
    
    # Perturb each input in niter random directions
    for _ in range(niter):
        # Generate Rademacher random vector (±1)
        v = torch.randint_like(inputs, high=2, device=device) * 2 - 1  # Rademacher (±1)
        v = h * v.float()  # Scale perturbation and ensure float type
        
        with torch.enable_grad():
            # Forward pass on perturbed and original inputs
            outputs_pos = model(inputs + v)
            outputs_orig = model(inputs)
            
            # Compute losses
            loss_pos = criterion(outputs_pos / temp, targets)
            loss_orig = criterion(outputs_orig / temp, targets)
            
            # Compute gradient difference (finite difference approximation)
            grad_diff = torch.autograd.grad(loss_pos - loss_orig, inputs, create_graph=False, retain_graph=False)[0]
        
        # Accumulate gradient norm per sample
        regr += grad_diff.reshape(num_samples, -1).norm(dim=1)
        
        # Clear gradients
        model.zero_grad()
        if inputs.grad is not None:
            inputs.grad.zero_()
    
    # Return per-sample curvatures (normalized by niter)
    per_sample_curvatures = regr / niter
    return per_sample_curvatures.cpu().numpy()  # Return as numpy array for easier handling


def compute_input_curvature_finite_diff_average(model, inputs, targets, criterion, h=1e-3, niter=10, temp=1.0):
    """
    Compute average input curvature using finite differences method.
    
    This is a wrapper around compute_input_curvature_finite_diff that returns
    the average curvature for backward compatibility.
    """
    per_sample_curvatures = compute_input_curvature_finite_diff(
        model, inputs, targets, criterion, h, niter, temp
    )
    return per_sample_curvatures.mean()


def hutchinson_trace_estimator(loss, inputs, n_samples=1):
    """
    DEPRECATED: Use compute_input_curvature_finite_diff instead.
    
    Estimate tr(H_x^2) using Hutchinson's estimator.
    
    Args:
        loss: scalar loss value
        inputs: input tensor for which to compute Hessian
        n_samples: number of random vector samples for estimation
    
    Returns:
        Estimated tr(H_x^2) value
    """
    trace_estimate = 0.0
    
    for _ in range(n_samples):
        # Sample random vector from standard normal
        v = torch.randn_like(inputs, requires_grad=False)
        
        # Compute Hessian-vector product
        grad_loss = torch.autograd.grad(loss, inputs, create_graph=True, retain_graph=True)[0]
        hvp = torch.autograd.grad(grad_loss, inputs, grad_outputs=v, retain_graph=True)[0]
        
        # Accumulate squared norm
        trace_estimate += (hvp * hvp).sum().item()
    
    return trace_estimate / n_samples


class InputAwareFirstOrderGlobalUPGD(torch.optim.Optimizer):
    """
    Input-aware UPGD that modulates parameter protection based on input curvature.
    Combines UPGD's utility-based protection with input-space curvature gating.
    """
    
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.99, sigma=1.0,
                 curvature_threshold=1.0, lambda_max=1.0, lambda_scale=0.1,
                 beta_curvature=0.9, hutchinson_samples=1, disable_regularization=False,
                 disable_gating=False):
        """
        Args:
            lr: learning rate
            weight_decay: L2 weight decay
            beta_utility: momentum coefficient for utility tracking
            sigma: noise standard deviation for perturbation
            curvature_threshold: threshold for considering input as "high curvature"
            lambda_max: maximum regularization strength
            lambda_scale: scaling factor for curvature-to-lambda mapping
            beta_curvature: momentum coefficient for curvature tracking
            hutchinson_samples: number of samples for Hutchinson trace estimator
            disable_regularization: if True, disable R_t term while keeping enhanced gating
            disable_gating: if True, disable enhanced gating while keeping R_t term
        """
        names, params = zip(*params)
        defaults = dict(
            lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma,
            curvature_threshold=curvature_threshold, lambda_max=lambda_max,
            lambda_scale=lambda_scale, beta_curvature=beta_curvature,
            hutchinson_samples=hutchinson_samples, disable_regularization=disable_regularization,
            disable_gating=disable_gating, names=names
        )
        super(InputAwareFirstOrderGlobalUPGD, self).__init__(params, defaults)
        
        # Store current input curvature (will be set from outside)
        self.current_input_curvature = 0.0
        self.avg_input_curvature = 0.0
        self.curvature_step = 0

    def set_input_curvature(self, curvature):
        """Set the current input curvature value (computed externally)."""
        self.current_input_curvature = curvature
        self.curvature_step += 1
        
        # Update running average of curvature
        beta = self.param_groups[0]['beta_curvature']
        self.avg_input_curvature = beta * self.avg_input_curvature + (1 - beta) * curvature
        
    def compute_lambda(self):
        """Compute dynamic regularization strength based on input curvature."""
        if self.curvature_step == 0:
            return 0.0
            
        # Bias correction for running average
        beta = self.param_groups[0]['beta_curvature']
        bias_correction = 1 - beta ** self.curvature_step
        corrected_avg_curvature = self.avg_input_curvature / bias_correction
        
        # Compute lambda using sigmoid mapping
        threshold = self.param_groups[0]['curvature_threshold']
        lambda_max = self.param_groups[0]['lambda_max']
        lambda_scale = self.param_groups[0]['lambda_scale']
        
        # Sigmoid mapping: high curvature -> high lambda
        normalized_curvature = (self.current_input_curvature - threshold) / lambda_scale
        lambda_value = lambda_max * torch.sigmoid(torch.tensor(normalized_curvature)).item()
        
        return lambda_value

    def step(self):
        # First pass: compute global max utility
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["initial_params"] = p.data.clone()  # Store initial params for EWC-like regularization
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        
        # Compute dynamic lambda based on input curvature
        lambda_reg = self.compute_lambda()
        
        # Second pass: update parameters with input-aware protection
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                
                # Compute scaled utility (importance of this parameter)
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                
                # Generate noise for exploration
                noise = torch.randn_like(p.grad) * group["sigma"]
                
                # Compute regularization term (protect important params when input curvature is high)
                # This is the input-aware component: high curvature + high utility = strong protection
                if group["disable_regularization"]:
                    regularization = torch.zeros_like(p.data)
                else:
                    regularization = lambda_reg * scaled_utility * (p.data - state["initial_params"])
                
                # Compute gating factor
                if group["disable_gating"]:
                    gating_factor = 1.0
                else:
                    gating_factor = torch.clamp(1 - scaled_utility * lambda_reg, min=0.0)

                # Update parameters
                # Original UPGD update with added input-aware regularization
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + regularization + noise) * gating_factor,
                    alpha=-1.0*group["lr"]
                )


class InputAwareSecondOrderGlobalUPGD(torch.optim.Optimizer):
    """
    Input-aware second-order UPGD using HesScale for parameter curvature
    and input curvature for gating protection.
    """
    method = HesScale()
    
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.99, sigma=1.0,
                 curvature_threshold=1.0, lambda_max=1.0, lambda_scale=0.1,
                 beta_curvature=0.9, hutchinson_samples=1, disable_regularization=False,
                 disable_gating=False):
        names, params = zip(*params)
        defaults = dict(
            lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma,
            curvature_threshold=curvature_threshold, lambda_max=lambda_max,
            lambda_scale=lambda_scale, beta_curvature=beta_curvature,
            hutchinson_samples=hutchinson_samples, disable_regularization=disable_regularization,
            disable_gating=disable_gating, method_field=type(self).method.savefield, names=names
        )
        super(InputAwareSecondOrderGlobalUPGD, self).__init__(params, defaults)
        
        self.current_input_curvature = 0.0
        self.avg_input_curvature = 0.0
        self.curvature_step = 0

    def set_input_curvature(self, curvature):
        """Set the current input curvature value (computed externally)."""
        self.current_input_curvature = curvature
        self.curvature_step += 1
        
        beta = self.param_groups[0]['beta_curvature']
        self.avg_input_curvature = beta * self.avg_input_curvature + (1 - beta) * curvature
        
    def compute_lambda(self):
        """Compute dynamic regularization strength based on input curvature."""
        if self.curvature_step == 0:
            return 0.0
            
        beta = self.param_groups[0]['beta_curvature']
        bias_correction = 1 - beta ** self.curvature_step
        corrected_avg_curvature = self.avg_input_curvature / bias_correction
        
        threshold = self.param_groups[0]['curvature_threshold']
        lambda_max = self.param_groups[0]['lambda_max']
        lambda_scale = self.param_groups[0]['lambda_scale']
        
        normalized_curvature = (self.current_input_curvature - threshold) / lambda_scale
        lambda_value = lambda_max * torch.sigmoid(torch.tensor(normalized_curvature)).item()
        
        return lambda_value

    def step(self):
        # First pass: compute global max utility using second-order information
        global_max_util = torch.tensor(-torch.inf) # initialize 
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["initial_params"] = p.data.clone()
                state["step"] += 1
                avg_utility = state["avg_utility"]
                
                # Get Hessian diagonal from HesScale
                # Use HesScale's Hessian diagonal for utility computation
                hess_param = getattr(p, group["method_field"])
                #  compute instantaneous utility
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data # u_t = -g_t ⊙ θ_t + 1/2 * H_t ⊙ θ_t^2
                
                # updat EMA (exponential moving average) of utility
                # The .mul_() call scales the running average of utility by beta_utility (EMA decay factor)
                # This is the first step of an exponential moving average update:
                # avg_utility = beta_utility * avg_utility + (1 - beta_utility) * utility
                # U_t = β·U_{t-1} + (1-β)·u_t
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                # update global max across all parameters
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        
        # Compute dynamic lambda
        # Dynamic protection strength based on input curvature
        lambda_reg = self.compute_lambda()
        
        # Second pass: update with input-aware protection
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                noise = torch.randn_like(p.grad) * group["sigma"]
                
                # Input-aware regularization
                # R_t^{(i)} = lambda_reg * scaled_utility * (p.data - state["initial_params"])
                # - λ(x_t) = lambda_reg: Dynamic protection strength based on input curvature
                # - ū_t^{(i)} = scaled_utility: Normalized parameter importance ∈ [0,1]
                # - θ_t^{(i)} = p.data: Current parameter value
                # - θ_0^{(i)} = state["initial_params"]: Initial parameter value
                if group["disable_regularization"]:
                    regularization = torch.zeros_like(p.data)
                else:
                    regularization = lambda_reg * scaled_utility * (p.data - state["initial_params"])
                
                # Compute gating factor
                if group["disable_gating"]:
                    gating_factor = 1.0
                else:
                    gating_factor = torch.clamp(1 - scaled_utility * lambda_reg, min=0.0)

                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + regularization + noise) * gating_factor,
                    alpha=-1.0*group["lr"]
                )
