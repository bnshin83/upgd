import torch
from torch.nn import functional as F
import sys
import os
sys.path.insert(1, os.getcwd())

# HesScale is optional - only needed for second-order methods
try:
    from HesScale.hesscale import HesScale
    HESSCALE_AVAILABLE = True
except ImportError:
    HesScale = None
    HESSCALE_AVAILABLE = False


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
            disable_gating=disable_gating, names=names,
            # New configurable behaviors (backward-compatible defaults)
            lambda_mapping='sigmoid',                 # 'sigmoid' | 'centered_linear' | 'ratio'
            gating_strategy='linear_clamp',           # 'linear_clamp' | 'option_c' | 'option_d'
            gating_option_c_a=1.0,                    # Option C parameter a
            gating_option_c_b=1.0,                    # Option C parameter b
            gating_option_d_alpha=0.5,                # Option D exponent α (controls curvature sensitivity)
            gating_min_g=0.0,                         # Optional lower clamp for gating factor
            scale_blend=1.0                           # Blend: 0.0=lambda_scale, 1.0=auto/tau scale
        )
        super(InputAwareFirstOrderGlobalUPGD, self).__init__(params, defaults)
        
        # Store current input curvature (will be set from outside)
        self.current_input_curvature = 0.0
        self.avg_input_curvature = 0.0
        self.curvature_step = 0

        # Allow environment variables to toggle new behavior without changing scripts
        # Mapping: UPGD_LAMBDA_MAPPING, UPGD_GATING_STRATEGY, UPGD_OPTION_C_A, UPGD_OPTION_C_B, UPGD_OPTION_D_ALPHA, UPGD_MIN_GATING, UPGD_SCALE_BLEND
        try:
            env_lambda_mapping = os.environ.get('UPGD_LAMBDA_MAPPING')
            env_gating_strategy = os.environ.get('UPGD_GATING_STRATEGY')
            env_optc_a = os.environ.get('UPGD_OPTION_C_A')
            env_optc_b = os.environ.get('UPGD_OPTION_C_B')
            env_optd_alpha = os.environ.get('UPGD_OPTION_D_ALPHA')
            env_min_g = os.environ.get('UPGD_MIN_GATING')
            env_scale_blend = os.environ.get('UPGD_SCALE_BLEND')
            if env_lambda_mapping:
                self.param_groups[0]['lambda_mapping'] = env_lambda_mapping
            if env_gating_strategy:
                self.param_groups[0]['gating_strategy'] = env_gating_strategy
            if env_optc_a is not None:
                self.param_groups[0]['gating_option_c_a'] = float(env_optc_a)
            if env_optc_b is not None:
                self.param_groups[0]['gating_option_c_b'] = float(env_optc_b)
            if env_optd_alpha is not None:
                self.param_groups[0]['gating_option_d_alpha'] = float(env_optd_alpha)
            if env_min_g is not None:
                self.param_groups[0]['gating_min_g'] = float(env_min_g)
            if env_scale_blend is not None:
                self.param_groups[0]['scale_blend'] = float(env_scale_blend)
        except Exception:
            # Fail-safe: ignore env parsing errors
            pass

    def set_input_curvature(self, curvature):
        """Set the current input curvature value (computed externally)."""
        self.current_input_curvature = curvature
        self.curvature_step += 1
        
        # Update running average of curvature
        beta = self.param_groups[0]['beta_curvature']
        self.avg_input_curvature = beta * self.avg_input_curvature + (1 - beta) * curvature
    
    # Compute lambda value based on input curvature: λ(x_t) = f(κ(x_t))
    # where κ(x_t) is the input curvature and f is a mapping function
    def compute_lambda(self):
        """Compute dynamic regularization strength based on input curvature.
        
        λ(x_t) = λ_max · σ((κ(x_t) - τ) / s)  [sigmoid mapping]
        λ(x_t) = clamp(1 + (κ(x_t) - τ) / s, 0, λ_max)  [centered_linear mapping]
        
        where:
            κ(x_t) = input curvature at step t
            τ = curvature threshold
            s = lambda_scale
            λ_max = maximum regularization strength
        """
        if self.curvature_step == 0:
            return 0.0
            
        # Bias correction for running average
        beta = self.param_groups[0]['beta_curvature']
        bias_correction = 1 - beta ** self.curvature_step
        corrected_avg_curvature = self.avg_input_curvature / bias_correction
        
        # Compute lambda using selected mapping
        threshold = self.param_groups[0]['curvature_threshold']
        lambda_max = self.param_groups[0]['lambda_max']
        lambda_scale = self.param_groups[0]['lambda_scale']
        mapping = self.param_groups[0].get('lambda_mapping', 'sigmoid')

        # Debug logging (controlled by env var)
        debug_lambda = os.environ.get('UPGD_DEBUG_LAMBDA', '0') == '1'
        if debug_lambda and self.curvature_step <= 100:  # Only first 100 steps
            print(f"[Step {self.curvature_step}] "
                  f"κ_curr={self.current_input_curvature:.6f}, "
                  f"E[κ]={corrected_avg_curvature:.6f}, "
                  f"τ={threshold:.6f}, "
                  f"mapping={mapping}")
        
        normalized_curvature = (self.current_input_curvature - threshold) / lambda_scale
        if mapping == 'centered_linear':
            # Centered linear with cap: λ = clamp(1 + (κ - τ)/s, 0, λ_max)
            lambda_value = torch.clamp(1.0 + torch.tensor(normalized_curvature), min=0.0, max=lambda_max).item()
            if debug_lambda and self.curvature_step <= 100:
                print(f"  → s=lambda_scale={lambda_scale:.6f}, λ={lambda_value:.6f}")
        elif mapping == 'centered_linear_auto_scale':
            # s_auto := |E[κ] - τ| (EMA-corrected), blend with lambda_scale
            eps = 1e-12
            s_auto = max(eps, float(corrected_avg_curvature))
            scale_blend = float(self.param_groups[0].get('scale_blend', 1.0))
            s_eff = max(eps, (1.0 - scale_blend) * float(lambda_scale) + scale_blend * s_auto)
            normalized_auto = (self.current_input_curvature - threshold) / s_eff
            lambda_value = torch.clamp(1.0 + torch.tensor(normalized_auto), min=0.0, max=lambda_max).item()
            if debug_lambda and self.curvature_step <= 100:
                print(f"  → s_auto={s_auto:.6f}, blend={scale_blend:.2f}, "
                      f"s_eff={s_eff:.6f}, λ={lambda_value:.6f}")
        elif mapping == 'centered_linear_tau_norm':
            # Tau-normalized: λ = clamp(1 + (κ - τ)/τ, 0, λ_max)
            # s_tau := τ, blend with lambda_scale
            eps = 1e-12
            s_tau = max(eps, float(threshold))
            scale_blend = float(self.param_groups[0].get('scale_blend', 1.0))
            s_eff = max(eps, (1.0 - scale_blend) * float(lambda_scale) + scale_blend * s_tau)
            normalized_tau = (self.current_input_curvature - threshold) / s_eff
            lambda_value = torch.clamp(1.0 + torch.tensor(normalized_tau), min=0.0, max=lambda_max).item()
            if debug_lambda and self.curvature_step <= 100:
                print(f"  → s_tau={s_tau:.6f}, blend={scale_blend:.2f}, "
                      f"s_eff={s_eff:.6f}, λ={lambda_value:.6f}")
        elif mapping == 'centered_linear_tau_ratio':
            # Simple ratio: λ = κ / τ (parameter-free, centered at τ)
            # When κ = τ: λ = 1, when κ = 2τ: λ = 2, when κ = 0: λ = 0
            eps = 1e-12
            ratio = self.current_input_curvature / max(eps, float(threshold))
            lambda_value = torch.clamp(torch.tensor(ratio), min=0.0, max=lambda_max).item()
            if debug_lambda and self.curvature_step <= 100:
                print(f"  → κ/τ ratio={ratio:.6f}, λ={lambda_value:.6f}")
        elif mapping == 'sigmoid_tau_ratio':
            # Sigmoid with tau-based scale: λ = λ_max · σ((κ - τ) / τ) (parameter-free)
            eps = 1e-12
            normalized_ratio = (self.current_input_curvature - threshold) / max(eps, float(threshold))
            lambda_value = (lambda_max * torch.sigmoid(torch.tensor(normalized_ratio))).item()
            if debug_lambda and self.curvature_step <= 100:
                print(f"  → (κ-τ)/τ={normalized_ratio:.6f}, λ={lambda_value:.6f}")
        elif mapping == 'ratio':
            # Self-normalizing ratio: λ = κ / E[κ] (no threshold needed)
            # When κ = E[κ]: λ = 1 (average curvature, neutral)
            # When κ > E[κ]: λ > 1 (high curvature, more protection)
            # When κ < E[κ]: λ < 1 (low curvature, less protection)
            eps = 1e-12
            lambda_value = self.current_input_curvature / max(eps, float(corrected_avg_curvature))
            # Clamp to [0, lambda_max] for safety
            lambda_value = min(max(0.0, lambda_value), lambda_max)
            if debug_lambda and self.curvature_step <= 100:
                print(f"  → κ/E[κ]={lambda_value:.6f} (E[κ]={corrected_avg_curvature:.6f})")
        else:
            # Default sigmoid mapping: λ = λ_max · σ((κ - τ)/lambda_scale)
            lambda_value = (lambda_max * torch.sigmoid(torch.tensor(normalized_curvature))).item()
            if debug_lambda and self.curvature_step <= 100:
                print(f"  → s=lambda_scale={lambda_scale:.6f}, λ={lambda_value:.6f}")
        
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

        # Store lambda for logging
        self.current_lambda = lambda_reg

        # Add epsilon to prevent division by zero
        global_max_util = torch.max(global_max_util, torch.tensor(1e-8))

        # Second pass: update parameters with input-aware protection and collect statistics
        all_scaled_utilities = []
        all_gradients = []
        all_weights = []
        all_raw_utilities = []

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]

                # Compute scaled utility (importance of this parameter)
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                # Collect for statistics
                all_scaled_utilities.append(scaled_utility.flatten())
                all_gradients.append(p.grad.flatten())
                all_weights.append(p.data.flatten())
                all_raw_utilities.append((state["avg_utility"] / bias_correction).flatten())

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
                    # When input-aware gating is disabled, use standard UPGD gating: (1 - utility)
                    gating_factor = 1.0 - scaled_utility
                else:
                    strategy = group.get('gating_strategy', 'linear_clamp')
                    if strategy == 'option_c':
                        a = float(group.get('gating_option_c_a', 1.0))
                        b = float(group.get('gating_option_c_b', 1.0))
                        min_g = float(group.get('gating_min_g', 0.0))
                        # g = 1 / (1 + a·λ + b·u)
                        gating_factor = 1.0 / (1.0 + a * lambda_reg + b * scaled_utility)
                        gating_factor = torch.clamp(gating_factor, min=min_g, max=1.0)
                    elif strategy == 'option_d':
                        # Option D: g = (1 - u) / λ^α
                        # - When λ = 1 (average curvature): g = 1 - u (original UPGD)
                        # - When λ > 1 (high curvature): g shrinks (more protection)
                        # - When λ < 1 (low curvature): g grows (less protection, clamped to 1)
                        alpha = float(group.get('gating_option_d_alpha', 0.5))
                        min_g = float(group.get('gating_min_g', 0.0))
                        eps = 1e-12
                        # Compute λ^α (safe for λ > 0)
                        lambda_pow = max(eps, lambda_reg) ** alpha
                        gating_factor = (1.0 - scaled_utility) / lambda_pow
                        gating_factor = torch.clamp(gating_factor, min=min_g, max=1.0)
                    else:
                        # Original: g = clamp(1 - u·λ, 0, 1)
                        gating_factor = torch.clamp(1 - scaled_utility * lambda_reg, min=0.0)

                # Update parameters
                # Original UPGD update with added input-aware regularization
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + regularization + noise) * gating_factor,
                    alpha=-1.0*group["lr"]
                )

        # Compute norms and histograms on scaled utilities, gradients, weights, and raw utilities
        if all_scaled_utilities:
            scaled_utility_tensor = torch.cat(all_scaled_utilities)
            gradient_tensor = torch.cat(all_gradients)
            weight_tensor = torch.cat(all_weights)
            raw_utility_tensor = torch.cat(all_raw_utilities)

            total_params = scaled_utility_tensor.numel()

            # Scaled utility statistics
            self.utility_L1 = torch.norm(scaled_utility_tensor, p=1).item()
            self.utility_L2 = torch.norm(scaled_utility_tensor, p=2).item()
            self.utility_L4 = torch.norm(scaled_utility_tensor, p=4).item()
            self.utility_L5 = torch.norm(scaled_utility_tensor, p=5).item()
            self.utility_L10 = torch.norm(scaled_utility_tensor, p=10).item()

            # Scaled utility histogram (5 bins: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0])
            self.utility_hist_0_20 = ((scaled_utility_tensor >= 0.0) & (scaled_utility_tensor < 0.2)).sum().item()
            self.utility_hist_20_40 = ((scaled_utility_tensor >= 0.2) & (scaled_utility_tensor < 0.4)).sum().item()
            self.utility_hist_40_60 = ((scaled_utility_tensor >= 0.4) & (scaled_utility_tensor < 0.6)).sum().item()
            self.utility_hist_60_80 = ((scaled_utility_tensor >= 0.6) & (scaled_utility_tensor < 0.8)).sum().item()
            self.utility_hist_80_100 = ((scaled_utility_tensor >= 0.8) & (scaled_utility_tensor <= 1.0)).sum().item()
            self.utility_hist_0_20_pct = (self.utility_hist_0_20 / total_params) * 100
            self.utility_hist_20_40_pct = (self.utility_hist_20_40 / total_params) * 100
            self.utility_hist_40_60_pct = (self.utility_hist_40_60 / total_params) * 100
            self.utility_hist_60_80_pct = (self.utility_hist_60_80 / total_params) * 100
            self.utility_hist_80_100_pct = (self.utility_hist_80_100 / total_params) * 100
            self.utility_total_params = total_params

            # Gradient histogram (log scale: <1e-4, [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), >=1e-1)
            grad_abs = torch.abs(gradient_tensor)
            self.grad_hist_lt_1e4 = (grad_abs < 1e-4).sum().item()
            self.grad_hist_1e4_1e3 = ((grad_abs >= 1e-4) & (grad_abs < 1e-3)).sum().item()
            self.grad_hist_1e3_1e2 = ((grad_abs >= 1e-3) & (grad_abs < 1e-2)).sum().item()
            self.grad_hist_1e2_1e1 = ((grad_abs >= 1e-2) & (grad_abs < 1e-1)).sum().item()
            self.grad_hist_gte_1e1 = (grad_abs >= 1e-1).sum().item()
            self.grad_hist_lt_1e4_pct = (self.grad_hist_lt_1e4 / total_params) * 100
            self.grad_hist_1e4_1e3_pct = (self.grad_hist_1e4_1e3 / total_params) * 100
            self.grad_hist_1e3_1e2_pct = (self.grad_hist_1e3_1e2 / total_params) * 100
            self.grad_hist_1e2_1e1_pct = (self.grad_hist_1e2_1e1 / total_params) * 100
            self.grad_hist_gte_1e1_pct = (self.grad_hist_gte_1e1 / total_params) * 100

            # Weight histogram (log scale: <1e-4, [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), >=1e-1)
            weight_abs = torch.abs(weight_tensor)
            self.weight_hist_lt_1e4 = (weight_abs < 1e-4).sum().item()
            self.weight_hist_1e4_1e3 = ((weight_abs >= 1e-4) & (weight_abs < 1e-3)).sum().item()
            self.weight_hist_1e3_1e2 = ((weight_abs >= 1e-3) & (weight_abs < 1e-2)).sum().item()
            self.weight_hist_1e2_1e1 = ((weight_abs >= 1e-2) & (weight_abs < 1e-1)).sum().item()
            self.weight_hist_gte_1e1 = (weight_abs >= 1e-1).sum().item()
            self.weight_hist_lt_1e4_pct = (self.weight_hist_lt_1e4 / total_params) * 100
            self.weight_hist_1e4_1e3_pct = (self.weight_hist_1e4_1e3 / total_params) * 100
            self.weight_hist_1e3_1e2_pct = (self.weight_hist_1e3_1e2 / total_params) * 100
            self.weight_hist_1e2_1e1_pct = (self.weight_hist_1e2_1e1 / total_params) * 100
            self.weight_hist_gte_1e1_pct = (self.weight_hist_gte_1e1 / total_params) * 100

            # Raw utility histogram (centered around 0: <-0.001, [-0.001,-0.0002), [-0.0002,0.0002], (0.0002,0.001], >0.001)
            self.raw_util_hist_lt_m001 = (raw_utility_tensor < -0.001).sum().item()
            self.raw_util_hist_m001_m0002 = ((raw_utility_tensor >= -0.001) & (raw_utility_tensor < -0.0002)).sum().item()
            self.raw_util_hist_m0002_p0002 = ((raw_utility_tensor >= -0.0002) & (raw_utility_tensor <= 0.0002)).sum().item()
            self.raw_util_hist_p0002_p001 = ((raw_utility_tensor > 0.0002) & (raw_utility_tensor <= 0.001)).sum().item()
            self.raw_util_hist_gt_p001 = (raw_utility_tensor > 0.001).sum().item()
            self.raw_util_hist_lt_m001_pct = (self.raw_util_hist_lt_m001 / total_params) * 100
            self.raw_util_hist_m001_m0002_pct = (self.raw_util_hist_m001_m0002 / total_params) * 100
            self.raw_util_hist_m0002_p0002_pct = (self.raw_util_hist_m0002_p0002 / total_params) * 100
            self.raw_util_hist_p0002_p001_pct = (self.raw_util_hist_p0002_p001 / total_params) * 100
            self.raw_util_hist_gt_p001_pct = (self.raw_util_hist_gt_p001 / total_params) * 100
        else:
            self.utility_L1 = 0.0
            self.utility_L2 = 0.0
            self.utility_L4 = 0.0
            self.utility_L5 = 0.0
            self.utility_L10 = 0.0
            self.utility_hist_0_20 = 0
            self.utility_hist_20_40 = 0
            self.utility_hist_40_60 = 0
            self.utility_hist_60_80 = 0
            self.utility_hist_80_100 = 0
            self.utility_hist_0_20_pct = 0.0
            self.utility_hist_20_40_pct = 0.0
            self.utility_hist_40_60_pct = 0.0
            self.utility_hist_60_80_pct = 0.0
            self.utility_hist_80_100_pct = 0.0
            self.utility_total_params = 0
            # Gradient histogram defaults
            self.grad_hist_lt_1e4 = 0
            self.grad_hist_1e4_1e3 = 0
            self.grad_hist_1e3_1e2 = 0
            self.grad_hist_1e2_1e1 = 0
            self.grad_hist_gte_1e1 = 0
            self.grad_hist_lt_1e4_pct = 0.0
            self.grad_hist_1e4_1e3_pct = 0.0
            self.grad_hist_1e3_1e2_pct = 0.0
            self.grad_hist_1e2_1e1_pct = 0.0
            self.grad_hist_gte_1e1_pct = 0.0
            # Weight histogram defaults
            self.weight_hist_lt_1e4 = 0
            self.weight_hist_1e4_1e3 = 0
            self.weight_hist_1e3_1e2 = 0
            self.weight_hist_1e2_1e1 = 0
            self.weight_hist_gte_1e1 = 0
            self.weight_hist_lt_1e4_pct = 0.0
            self.weight_hist_1e4_1e3_pct = 0.0
            self.weight_hist_1e3_1e2_pct = 0.0
            self.weight_hist_1e2_1e1_pct = 0.0
            self.weight_hist_gte_1e1_pct = 0.0
            # Raw utility histogram defaults
            self.raw_util_hist_lt_m001 = 0
            self.raw_util_hist_m001_m0002 = 0
            self.raw_util_hist_m0002_p0002 = 0
            self.raw_util_hist_p0002_p001 = 0
            self.raw_util_hist_gt_p001 = 0
            self.raw_util_hist_lt_m001_pct = 0.0
            self.raw_util_hist_m001_m0002_pct = 0.0
            self.raw_util_hist_m0002_p0002_pct = 0.0
            self.raw_util_hist_p0002_p001_pct = 0.0
            self.raw_util_hist_gt_p001_pct = 0.0

        # Store utility statistics for logging
        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util

    def get_utility_stats(self):
        """Return utility statistics for logging."""
        if not hasattr(self, 'global_max_util'):
            return {}

        stats = {
            'utility/global_max': self.global_max_util,
        }

        # Add lambda and input curvature tracking
        if hasattr(self, 'current_lambda'):
            stats['lambda/current'] = self.current_lambda
        if hasattr(self, 'current_input_curvature'):
            stats['curvature/input_current'] = self.current_input_curvature
        if hasattr(self, 'avg_input_curvature') and self.curvature_step > 0:
            beta = self.param_groups[0]['beta_curvature']
            bias_correction = 1 - beta ** self.curvature_step
            stats['curvature/input_avg'] = self.avg_input_curvature / bias_correction

        # Add utility norms if available
        if hasattr(self, 'utility_L1'):
            stats['utility/L1_norm'] = self.utility_L1
            stats['utility/L2_norm'] = self.utility_L2
            stats['utility/L4_norm'] = self.utility_L4
            stats['utility/L5_norm'] = self.utility_L5
            stats['utility/L10_norm'] = self.utility_L10

        # Add utility histogram statistics if available
        if hasattr(self, 'utility_hist_0_20'):
            stats['utility/hist_0_20'] = self.utility_hist_0_20
            stats['utility/hist_20_40'] = self.utility_hist_20_40
            stats['utility/hist_40_60'] = self.utility_hist_40_60
            stats['utility/hist_60_80'] = self.utility_hist_60_80
            stats['utility/hist_80_100'] = self.utility_hist_80_100
            stats['utility/hist_0_20_pct'] = self.utility_hist_0_20_pct
            stats['utility/hist_20_40_pct'] = self.utility_hist_20_40_pct
            stats['utility/hist_40_60_pct'] = self.utility_hist_40_60_pct
            stats['utility/hist_60_80_pct'] = self.utility_hist_60_80_pct
            stats['utility/hist_80_100_pct'] = self.utility_hist_80_100_pct
            stats['utility/total_params'] = self.utility_total_params

        # Add gradient histogram statistics
        if hasattr(self, 'grad_hist_lt_1e4'):
            stats['gradient/hist_lt_1e4'] = self.grad_hist_lt_1e4
            stats['gradient/hist_1e4_1e3'] = self.grad_hist_1e4_1e3
            stats['gradient/hist_1e3_1e2'] = self.grad_hist_1e3_1e2
            stats['gradient/hist_1e2_1e1'] = self.grad_hist_1e2_1e1
            stats['gradient/hist_gte_1e1'] = self.grad_hist_gte_1e1
            stats['gradient/hist_lt_1e4_pct'] = self.grad_hist_lt_1e4_pct
            stats['gradient/hist_1e4_1e3_pct'] = self.grad_hist_1e4_1e3_pct
            stats['gradient/hist_1e3_1e2_pct'] = self.grad_hist_1e3_1e2_pct
            stats['gradient/hist_1e2_1e1_pct'] = self.grad_hist_1e2_1e1_pct
            stats['gradient/hist_gte_1e1_pct'] = self.grad_hist_gte_1e1_pct

        # Add weight histogram statistics
        if hasattr(self, 'weight_hist_lt_1e4'):
            stats['weight/hist_lt_1e4'] = self.weight_hist_lt_1e4
            stats['weight/hist_1e4_1e3'] = self.weight_hist_1e4_1e3
            stats['weight/hist_1e3_1e2'] = self.weight_hist_1e3_1e2
            stats['weight/hist_1e2_1e1'] = self.weight_hist_1e2_1e1
            stats['weight/hist_gte_1e1'] = self.weight_hist_gte_1e1
            stats['weight/hist_lt_1e4_pct'] = self.weight_hist_lt_1e4_pct
            stats['weight/hist_1e4_1e3_pct'] = self.weight_hist_1e4_1e3_pct
            stats['weight/hist_1e3_1e2_pct'] = self.weight_hist_1e3_1e2_pct
            stats['weight/hist_1e2_1e1_pct'] = self.weight_hist_1e2_1e1_pct
            stats['weight/hist_gte_1e1_pct'] = self.weight_hist_gte_1e1_pct

        # Add raw utility histogram statistics
        if hasattr(self, 'raw_util_hist_lt_m001'):
            stats['raw_utility/hist_lt_m001'] = self.raw_util_hist_lt_m001
            stats['raw_utility/hist_m001_m0002'] = self.raw_util_hist_m001_m0002
            stats['raw_utility/hist_m0002_p0002'] = self.raw_util_hist_m0002_p0002
            stats['raw_utility/hist_p0002_p001'] = self.raw_util_hist_p0002_p001
            stats['raw_utility/hist_gt_p001'] = self.raw_util_hist_gt_p001
            stats['raw_utility/hist_lt_m001_pct'] = self.raw_util_hist_lt_m001_pct
            stats['raw_utility/hist_m001_m0002_pct'] = self.raw_util_hist_m001_m0002_pct
            stats['raw_utility/hist_m0002_p0002_pct'] = self.raw_util_hist_m0002_p0002_pct
            stats['raw_utility/hist_p0002_p001_pct'] = self.raw_util_hist_p0002_p001_pct
            stats['raw_utility/hist_gt_p001_pct'] = self.raw_util_hist_gt_p001_pct

        return stats


class InputAwareSecondOrderGlobalUPGD(torch.optim.Optimizer):
    """
    Input-aware second-order UPGD using HesScale for parameter curvature
    and input curvature for gating protection.
    """
    if not HESSCALE_AVAILABLE:
        # Placeholder when HesScale is not available
        def __init__(self, *args, **kwargs):
            raise ImportError("InputAwareSecondOrderGlobalUPGD requires HesScale, which is not compatible with PyTorch 2.x")
    else:
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
            disable_gating=disable_gating, method_field=type(self).method.savefield, names=names,
            # New configurable behaviors (backward-compatible defaults)
            lambda_mapping='sigmoid',                 # 'sigmoid' | 'centered_linear' | 'ratio'
            gating_strategy='linear_clamp',           # 'linear_clamp' | 'option_c' | 'option_d'
            gating_option_c_a=1.0,                    # Option C parameter a
            gating_option_c_b=1.0,                    # Option C parameter b
            gating_option_d_alpha=0.5,                # Option D exponent α (controls curvature sensitivity)
            gating_min_g=0.0,                         # Optional lower clamp for gating factor
            scale_blend=1.0                           # Blend: 0.0=lambda_scale, 1.0=auto/tau scale
        )
        super(InputAwareSecondOrderGlobalUPGD, self).__init__(params, defaults)
        
        self.current_input_curvature = 0.0
        self.avg_input_curvature = 0.0
        self.curvature_step = 0

        # Allow environment variables to toggle new behavior without changing scripts
        try:
            env_lambda_mapping = os.environ.get('UPGD_LAMBDA_MAPPING')
            env_gating_strategy = os.environ.get('UPGD_GATING_STRATEGY')
            env_optc_a = os.environ.get('UPGD_OPTION_C_A')
            env_optc_b = os.environ.get('UPGD_OPTION_C_B')
            env_optd_alpha = os.environ.get('UPGD_OPTION_D_ALPHA')
            env_min_g = os.environ.get('UPGD_MIN_GATING')
            env_scale_blend = os.environ.get('UPGD_SCALE_BLEND')
            if env_lambda_mapping:
                self.param_groups[0]['lambda_mapping'] = env_lambda_mapping
            if env_gating_strategy:
                self.param_groups[0]['gating_strategy'] = env_gating_strategy
            if env_optc_a is not None:
                self.param_groups[0]['gating_option_c_a'] = float(env_optc_a)
            if env_optc_b is not None:
                self.param_groups[0]['gating_option_c_b'] = float(env_optc_b)
            if env_optd_alpha is not None:
                self.param_groups[0]['gating_option_d_alpha'] = float(env_optd_alpha)
            if env_min_g is not None:
                self.param_groups[0]['gating_min_g'] = float(env_min_g)
            if env_scale_blend is not None:
                self.param_groups[0]['scale_blend'] = float(env_scale_blend)
        except Exception:
            pass

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
        mapping = self.param_groups[0].get('lambda_mapping', 'sigmoid')
        
        normalized_curvature = (self.current_input_curvature - threshold) / lambda_scale
        if mapping == 'centered_linear':
            lambda_value = torch.clamp(1.0 + torch.tensor(normalized_curvature), min=0.0, max=lambda_max).item()
        elif mapping == 'centered_linear_auto_scale':
            eps = 1e-12
            s_auto = max(eps, float(corrected_avg_curvature))
            scale_blend = float(self.param_groups[0].get('scale_blend', 1.0))
            s_eff = max(eps, (1.0 - scale_blend) * float(lambda_scale) + scale_blend * s_auto)
            normalized_auto = (self.current_input_curvature - threshold) / s_eff
            lambda_value = torch.clamp(1.0 + torch.tensor(normalized_auto), min=0.0, max=lambda_max).item()
        elif mapping == 'centered_linear_tau_norm':
            eps = 1e-12
            s_tau = max(eps, float(threshold))
            scale_blend = float(self.param_groups[0].get('scale_blend', 1.0))
            s_eff = max(eps, (1.0 - scale_blend) * float(lambda_scale) + scale_blend * s_tau)
            normalized_tau = (self.current_input_curvature - threshold) / s_eff
            lambda_value = torch.clamp(1.0 + torch.tensor(normalized_tau), min=0.0, max=lambda_max).item()
        elif mapping == 'centered_linear_tau_ratio':
            # Simple ratio: λ = κ / τ (parameter-free, centered at τ)
            eps = 1e-12
            ratio = self.current_input_curvature / max(eps, float(threshold))
            lambda_value = torch.clamp(torch.tensor(ratio), min=0.0, max=lambda_max).item()
        elif mapping == 'sigmoid_auto_scale':
            eps = 1e-12
            s_auto = max(eps, float(corrected_avg_curvature))
            scale_blend = float(self.param_groups[0].get('scale_blend', 1.0))
            s_eff = max(eps, (1.0 - scale_blend) * float(lambda_scale) + scale_blend * s_auto)
            normalized_auto = (self.current_input_curvature - threshold) / s_eff
            lambda_value = (lambda_max * torch.sigmoid(torch.tensor(normalized_auto))).item()
        elif mapping == 'sigmoid_tau_ratio':
            # Sigmoid with tau-based scale: λ = λ_max · σ((κ - τ) / τ) (parameter-free)
            eps = 1e-12
            normalized_ratio = (self.current_input_curvature - threshold) / max(eps, float(threshold))
            lambda_value = (lambda_max * torch.sigmoid(torch.tensor(normalized_ratio))).item()
        elif mapping == 'ratio':
            # Self-normalizing ratio: λ = κ / E[κ] (no threshold needed)
            # When κ = E[κ]: λ = 1 (average curvature, neutral)
            # When κ > E[κ]: λ > 1 (high curvature, more protection)
            # When κ < E[κ]: λ < 1 (low curvature, less protection)
            eps = 1e-12
            lambda_value = self.current_input_curvature / max(eps, float(corrected_avg_curvature))
            # Clamp to [0, lambda_max] for safety
            lambda_value = min(max(0.0, lambda_value), lambda_max)
        else:
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

        # Store lambda for logging
        self.current_lambda = lambda_reg

        # Add epsilon to prevent division by zero
        global_max_util = torch.max(global_max_util, torch.tensor(1e-8))

        # Second pass: update with input-aware protection and collect statistics
        all_scaled_utilities = []
        all_gradients = []
        all_weights = []
        all_raw_utilities = []

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]

                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                # Collect for statistics
                all_scaled_utilities.append(scaled_utility.flatten())
                all_gradients.append(p.grad.flatten())
                all_weights.append(p.data.flatten())
                all_raw_utilities.append((state["avg_utility"] / bias_correction).flatten())

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
                    # When input-aware gating is disabled, use standard UPGD gating: (1 - utility)
                    gating_factor = 1.0 - scaled_utility
                else:
                    strategy = group.get('gating_strategy', 'linear_clamp')
                    if strategy == 'option_c':
                        a = float(group.get('gating_option_c_a', 1.0))
                        b = float(group.get('gating_option_c_b', 1.0))
                        min_g = float(group.get('gating_min_g', 0.0))
                        gating_factor = 1.0 / (1.0 + a * lambda_reg + b * scaled_utility)
                        gating_factor = torch.clamp(gating_factor, min=min_g, max=1.0)
                    elif strategy == 'option_d':
                        # Option D: g = (1 - u) / λ^α
                        # - When λ = 1 (average curvature): g = 1 - u (original UPGD)
                        # - When λ > 1 (high curvature): g shrinks (more protection)
                        # - When λ < 1 (low curvature): g grows (less protection, clamped to 1)
                        alpha = float(group.get('gating_option_d_alpha', 0.5))
                        min_g = float(group.get('gating_min_g', 0.0))
                        eps = 1e-12
                        # Compute λ^α (safe for λ > 0)
                        lambda_pow = max(eps, lambda_reg) ** alpha
                        gating_factor = (1.0 - scaled_utility) / lambda_pow
                        gating_factor = torch.clamp(gating_factor, min=min_g, max=1.0)
                    else:
                        gating_factor = torch.clamp(1 - scaled_utility * lambda_reg, min=0.0)

                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + regularization + noise) * gating_factor,
                    alpha=-1.0*group["lr"]
                )

        # Compute norms and histograms on scaled utilities, gradients, weights, and raw utilities
        if all_scaled_utilities:
            scaled_utility_tensor = torch.cat(all_scaled_utilities)
            gradient_tensor = torch.cat(all_gradients)
            weight_tensor = torch.cat(all_weights)
            raw_utility_tensor = torch.cat(all_raw_utilities)

            total_params = scaled_utility_tensor.numel()

            # Scaled utility statistics
            self.utility_L1 = torch.norm(scaled_utility_tensor, p=1).item()
            self.utility_L2 = torch.norm(scaled_utility_tensor, p=2).item()
            self.utility_L4 = torch.norm(scaled_utility_tensor, p=4).item()
            self.utility_L5 = torch.norm(scaled_utility_tensor, p=5).item()
            self.utility_L10 = torch.norm(scaled_utility_tensor, p=10).item()

            # Scaled utility histogram (5 bins: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0])
            self.utility_hist_0_20 = ((scaled_utility_tensor >= 0.0) & (scaled_utility_tensor < 0.2)).sum().item()
            self.utility_hist_20_40 = ((scaled_utility_tensor >= 0.2) & (scaled_utility_tensor < 0.4)).sum().item()
            self.utility_hist_40_60 = ((scaled_utility_tensor >= 0.4) & (scaled_utility_tensor < 0.6)).sum().item()
            self.utility_hist_60_80 = ((scaled_utility_tensor >= 0.6) & (scaled_utility_tensor < 0.8)).sum().item()
            self.utility_hist_80_100 = ((scaled_utility_tensor >= 0.8) & (scaled_utility_tensor <= 1.0)).sum().item()
            self.utility_hist_0_20_pct = (self.utility_hist_0_20 / total_params) * 100
            self.utility_hist_20_40_pct = (self.utility_hist_20_40 / total_params) * 100
            self.utility_hist_40_60_pct = (self.utility_hist_40_60 / total_params) * 100
            self.utility_hist_60_80_pct = (self.utility_hist_60_80 / total_params) * 100
            self.utility_hist_80_100_pct = (self.utility_hist_80_100 / total_params) * 100
            self.utility_total_params = total_params

            # Gradient histogram (log scale: <1e-4, [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), >=1e-1)
            grad_abs = torch.abs(gradient_tensor)
            self.grad_hist_lt_1e4 = (grad_abs < 1e-4).sum().item()
            self.grad_hist_1e4_1e3 = ((grad_abs >= 1e-4) & (grad_abs < 1e-3)).sum().item()
            self.grad_hist_1e3_1e2 = ((grad_abs >= 1e-3) & (grad_abs < 1e-2)).sum().item()
            self.grad_hist_1e2_1e1 = ((grad_abs >= 1e-2) & (grad_abs < 1e-1)).sum().item()
            self.grad_hist_gte_1e1 = (grad_abs >= 1e-1).sum().item()
            self.grad_hist_lt_1e4_pct = (self.grad_hist_lt_1e4 / total_params) * 100
            self.grad_hist_1e4_1e3_pct = (self.grad_hist_1e4_1e3 / total_params) * 100
            self.grad_hist_1e3_1e2_pct = (self.grad_hist_1e3_1e2 / total_params) * 100
            self.grad_hist_1e2_1e1_pct = (self.grad_hist_1e2_1e1 / total_params) * 100
            self.grad_hist_gte_1e1_pct = (self.grad_hist_gte_1e1 / total_params) * 100

            # Weight histogram (log scale: <1e-4, [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), >=1e-1)
            weight_abs = torch.abs(weight_tensor)
            self.weight_hist_lt_1e4 = (weight_abs < 1e-4).sum().item()
            self.weight_hist_1e4_1e3 = ((weight_abs >= 1e-4) & (weight_abs < 1e-3)).sum().item()
            self.weight_hist_1e3_1e2 = ((weight_abs >= 1e-3) & (weight_abs < 1e-2)).sum().item()
            self.weight_hist_1e2_1e1 = ((weight_abs >= 1e-2) & (weight_abs < 1e-1)).sum().item()
            self.weight_hist_gte_1e1 = (weight_abs >= 1e-1).sum().item()
            self.weight_hist_lt_1e4_pct = (self.weight_hist_lt_1e4 / total_params) * 100
            self.weight_hist_1e4_1e3_pct = (self.weight_hist_1e4_1e3 / total_params) * 100
            self.weight_hist_1e3_1e2_pct = (self.weight_hist_1e3_1e2 / total_params) * 100
            self.weight_hist_1e2_1e1_pct = (self.weight_hist_1e2_1e1 / total_params) * 100
            self.weight_hist_gte_1e1_pct = (self.weight_hist_gte_1e1 / total_params) * 100

            # Raw utility histogram (centered around 0: <-0.001, [-0.001,-0.0002), [-0.0002,0.0002], (0.0002,0.001], >0.001)
            self.raw_util_hist_lt_m001 = (raw_utility_tensor < -0.001).sum().item()
            self.raw_util_hist_m001_m0002 = ((raw_utility_tensor >= -0.001) & (raw_utility_tensor < -0.0002)).sum().item()
            self.raw_util_hist_m0002_p0002 = ((raw_utility_tensor >= -0.0002) & (raw_utility_tensor <= 0.0002)).sum().item()
            self.raw_util_hist_p0002_p001 = ((raw_utility_tensor > 0.0002) & (raw_utility_tensor <= 0.001)).sum().item()
            self.raw_util_hist_gt_p001 = (raw_utility_tensor > 0.001).sum().item()
            self.raw_util_hist_lt_m001_pct = (self.raw_util_hist_lt_m001 / total_params) * 100
            self.raw_util_hist_m001_m0002_pct = (self.raw_util_hist_m001_m0002 / total_params) * 100
            self.raw_util_hist_m0002_p0002_pct = (self.raw_util_hist_m0002_p0002 / total_params) * 100
            self.raw_util_hist_p0002_p001_pct = (self.raw_util_hist_p0002_p001 / total_params) * 100
            self.raw_util_hist_gt_p001_pct = (self.raw_util_hist_gt_p001 / total_params) * 100
        else:
            self.utility_L1 = 0.0
            self.utility_L2 = 0.0
            self.utility_L4 = 0.0
            self.utility_L5 = 0.0
            self.utility_L10 = 0.0
            self.utility_hist_0_20 = 0
            self.utility_hist_20_40 = 0
            self.utility_hist_40_60 = 0
            self.utility_hist_60_80 = 0
            self.utility_hist_80_100 = 0
            self.utility_hist_0_20_pct = 0.0
            self.utility_hist_20_40_pct = 0.0
            self.utility_hist_40_60_pct = 0.0
            self.utility_hist_60_80_pct = 0.0
            self.utility_hist_80_100_pct = 0.0
            self.utility_total_params = 0
            # Gradient histogram defaults
            self.grad_hist_lt_1e4 = 0
            self.grad_hist_1e4_1e3 = 0
            self.grad_hist_1e3_1e2 = 0
            self.grad_hist_1e2_1e1 = 0
            self.grad_hist_gte_1e1 = 0
            self.grad_hist_lt_1e4_pct = 0.0
            self.grad_hist_1e4_1e3_pct = 0.0
            self.grad_hist_1e3_1e2_pct = 0.0
            self.grad_hist_1e2_1e1_pct = 0.0
            self.grad_hist_gte_1e1_pct = 0.0
            # Weight histogram defaults
            self.weight_hist_lt_1e4 = 0
            self.weight_hist_1e4_1e3 = 0
            self.weight_hist_1e3_1e2 = 0
            self.weight_hist_1e2_1e1 = 0
            self.weight_hist_gte_1e1 = 0
            self.weight_hist_lt_1e4_pct = 0.0
            self.weight_hist_1e4_1e3_pct = 0.0
            self.weight_hist_1e3_1e2_pct = 0.0
            self.weight_hist_1e2_1e1_pct = 0.0
            self.weight_hist_gte_1e1_pct = 0.0
            # Raw utility histogram defaults
            self.raw_util_hist_lt_m001 = 0
            self.raw_util_hist_m001_m0002 = 0
            self.raw_util_hist_m0002_p0002 = 0
            self.raw_util_hist_p0002_p001 = 0
            self.raw_util_hist_gt_p001 = 0
            self.raw_util_hist_lt_m001_pct = 0.0
            self.raw_util_hist_m001_m0002_pct = 0.0
            self.raw_util_hist_m0002_p0002_pct = 0.0
            self.raw_util_hist_p0002_p001_pct = 0.0
            self.raw_util_hist_gt_p001_pct = 0.0

        # Store utility statistics for logging (second-order)
        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util

    def get_utility_stats(self):
        """Return utility statistics for logging."""
        if not hasattr(self, 'global_max_util'):
            return {}

        stats = {
            'utility/global_max': self.global_max_util,
        }

        # Add lambda and input curvature tracking
        if hasattr(self, 'current_lambda'):
            stats['lambda/current'] = self.current_lambda
        if hasattr(self, 'current_input_curvature'):
            stats['curvature/input_current'] = self.current_input_curvature
        if hasattr(self, 'avg_input_curvature') and self.curvature_step > 0:
            beta = self.param_groups[0]['beta_curvature']
            bias_correction = 1 - beta ** self.curvature_step
            stats['curvature/input_avg'] = self.avg_input_curvature / bias_correction

        # Add utility norms if available
        if hasattr(self, 'utility_L1'):
            stats['utility/L1_norm'] = self.utility_L1
            stats['utility/L2_norm'] = self.utility_L2
            stats['utility/L4_norm'] = self.utility_L4
            stats['utility/L5_norm'] = self.utility_L5
            stats['utility/L10_norm'] = self.utility_L10

        # Add utility histogram statistics if available
        if hasattr(self, 'utility_hist_0_20'):
            stats['utility/hist_0_20'] = self.utility_hist_0_20
            stats['utility/hist_20_40'] = self.utility_hist_20_40
            stats['utility/hist_40_60'] = self.utility_hist_40_60
            stats['utility/hist_60_80'] = self.utility_hist_60_80
            stats['utility/hist_80_100'] = self.utility_hist_80_100
            stats['utility/hist_0_20_pct'] = self.utility_hist_0_20_pct
            stats['utility/hist_20_40_pct'] = self.utility_hist_20_40_pct
            stats['utility/hist_40_60_pct'] = self.utility_hist_40_60_pct
            stats['utility/hist_60_80_pct'] = self.utility_hist_60_80_pct
            stats['utility/hist_80_100_pct'] = self.utility_hist_80_100_pct
            stats['utility/total_params'] = self.utility_total_params

        # Add gradient histogram statistics
        if hasattr(self, 'grad_hist_lt_1e4'):
            stats['gradient/hist_lt_1e4'] = self.grad_hist_lt_1e4
            stats['gradient/hist_1e4_1e3'] = self.grad_hist_1e4_1e3
            stats['gradient/hist_1e3_1e2'] = self.grad_hist_1e3_1e2
            stats['gradient/hist_1e2_1e1'] = self.grad_hist_1e2_1e1
            stats['gradient/hist_gte_1e1'] = self.grad_hist_gte_1e1
            stats['gradient/hist_lt_1e4_pct'] = self.grad_hist_lt_1e4_pct
            stats['gradient/hist_1e4_1e3_pct'] = self.grad_hist_1e4_1e3_pct
            stats['gradient/hist_1e3_1e2_pct'] = self.grad_hist_1e3_1e2_pct
            stats['gradient/hist_1e2_1e1_pct'] = self.grad_hist_1e2_1e1_pct
            stats['gradient/hist_gte_1e1_pct'] = self.grad_hist_gte_1e1_pct

        # Add weight histogram statistics
        if hasattr(self, 'weight_hist_lt_1e4'):
            stats['weight/hist_lt_1e4'] = self.weight_hist_lt_1e4
            stats['weight/hist_1e4_1e3'] = self.weight_hist_1e4_1e3
            stats['weight/hist_1e3_1e2'] = self.weight_hist_1e3_1e2
            stats['weight/hist_1e2_1e1'] = self.weight_hist_1e2_1e1
            stats['weight/hist_gte_1e1'] = self.weight_hist_gte_1e1
            stats['weight/hist_lt_1e4_pct'] = self.weight_hist_lt_1e4_pct
            stats['weight/hist_1e4_1e3_pct'] = self.weight_hist_1e4_1e3_pct
            stats['weight/hist_1e3_1e2_pct'] = self.weight_hist_1e3_1e2_pct
            stats['weight/hist_1e2_1e1_pct'] = self.weight_hist_1e2_1e1_pct
            stats['weight/hist_gte_1e1_pct'] = self.weight_hist_gte_1e1_pct

        # Add raw utility histogram statistics
        if hasattr(self, 'raw_util_hist_lt_m001'):
            stats['raw_utility/hist_lt_m001'] = self.raw_util_hist_lt_m001
            stats['raw_utility/hist_m001_m0002'] = self.raw_util_hist_m001_m0002
            stats['raw_utility/hist_m0002_p0002'] = self.raw_util_hist_m0002_p0002
            stats['raw_utility/hist_p0002_p001'] = self.raw_util_hist_p0002_p001
            stats['raw_utility/hist_gt_p001'] = self.raw_util_hist_gt_p001
            stats['raw_utility/hist_lt_m001_pct'] = self.raw_util_hist_lt_m001_pct
            stats['raw_utility/hist_m001_m0002_pct'] = self.raw_util_hist_m001_m0002_pct
            stats['raw_utility/hist_m0002_p0002_pct'] = self.raw_util_hist_m0002_p0002_pct
            stats['raw_utility/hist_p0002_p001_pct'] = self.raw_util_hist_p0002_p001_pct
            stats['raw_utility/hist_gt_p001_pct'] = self.raw_util_hist_gt_p001_pct

        return stats
