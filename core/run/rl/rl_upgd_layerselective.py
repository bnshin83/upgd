"""
Layer-Selective Adaptive UPGD Optimizer for RL (Actor-Critic Networks)

Supports different gating modes:
- 'full': Apply gating to all layers (standard UPGD)
- 'output_only': Apply gating only to output layers (actor_mean.4, critic.4)
- 'hidden_only': Apply gating only to hidden layers (actor_mean.0/2, critic.0/2)
- 'actor_output_only': Apply gating only to actor output (actor_mean.4)
- 'critic_output_only': Apply gating only to critic output (critic.4)
"""

import torch


class RLLayerSelectiveUPGD(torch.optim.Optimizer):
    """
    Adaptive UPGD with layer-selective gating for actor-critic networks.
    
    Args:
        params: Iterator of (name, param) tuples from model.named_parameters()
        lr: Learning rate
        weight_decay: Weight decay coefficient
        beta_utility: Decay rate for utility tracking (0.999 default)
        sigma: Noise scale for perturbation
        beta1: First moment decay (Adam-style)
        beta2: Second moment decay (Adam-style)
        eps: Epsilon for numerical stability
        gating_mode: One of 'full', 'output_only', 'hidden_only', 
                     'actor_output_only', 'critic_output_only'
        non_gated_scale: Scaling factor for non-gated layers (default 0.5)
    """
    
    def __init__(self, params, lr=1e-5, weight_decay=0.001, beta_utility=0.999, 
                 sigma=0.001, beta1=0.9, beta2=0.999, eps=1e-5,
                 gating_mode='full', non_gated_scale=0.5):
        # Store parameter names
        names, params = zip(*params)
        defaults = dict(
            lr=lr, weight_decay=weight_decay, beta_utility=beta_utility,
            sigma=sigma, beta1=beta1, beta2=beta2, eps=eps,
            names=names, gating_mode=gating_mode, non_gated_scale=non_gated_scale
        )
        super(RLLayerSelectiveUPGD, self).__init__(params, defaults)
    
    def _should_apply_gating(self, param_name, gating_mode):
        """Determine if utility gating should be applied to this parameter."""
        if gating_mode == 'full':
            return True
        elif gating_mode == 'output_only':
            # Output layers: actor_mean.4, critic.4
            return 'actor_mean.4' in param_name or 'critic.4' in param_name
        elif gating_mode == 'hidden_only':
            # Hidden layers: actor_mean.0/2, critic.0/2
            return ('actor_mean.0' in param_name or 'actor_mean.2' in param_name or
                    'critic.0' in param_name or 'critic.2' in param_name)
        elif gating_mode == 'actor_output_only':
            return 'actor_mean.4' in param_name
        elif gating_mode == 'critic_output_only':
            return 'critic.4' in param_name
        else:
            raise ValueError(f"Unknown gating_mode: {gating_mode}")
    
    def step(self):
        # First pass: compute global max utility and collect utilities
        global_max_util = torch.tensor(-torch.inf)
        
        # Storage for per-layer statistics
        layer_utilities = {}
        layer_gating_applied = {}
        all_scaled_utilities = []
        
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["sec_moment"] = torch.zeros_like(p.data)
                    state["name"] = name
                
                state["step"] += 1
                
                # Update utility estimate
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                
                # Update Adam moments
                first_moment, sec_moment = state["first_moment"], state["sec_moment"]
                first_moment.mul_(group["beta1"]).add_(p.grad.data, alpha=1 - group["beta1"])
                sec_moment.mul_(group["beta2"]).add_(p.grad.data ** 2, alpha=1 - group["beta2"])
                
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        
        global_max_util = torch.max(global_max_util, torch.tensor(1e-8))
        
        # Second pass: update parameters and collect statistics
        for group in self.param_groups:
            gating_mode = group["gating_mode"]
            non_gated_scale = group["non_gated_scale"]
            
            for name, p in zip(group["names"], group["params"]):
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                
                # Bias corrections
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]
                bias_correction_beta1 = 1 - group["beta1"] ** state["step"]
                bias_correction_beta2 = 1 - group["beta2"] ** state["step"]
                
                exp_avg = state["first_moment"] / bias_correction_beta1
                exp_avg_sq = state["sec_moment"] / bias_correction_beta2
                
                noise = torch.randn_like(p.grad) * group["sigma"]
                
                # Compute scaled utility for statistics
                scaled_utility = torch.sigmoid(
                    (state["avg_utility"] / bias_correction_utility) / global_max_util
                )
                
                # Collect per-layer utilities
                layer_name = name.rsplit('.', 1)[0] if '.' in name else name
                if layer_name not in layer_utilities:
                    layer_utilities[layer_name] = []
                    layer_gating_applied[layer_name] = self._should_apply_gating(name, gating_mode)
                layer_utilities[layer_name].append(scaled_utility.flatten())
                all_scaled_utilities.append(scaled_utility.flatten())
                
                if self._should_apply_gating(name, gating_mode):
                    # Apply utility gating (in-place sigmoid already computed, recompute for update)
                    scaled_utility_update = torch.sigmoid_(
                        (state["avg_utility"] / bias_correction_utility) / global_max_util
                    )
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (exp_avg * (1 - scaled_utility_update)) / (exp_avg_sq.sqrt() + group["eps"]) + noise * (1 - scaled_utility_update),
                        alpha=-2.0 * group["lr"],
                    )
                else:
                    # Use fixed scaling (no utility gating)
                    if non_gated_scale > 0.0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                            (exp_avg * non_gated_scale) / (exp_avg_sq.sqrt() + group["eps"]) + noise * non_gated_scale,
                            alpha=-2.0 * group["lr"],
                        )
        
        # Store for statistics
        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util
        self._layer_utilities = layer_utilities
        self._layer_gating_applied = layer_gating_applied
        self._all_scaled_utilities = all_scaled_utilities
    
    def get_utility_stats(self):
        """Return comprehensive utility statistics for logging."""
        stats = {'utility/global_max': getattr(self, 'global_max_util', 0.0)}
        
        gated_count = 0
        non_gated_count = 0
        
        for group in self.param_groups:
            gating_mode = group["gating_mode"]
            for name, p in zip(group["names"], group["params"]):
                if p.grad is None:
                    continue
                if self._should_apply_gating(name, gating_mode):
                    gated_count += p.numel()
                else:
                    non_gated_count += p.numel()
        
        stats['utility/gated_params'] = gated_count
        stats['utility/non_gated_params'] = non_gated_count
        
        # Global histogram (9 bins)
        if hasattr(self, '_all_scaled_utilities') and self._all_scaled_utilities:
            all_utils = torch.cat(self._all_scaled_utilities)
            total = all_utils.numel()
            
            stats['utility/hist_0_20_pct'] = ((all_utils >= 0.0) & (all_utils < 0.2)).sum().item() / total * 100
            stats['utility/hist_20_40_pct'] = ((all_utils >= 0.2) & (all_utils < 0.4)).sum().item() / total * 100
            stats['utility/hist_40_48_pct'] = ((all_utils >= 0.4) & (all_utils < 0.48)).sum().item() / total * 100
            stats['utility/hist_48_52_pct'] = ((all_utils >= 0.48) & (all_utils < 0.52)).sum().item() / total * 100
            stats['utility/hist_52_60_pct'] = ((all_utils >= 0.52) & (all_utils < 0.6)).sum().item() / total * 100
            stats['utility/hist_60_80_pct'] = ((all_utils >= 0.6) & (all_utils < 0.8)).sum().item() / total * 100
            stats['utility/hist_80_100_pct'] = ((all_utils >= 0.8) & (all_utils <= 1.0)).sum().item() / total * 100
            stats['utility/mean'] = all_utils.mean().item()
            stats['utility/std'] = all_utils.std().item()
        
        # Per-layer histograms
        if hasattr(self, '_layer_utilities'):
            for layer_name, util_list in self._layer_utilities.items():
                layer_utils = torch.cat(util_list)
                layer_total = layer_utils.numel()
                gated = self._layer_gating_applied.get(layer_name, False)
                
                prefix = f"layer/{layer_name}"
                stats[f"{prefix}/gating_applied"] = 1.0 if gated else 0.0
                stats[f"{prefix}/mean"] = layer_utils.mean().item()
                stats[f"{prefix}/std"] = layer_utils.std().item()
                stats[f"{prefix}/min"] = layer_utils.min().item()
                stats[f"{prefix}/max"] = layer_utils.max().item()
                
                # Per-layer histogram bins
                stats[f"{prefix}/hist_0_20_pct"] = ((layer_utils >= 0.0) & (layer_utils < 0.2)).sum().item() / layer_total * 100
                stats[f"{prefix}/hist_20_40_pct"] = ((layer_utils >= 0.2) & (layer_utils < 0.4)).sum().item() / layer_total * 100
                stats[f"{prefix}/hist_40_48_pct"] = ((layer_utils >= 0.4) & (layer_utils < 0.48)).sum().item() / layer_total * 100
                stats[f"{prefix}/hist_48_52_pct"] = ((layer_utils >= 0.48) & (layer_utils < 0.52)).sum().item() / layer_total * 100
                stats[f"{prefix}/hist_52_60_pct"] = ((layer_utils >= 0.52) & (layer_utils < 0.6)).sum().item() / layer_total * 100
                stats[f"{prefix}/hist_60_80_pct"] = ((layer_utils >= 0.6) & (layer_utils < 0.8)).sum().item() / layer_total * 100
                stats[f"{prefix}/hist_80_100_pct"] = ((layer_utils >= 0.8) & (layer_utils <= 1.0)).sum().item() / layer_total * 100
        
        return stats

