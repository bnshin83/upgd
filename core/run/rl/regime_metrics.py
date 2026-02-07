"""
Regime Detection Metrics for RL Training

Measures input-shift vs target-shift regime characteristics:
- State distribution shift (input shift indicator)
- TD-error volatility (target shift indicator)
- Policy divergence (exploration indicator)
- Layer-wise adaptation rates

Created: 2026-02-07
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque


class RegimeMetrics:
    """Track and compute regime detection metrics during RL training."""
    
    def __init__(self, window_size: int = 10, device: str = 'cuda'):
        self.window_size = window_size
        self.device = device
        
        # Rolling buffers for computing shifts
        self.obs_stats_buffer = deque(maxlen=window_size)
        self.td_error_buffer = deque(maxlen=window_size)
        self.policy_kl_buffer = deque(maxlen=window_size)
        self.action_entropy_buffer = deque(maxlen=window_size)
        
        # Previous batch stats for shift computation
        self.prev_obs_mean = None
        self.prev_obs_std = None
        self.prev_value_mean = None
        
        # Per-layer gradient tracking
        self.layer_grad_history = {}
        
    def update_observation_stats(self, obs: torch.Tensor) -> Dict[str, float]:
        """
        Update observation statistics and compute state distribution shift.
        
        Args:
            obs: Batch of observations [batch_size, obs_dim]
        
        Returns:
            Dictionary with observation shift metrics
        """
        stats = {}
        
        with torch.no_grad():
            obs_mean = obs.mean(dim=0)
            obs_std = obs.std(dim=0) + 1e-8
            
            # Store raw stats
            stats['obs/mean_norm'] = obs_mean.norm().item()
            stats['obs/std_mean'] = obs_std.mean().item()
            stats['obs/std_min'] = obs_std.min().item()
            stats['obs/std_max'] = obs_std.max().item()
            
            # Compute shift from previous batch
            if self.prev_obs_mean is not None:
                # Mean shift (L2 distance)
                mean_shift = (obs_mean - self.prev_obs_mean).norm().item()
                stats['regime/obs_mean_shift'] = mean_shift
                
                # Distribution shift (approximate KL via moment matching)
                # KL(N(μ1,σ1) || N(μ2,σ2)) ≈ log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 0.5
                std_ratio = (self.prev_obs_std / obs_std).log().mean().item()
                var_term = ((self.prev_obs_std**2 + (self.prev_obs_mean - obs_mean)**2) / (2 * obs_std**2)).mean().item()
                approx_kl = std_ratio + var_term - 0.5
                stats['regime/obs_kl_shift'] = max(0, approx_kl)  # Clamp to non-negative
                
                # Track in buffer
                self.obs_stats_buffer.append({
                    'mean_shift': mean_shift,
                    'kl_shift': max(0, approx_kl)
                })
            
            # Update previous stats
            self.prev_obs_mean = obs_mean.clone()
            self.prev_obs_std = obs_std.clone()
            
            # Compute rolling statistics if enough samples
            if len(self.obs_stats_buffer) >= 3:
                mean_shifts = [x['mean_shift'] for x in self.obs_stats_buffer]
                kl_shifts = [x['kl_shift'] for x in self.obs_stats_buffer]
                stats['regime/obs_mean_shift_avg'] = np.mean(mean_shifts)
                stats['regime/obs_kl_shift_avg'] = np.mean(kl_shifts)
                stats['regime/input_shift_intensity'] = np.mean(kl_shifts)  # Main input shift indicator
        
        return stats
    
    def update_td_error_stats(self, td_errors: torch.Tensor, values: torch.Tensor) -> Dict[str, float]:
        """
        Update TD-error statistics to measure target shift.
        
        Args:
            td_errors: TD errors [batch_size]
            values: Predicted values [batch_size]
        
        Returns:
            Dictionary with target shift metrics
        """
        stats = {}
        
        with torch.no_grad():
            td_var = td_errors.var().item()
            td_mean = td_errors.mean().item()
            td_abs_mean = td_errors.abs().mean().item()
            value_mean = values.mean().item()
            value_std = values.std().item()
            
            stats['td/variance'] = td_var
            stats['td/mean'] = td_mean
            stats['td/abs_mean'] = td_abs_mean
            stats['value/mean'] = value_mean
            stats['value/std'] = value_std
            
            # Compute value shift
            if self.prev_value_mean is not None:
                value_shift = abs(value_mean - self.prev_value_mean)
                stats['regime/value_shift'] = value_shift
            
            self.prev_value_mean = value_mean
            
            # Track TD error volatility
            self.td_error_buffer.append(td_var)
            
            if len(self.td_error_buffer) >= 3:
                td_vars = list(self.td_error_buffer)
                stats['regime/td_var_avg'] = np.mean(td_vars)
                stats['regime/td_var_trend'] = td_vars[-1] - td_vars[0]  # Increasing or decreasing
                stats['regime/target_shift_intensity'] = np.mean(td_vars)  # Main target shift indicator
        
        return stats
    
    def update_policy_stats(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                           entropy: torch.Tensor) -> Dict[str, float]:
        """
        Update policy divergence statistics.
        
        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            entropy: Policy entropy
        
        Returns:
            Dictionary with policy shift metrics
        """
        stats = {}
        
        with torch.no_grad():
            # Approximate KL divergence
            approx_kl = (old_log_probs - log_probs).mean().item()
            stats['policy/approx_kl'] = approx_kl
            
            # Entropy
            entropy_val = entropy.mean().item()
            stats['policy/entropy'] = entropy_val
            
            # Track in buffers
            self.policy_kl_buffer.append(approx_kl)
            self.action_entropy_buffer.append(entropy_val)
            
            if len(self.policy_kl_buffer) >= 3:
                kls = list(self.policy_kl_buffer)
                entropies = list(self.action_entropy_buffer)
                stats['regime/policy_kl_avg'] = np.mean(kls)
                stats['regime/entropy_avg'] = np.mean(entropies)
                stats['regime/entropy_trend'] = entropies[-1] - entropies[0]
        
        return stats
    
    def update_layer_gradients(self, named_parameters) -> Dict[str, float]:
        """
        Track per-layer gradient statistics for adaptation analysis.
        
        Args:
            named_parameters: Iterator of (name, param) tuples
        
        Returns:
            Dictionary with layer-wise gradient metrics
        """
        stats = {}
        hidden_grad_norm = 0.0
        output_grad_norm = 0.0
        hidden_count = 0
        output_count = 0
        
        for name, param in named_parameters:
            if param.grad is None:
                continue
                
            grad_norm = param.grad.norm(2).item()
            weight_norm = param.norm(2).item()
            relative_grad = grad_norm / (weight_norm + 1e-8)
            
            stats[f'grad_norm/{name}'] = grad_norm
            stats[f'relative_grad/{name}'] = relative_grad
            
            # Categorize as hidden or output
            if '.0.' in name or '.2.' in name:  # Hidden layers
                hidden_grad_norm += grad_norm
                hidden_count += 1
            elif '.4.' in name:  # Output layer
                output_grad_norm += grad_norm
                output_count += 1
        
        if hidden_count > 0:
            stats['regime/hidden_grad_avg'] = hidden_grad_norm / hidden_count
        if output_count > 0:
            stats['regime/output_grad_avg'] = output_grad_norm / output_count
        
        if hidden_count > 0 and output_count > 0:
            # Ratio indicates which layers are adapting more
            stats['regime/hidden_output_grad_ratio'] = (hidden_grad_norm / hidden_count) / (output_grad_norm / output_count + 1e-8)
        
        return stats
    
    def compute_regime_indicator(self) -> Dict[str, float]:
        """
        Compute overall regime indicator based on collected metrics.
        
        Returns:
            regime/indicator > 0: input shift dominant (favor hidden_only)
            regime/indicator < 0: target shift dominant (favor output_only)
        """
        stats = {}
        
        if len(self.obs_stats_buffer) < 3 or len(self.td_error_buffer) < 3:
            return stats
        
        # Get average shifts
        input_shift = np.mean([x['kl_shift'] for x in self.obs_stats_buffer])
        target_shift = np.mean(list(self.td_error_buffer))
        
        # Normalize (rough heuristic, may need tuning)
        input_norm = input_shift / (input_shift + 0.1)  # 0 to ~1
        target_norm = target_shift / (target_shift + 1.0)  # 0 to ~1
        
        # Indicator: positive = input shift dominant
        indicator = input_norm - target_norm
        stats['regime/indicator'] = indicator
        stats['regime/input_normalized'] = input_norm
        stats['regime/target_normalized'] = target_norm
        
        # Suggested gating
        if indicator > 0.1:
            stats['regime/suggested_gating'] = 1  # hidden_only
        elif indicator < -0.1:
            stats['regime/suggested_gating'] = -1  # output_only
        else:
            stats['regime/suggested_gating'] = 0  # mixed/full
        
        return stats


def create_regime_tracker(device: str = 'cuda') -> RegimeMetrics:
    """Factory function to create regime tracker."""
    return RegimeMetrics(window_size=10, device=device)
