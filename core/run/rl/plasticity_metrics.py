"""
Plasticity Metrics for RL Networks

Computes standard plasticity/loss-of-plasticity metrics:
- Dead neurons: neurons with near-zero activation
- Stable rank (srank): ratio of trace to max eigenvalue of covariance
- Effective rank: based on entropy of normalized eigenvalues
- Weight statistics: norms, magnitudes

Based on loss-of-plasticity Nature paper (Dohare et al., 2024)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class PlasticityMetrics:
    """Compute and track plasticity metrics for neural networks."""
    
    def __init__(self, network: nn.Module, device: str = 'cuda'):
        self.network = network
        self.device = device
        self.activation_buffers = {}  # Store activations for dead neuron detection
        self.hooks = []
        
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_buffers[name] = output.detach()
            return hook
        
        for name, module in self.network.named_modules():
            if layer_names is None or name in layer_names:
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    self.hooks.append(module.register_forward_hook(make_hook(name)))
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_dead_neurons(self, threshold: float = 0.01) -> Dict[str, float]:
        """
        Compute fraction of dead neurons per layer.
        A neuron is considered dead if its mean absolute activation < threshold.
        """
        stats = {}
        total_dead = 0
        total_neurons = 0
        
        for name, activations in self.activation_buffers.items():
            if activations.dim() >= 2:
                # Mean over batch and spatial dims, keep neurons
                mean_act = activations.abs().mean(dim=0)
                while mean_act.dim() > 1:
                    mean_act = mean_act.mean(dim=-1)
                
                n_neurons = mean_act.numel()
                n_dead = (mean_act < threshold).sum().item()
                
                stats[f"dead_neurons/{name}"] = n_dead / n_neurons if n_neurons > 0 else 0.0
                total_dead += n_dead
                total_neurons += n_neurons
        
        stats["dead_neurons/total_ratio"] = total_dead / total_neurons if total_neurons > 0 else 0.0
        stats["dead_neurons/total_count"] = total_dead
        return stats
    
    def compute_stable_rank(self) -> Dict[str, float]:
        """
        Compute stable rank (srank) of weight matrices.
        srank = ||W||_F^2 / ||W||_2^2 = trace(W^T W) / max_eigenvalue(W^T W)
        """
        stats = {}
        sranks = []
        
        for name, param in self.network.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                W = param.view(param.size(0), -1)  # Flatten to 2D
                frobenius_sq = (W ** 2).sum().item()
                
                # Spectral norm (max singular value)
                try:
                    U, S, V = torch.linalg.svd(W, full_matrices=False)
                    spectral_sq = (S[0] ** 2).item()
                    srank = frobenius_sq / spectral_sq if spectral_sq > 1e-10 else 0.0
                    stats[f"srank/{name}"] = srank
                    sranks.append(srank)
                except:
                    pass
        
        if sranks:
            stats["srank/mean"] = np.mean(sranks)
            stats["srank/min"] = np.min(sranks)
        return stats
    
    def compute_effective_rank(self) -> Dict[str, float]:
        """
        Compute effective rank based on entropy of normalized singular values.
        eff_rank = exp(-sum(p * log(p))) where p = σ_i / sum(σ)
        """
        stats = {}
        ranks = []
        
        for name, param in self.network.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                W = param.view(param.size(0), -1)
                try:
                    U, S, V = torch.linalg.svd(W, full_matrices=False)
                    S = S + 1e-10  # Avoid log(0)
                    p = S / S.sum()
                    entropy = -(p * torch.log(p)).sum().item()
                    eff_rank = np.exp(entropy)
                    stats[f"effective_rank/{name}"] = eff_rank
                    ranks.append(eff_rank)
                except:
                    pass
        
        if ranks:
            stats["effective_rank/mean"] = np.mean(ranks)
        return stats
    
    def compute_weight_stats(self) -> Dict[str, float]:
        """Compute weight magnitude statistics."""
        stats = {}
        norms = []
        magnitudes = []
        
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                l2_norm = param.norm(2).item()
                mean_abs = param.abs().mean().item()
                stats[f"weight_norm/{name}"] = l2_norm
                stats[f"weight_mean_abs/{name}"] = mean_abs
                norms.append(l2_norm)
                magnitudes.append(mean_abs)
        
        if norms:
            stats["weight_norm/total"] = np.sqrt(sum(n**2 for n in norms))
            stats["weight_mean_abs/avg"] = np.mean(magnitudes)
        return stats
    
    def compute_all_metrics(self, include_svd: bool = True) -> Dict[str, float]:
        """Compute all plasticity metrics."""
        stats = {}
        
        # Dead neurons (requires activations from forward pass)
        if self.activation_buffers:
            stats.update(self.compute_dead_neurons())
        
        # Weight statistics (always available)
        stats.update(self.compute_weight_stats())
        
        # SVD-based metrics (expensive, optional)
        if include_svd:
            stats.update(self.compute_stable_rank())
            stats.update(self.compute_effective_rank())
        
        return stats


def compute_layer_activations_stats(agent: nn.Module, obs: torch.Tensor, 
                                    device: str = 'cuda') -> Dict[str, float]:
    """
    Quick function to compute activation statistics for PPO agent.
    Call this during training with a batch of observations.
    """
    stats = {}
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in agent.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Forward pass to collect activations
    with torch.no_grad():
        _ = agent.get_value(obs)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Compute dead neuron ratios
    threshold = 0.01
    total_dead = 0
    total_neurons = 0
    
    for name, act in activations.items():
        mean_act = act.abs().mean(dim=0)
        n_neurons = mean_act.numel()
        n_dead = (mean_act < threshold).sum().item()
        stats[f"dead/{name}"] = n_dead / n_neurons if n_neurons > 0 else 0.0
        total_dead += n_dead
        total_neurons += n_neurons
    
    stats["dead/total_ratio"] = total_dead / total_neurons if total_neurons > 0 else 0.0
    
    return stats
