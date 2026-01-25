import torch


class AdaptiveUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.001, beta_utility=0.999, sigma=0.001, beta1=0.9, beta2=0.999, eps=1e-5):
        # Convert params to list to allow multiple iterations
        param_list = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, beta1=beta1, beta2=beta2, eps=eps)
        super(AdaptiveUPGD, self).__init__(param_list, defaults)
        
        # For utility stats tracking
        self.global_max_util = 0.0
        self._all_scaled_utilities = []
        self._layer_utilities = {}
        self._param_to_layer = {}
        
        # Assign layer names to params (generic naming since we don't have named_parameters)
        layer_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                layer_name = f"layer_{layer_idx}"
                self._param_to_layer[id(p)] = layer_name
                layer_idx += 1

    def step(self):
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["sec_moment"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                first_moment, sec_moment = state["first_moment"], state["sec_moment"]
                first_moment.mul_(group["beta1"]).add_(p.grad.data, alpha=1 - group["beta1"])
                sec_moment.mul_(group["beta2"]).add_(p.grad.data ** 2, alpha=1 - group["beta2"])
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        
        # Store global max for logging
        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util
        
        # Clear previous utility tracking
        self._all_scaled_utilities = []
        self._layer_utilities = {}
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]
                bias_correction_beta1 = 1 - group["beta1"] ** state["step"]
                bias_correction_beta2 = 1 - group["beta2"] ** state["step"]
                exp_avg = state["first_moment"] / bias_correction_beta1
                exp_avg_sq = state["sec_moment"] / bias_correction_beta2
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction_utility) / global_max_util)
                
                # Track utilities for logging
                self._all_scaled_utilities.append(scaled_utility.detach().flatten())
                layer_name = self._param_to_layer.get(id(p), "unknown")
                if layer_name not in self._layer_utilities:
                    self._layer_utilities[layer_name] = []
                self._layer_utilities[layer_name].append(scaled_utility.detach().flatten())
                
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (exp_avg * (1 - scaled_utility)) / (exp_avg_sq.sqrt() + group["eps"]) + noise * (1-scaled_utility),
                    alpha=-2.0*group["lr"],
                )

    def get_utility_stats(self):
        """Return comprehensive utility statistics for logging."""
        stats = {'utility/global_max': self.global_max_util}
        
        # Count total params (all are gated in full UPGD)
        total_params = sum(p.numel() for group in self.param_groups for p in group["params"])
        stats['utility/gated_params'] = total_params
        stats['utility/non_gated_params'] = 0
        
        # Global histogram (9 bins)
        if self._all_scaled_utilities:
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
        for layer_name, util_list in self._layer_utilities.items():
            layer_utils = torch.cat(util_list)
            layer_total = layer_utils.numel()
            
            prefix = f"layer/{layer_name}"
            stats[f"{prefix}/gating_applied"] = 1.0  # All layers gated in full UPGD
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