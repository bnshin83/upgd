import torch

class FirstOrderGlobalUPGDClampedSymmetric(torch.optim.Optimizer):
    """
    First-order global UPGD with utilities clamped to symmetric range [min_clamp, max_clamp].

    Tests how much of the utility range is needed for UPGD's performance.

    Args:
        min_clamp: Minimum utility value (default: 0.48)
        max_clamp: Maximum utility value (default: 0.52)

    Common configurations:
        - [0.48, 0.52]: Very narrow (±0.02 from 0.5)
        - [0.44, 0.56]: Narrow (±0.06 from 0.5)
        - [0.40, 0.60]: Moderate (±0.10 from 0.5)
    """

    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0,
                 min_clamp=0.48, max_clamp=0.52):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma,
                       names=names, min_clamp=min_clamp, max_clamp=max_clamp)
        super(FirstOrderGlobalUPGDClampedSymmetric, self).__init__(params, defaults)

    def step(self):
        # First pass: compute global max utility
        global_max_util = torch.tensor(-torch.inf)

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if p.grad is None or 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max

        global_max_util = torch.max(global_max_util, torch.tensor(1e-8))

        # Collect utilities
        all_scaled_utilities = []
        all_scaled_utilities_unclamped = []
        all_gradients = []
        all_weights = []
        all_raw_utilities = []
        num_clamped_params = 0
        total_params = 0

        # Per-layer tracking
        layer_utilities = {}
        layer_gradients = {}
        layer_weights = {}

        # Second pass: update parameters with SYMMETRIC CLAMPING
        for group in self.param_groups:
            min_clamp = group["min_clamp"]
            max_clamp = group["max_clamp"]

            for name, p in zip(group["names"], group["params"]):
                if p.grad is None or 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]

                # Compute unclamped scaled utility
                scaled_utility_unclamped = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                # CLAMP to symmetric range [min_clamp, max_clamp]
                scaled_utility = torch.clamp(scaled_utility_unclamped, min=min_clamp, max=max_clamp)

                # Count how many parameters were clamped
                num_clamped_params += ((scaled_utility_unclamped < min_clamp) | (scaled_utility_unclamped > max_clamp)).sum().item()
                total_params += scaled_utility_unclamped.numel()

                # Collect for statistics
                all_scaled_utilities.append(scaled_utility.flatten())
                all_scaled_utilities_unclamped.append(scaled_utility_unclamped.flatten())
                all_gradients.append(p.grad.flatten())
                all_weights.append(p.data.flatten())
                all_raw_utilities.append((state["avg_utility"] / bias_correction).flatten())

                # Collect for per-layer statistics
                # Extract layer name from parameter name (e.g., "linear_1.weight" -> "linear_1")
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in layer_utilities:
                    layer_utilities[layer_name] = []
                    layer_gradients[layer_name] = []
                    layer_weights[layer_name] = []
                layer_utilities[layer_name].append(scaled_utility.flatten())
                layer_gradients[layer_name].append(p.grad.flatten())
                layer_weights[layer_name].append(p.data.flatten())

                # UPGD update with CLAMPED utility
                noise = torch.randn_like(p.grad) * group["sigma"]
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1 - scaled_utility),
                    alpha=-1.0*group["lr"],
                )

        # Compute statistics
        if all_scaled_utilities:
            scaled_utility_tensor = torch.cat(all_scaled_utilities)
            scaled_utility_unclamped_tensor = torch.cat(all_scaled_utilities_unclamped)
            gradient_tensor = torch.cat(all_gradients)
            weight_tensor = torch.cat(all_weights)
            raw_utility_tensor = torch.cat(all_raw_utilities)

            total_params_count = scaled_utility_tensor.numel()

            # Store clamping statistics
            self.num_clamped_params = num_clamped_params
            self.total_params = total_params
            self.clamped_percentage = (num_clamped_params / total_params_count) * 100 if total_params_count > 0 else 0.0
            self.mean_utility_clamped = scaled_utility_tensor.mean().item()
            self.mean_utility_unclamped = scaled_utility_unclamped_tensor.mean().item()

            # Store clamping range
            self.min_clamp_value = min_clamp
            self.max_clamp_value = max_clamp
            self.clamp_range = max_clamp - min_clamp

            # Count clamped on each side
            self.num_clamped_low = (scaled_utility_unclamped_tensor < min_clamp).sum().item()
            self.num_clamped_high = (scaled_utility_unclamped_tensor > max_clamp).sum().item()
            self.pct_clamped_low = (self.num_clamped_low / total_params_count) * 100
            self.pct_clamped_high = (self.num_clamped_high / total_params_count) * 100

            # Standard utility statistics
            self.utility_L1 = torch.norm(scaled_utility_tensor, p=1).item()
            self.utility_L2 = torch.norm(scaled_utility_tensor, p=2).item()
            self.utility_L4 = torch.norm(scaled_utility_tensor, p=4).item()
            self.utility_L5 = torch.norm(scaled_utility_tensor, p=5).item()
            self.utility_L10 = torch.norm(scaled_utility_tensor, p=10).item()

            # 9-bin histogram
            self.utility_hist_0_20 = ((scaled_utility_tensor >= 0.0) & (scaled_utility_tensor < 0.2)).sum().item()
            self.utility_hist_20_40 = ((scaled_utility_tensor >= 0.2) & (scaled_utility_tensor < 0.4)).sum().item()
            self.utility_hist_40_44 = ((scaled_utility_tensor >= 0.4) & (scaled_utility_tensor < 0.44)).sum().item()
            self.utility_hist_44_48 = ((scaled_utility_tensor >= 0.44) & (scaled_utility_tensor < 0.48)).sum().item()
            self.utility_hist_48_52 = ((scaled_utility_tensor >= 0.48) & (scaled_utility_tensor < 0.52)).sum().item()
            self.utility_hist_52_56 = ((scaled_utility_tensor >= 0.52) & (scaled_utility_tensor < 0.56)).sum().item()
            self.utility_hist_56_60 = ((scaled_utility_tensor >= 0.56) & (scaled_utility_tensor < 0.6)).sum().item()
            self.utility_hist_60_80 = ((scaled_utility_tensor >= 0.6) & (scaled_utility_tensor < 0.8)).sum().item()
            self.utility_hist_80_100 = ((scaled_utility_tensor >= 0.8) & (scaled_utility_tensor <= 1.0)).sum().item()

            self.utility_hist_0_20_pct = (self.utility_hist_0_20 / total_params_count) * 100
            self.utility_hist_20_40_pct = (self.utility_hist_20_40 / total_params_count) * 100
            self.utility_hist_40_44_pct = (self.utility_hist_40_44 / total_params_count) * 100
            self.utility_hist_44_48_pct = (self.utility_hist_44_48 / total_params_count) * 100
            self.utility_hist_48_52_pct = (self.utility_hist_48_52 / total_params_count) * 100
            self.utility_hist_52_56_pct = (self.utility_hist_52_56 / total_params_count) * 100
            self.utility_hist_56_60_pct = (self.utility_hist_56_60 / total_params_count) * 100
            self.utility_hist_60_80_pct = (self.utility_hist_60_80 / total_params_count) * 100
            self.utility_hist_80_100_pct = (self.utility_hist_80_100 / total_params_count) * 100
            self.utility_total_params = total_params_count

            # Compute per-layer statistics
            self.layer_stats = {}
            for layer_name in layer_utilities:
                layer_util_tensor = torch.cat(layer_utilities[layer_name])
                layer_grad_tensor = torch.cat(layer_gradients[layer_name])
                layer_weight_tensor = torch.cat(layer_weights[layer_name])

                layer_total = layer_util_tensor.numel()

                # Compute 9-bin histogram (same as global)
                hist_0_20 = ((layer_util_tensor >= 0.0) & (layer_util_tensor < 0.2)).sum().item()
                hist_20_40 = ((layer_util_tensor >= 0.2) & (layer_util_tensor < 0.4)).sum().item()
                hist_40_44 = ((layer_util_tensor >= 0.4) & (layer_util_tensor < 0.44)).sum().item()
                hist_44_48 = ((layer_util_tensor >= 0.44) & (layer_util_tensor < 0.48)).sum().item()
                hist_48_52 = ((layer_util_tensor >= 0.48) & (layer_util_tensor < 0.52)).sum().item()
                hist_52_56 = ((layer_util_tensor >= 0.52) & (layer_util_tensor < 0.56)).sum().item()
                hist_56_60 = ((layer_util_tensor >= 0.56) & (layer_util_tensor < 0.6)).sum().item()
                hist_60_80 = ((layer_util_tensor >= 0.6) & (layer_util_tensor < 0.8)).sum().item()
                hist_80_100 = ((layer_util_tensor >= 0.8) & (layer_util_tensor <= 1.0)).sum().item()

                self.layer_stats[layer_name] = {
                    'mean': layer_util_tensor.mean().item(),
                    'std': layer_util_tensor.std().item(),
                    'min': layer_util_tensor.min().item(),
                    'max': layer_util_tensor.max().item(),
                    'count': layer_total,
                    # 9-bin histogram (counts)
                    'hist_0_20': hist_0_20,
                    'hist_20_40': hist_20_40,
                    'hist_40_44': hist_40_44,
                    'hist_44_48': hist_44_48,
                    'hist_48_52': hist_48_52,
                    'hist_52_56': hist_52_56,
                    'hist_56_60': hist_56_60,
                    'hist_60_80': hist_60_80,
                    'hist_80_100': hist_80_100,
                    # 9-bin histogram (percentages)
                    'hist_0_20_pct': (hist_0_20 / layer_total) * 100 if layer_total > 0 else 0,
                    'hist_20_40_pct': (hist_20_40 / layer_total) * 100 if layer_total > 0 else 0,
                    'hist_40_44_pct': (hist_40_44 / layer_total) * 100 if layer_total > 0 else 0,
                    'hist_44_48_pct': (hist_44_48 / layer_total) * 100 if layer_total > 0 else 0,
                    'hist_48_52_pct': (hist_48_52 / layer_total) * 100 if layer_total > 0 else 0,
                    'hist_52_56_pct': (hist_52_56 / layer_total) * 100 if layer_total > 0 else 0,
                    'hist_56_60_pct': (hist_56_60 / layer_total) * 100 if layer_total > 0 else 0,
                    'hist_60_80_pct': (hist_60_80 / layer_total) * 100 if layer_total > 0 else 0,
                    'hist_80_100_pct': (hist_80_100 / layer_total) * 100 if layer_total > 0 else 0,
                    # Gradient stats
                    'grad_mean': torch.abs(layer_grad_tensor).mean().item(),
                    'grad_std': torch.abs(layer_grad_tensor).std().item(),
                    # Weight stats
                    'weight_mean': torch.abs(layer_weight_tensor).mean().item(),
                    'weight_std': torch.abs(layer_weight_tensor).std().item(),
                }
        else:
            # Defaults
            self.num_clamped_params = 0
            self.total_params = 0
            self.clamped_percentage = 0.0
            self.mean_utility_clamped = 0.0
            self.mean_utility_unclamped = 0.0
            self.layer_stats = {}

        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util

    def get_utility_stats(self):
        """Return utility statistics including symmetric clamping info."""
        if not hasattr(self, 'global_max_util'):
            return {}

        stats = {
            'utility/global_max': self.global_max_util,
            # Clamping-specific stats
            'clamping/min_clamp': self.min_clamp_value,
            'clamping/max_clamp': self.max_clamp_value,
            'clamping/range': self.clamp_range,
            'clamping/num_clamped': self.num_clamped_params,
            'clamping/percentage': self.clamped_percentage,
            'clamping/num_clamped_low': self.num_clamped_low,
            'clamping/num_clamped_high': self.num_clamped_high,
            'clamping/pct_clamped_low': self.pct_clamped_low,
            'clamping/pct_clamped_high': self.pct_clamped_high,
            'clamping/mean_clamped': self.mean_utility_clamped,
            'clamping/mean_unclamped': self.mean_utility_unclamped,
        }

        # Add standard utility stats
        if hasattr(self, 'utility_L1'):
            stats.update({
                'utility/L1_norm': self.utility_L1,
                'utility/L2_norm': self.utility_L2,
                'utility/L4_norm': self.utility_L4,
                'utility/L5_norm': self.utility_L5,
                'utility/L10_norm': self.utility_L10,
            })

        # Add utility histogram statistics (9 bins)
        if hasattr(self, 'utility_hist_0_20'):
            stats.update({
                'utility/hist_0_20': self.utility_hist_0_20,
                'utility/hist_20_40': self.utility_hist_20_40,
                'utility/hist_40_44': self.utility_hist_40_44,
                'utility/hist_44_48': self.utility_hist_44_48,
                'utility/hist_48_52': self.utility_hist_48_52,
                'utility/hist_52_56': self.utility_hist_52_56,
                'utility/hist_56_60': self.utility_hist_56_60,
                'utility/hist_60_80': self.utility_hist_60_80,
                'utility/hist_80_100': self.utility_hist_80_100,
                'utility/hist_0_20_pct': self.utility_hist_0_20_pct,
                'utility/hist_20_40_pct': self.utility_hist_20_40_pct,
                'utility/hist_40_44_pct': self.utility_hist_40_44_pct,
                'utility/hist_44_48_pct': self.utility_hist_44_48_pct,
                'utility/hist_48_52_pct': self.utility_hist_48_52_pct,
                'utility/hist_52_56_pct': self.utility_hist_52_56_pct,
                'utility/hist_56_60_pct': self.utility_hist_56_60_pct,
                'utility/hist_60_80_pct': self.utility_hist_60_80_pct,
                'utility/hist_80_100_pct': self.utility_hist_80_100_pct,
                'utility/total_params': self.utility_total_params,
            })

        # Add per-layer statistics
        if hasattr(self, 'layer_stats') and self.layer_stats:
            for layer_name, layer_stat in self.layer_stats.items():
                # Basic utility statistics per layer
                stats[f'layer/{layer_name}/utility_mean'] = layer_stat['mean']
                stats[f'layer/{layer_name}/utility_std'] = layer_stat['std']
                stats[f'layer/{layer_name}/utility_min'] = layer_stat['min']
                stats[f'layer/{layer_name}/utility_max'] = layer_stat['max']
                stats[f'layer/{layer_name}/param_count'] = layer_stat['count']

                # 9-bin histogram (counts)
                stats[f'layer/{layer_name}/hist_0_20'] = layer_stat['hist_0_20']
                stats[f'layer/{layer_name}/hist_20_40'] = layer_stat['hist_20_40']
                stats[f'layer/{layer_name}/hist_40_44'] = layer_stat['hist_40_44']
                stats[f'layer/{layer_name}/hist_44_48'] = layer_stat['hist_44_48']
                stats[f'layer/{layer_name}/hist_48_52'] = layer_stat['hist_48_52']
                stats[f'layer/{layer_name}/hist_52_56'] = layer_stat['hist_52_56']
                stats[f'layer/{layer_name}/hist_56_60'] = layer_stat['hist_56_60']
                stats[f'layer/{layer_name}/hist_60_80'] = layer_stat['hist_60_80']
                stats[f'layer/{layer_name}/hist_80_100'] = layer_stat['hist_80_100']

                # 9-bin histogram (percentages)
                stats[f'layer/{layer_name}/hist_0_20_pct'] = layer_stat['hist_0_20_pct']
                stats[f'layer/{layer_name}/hist_20_40_pct'] = layer_stat['hist_20_40_pct']
                stats[f'layer/{layer_name}/hist_40_44_pct'] = layer_stat['hist_40_44_pct']
                stats[f'layer/{layer_name}/hist_44_48_pct'] = layer_stat['hist_44_48_pct']
                stats[f'layer/{layer_name}/hist_48_52_pct'] = layer_stat['hist_48_52_pct']
                stats[f'layer/{layer_name}/hist_52_56_pct'] = layer_stat['hist_52_56_pct']
                stats[f'layer/{layer_name}/hist_56_60_pct'] = layer_stat['hist_56_60_pct']
                stats[f'layer/{layer_name}/hist_60_80_pct'] = layer_stat['hist_60_80_pct']
                stats[f'layer/{layer_name}/hist_80_100_pct'] = layer_stat['hist_80_100_pct']

                # Gradient statistics per layer
                stats[f'layer/{layer_name}/grad_mean'] = layer_stat['grad_mean']
                stats[f'layer/{layer_name}/grad_std'] = layer_stat['grad_std']

                # Weight statistics per layer
                stats[f'layer/{layer_name}/weight_mean'] = layer_stat['weight_mean']
                stats[f'layer/{layer_name}/weight_std'] = layer_stat['weight_std']

        return stats

    def get_histogram_tensors(self):
        """Return tensor samples for histogram visualization."""
        return {}  # Can be implemented if needed
