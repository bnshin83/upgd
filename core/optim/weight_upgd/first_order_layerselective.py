import torch

class FirstOrderGlobalUPGDLayerSelective(torch.optim.Optimizer):
    """
    First-order global UPGD with selective layer gating.

    Allows applying utility gating only to specific layers while using
    fixed scaling (like SGD) for others.

    Args:
        gating_mode: One of:
            - 'full': Apply gating to all layers (standard UPGD)
            - 'output_only': Apply gating only to output layer (linear_3)
            - 'hidden_only': Apply gating only to hidden layers (linear_1, linear_2)
            - 'hidden_and_output': Apply gating to hidden + output (linear_1, linear_2, linear_3)
    """

    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0, gating_mode='full'):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma,
                       names=names, gating_mode=gating_mode)
        super(FirstOrderGlobalUPGDLayerSelective, self).__init__(params, defaults)

    def _should_apply_gating(self, layer_name, gating_mode):
        """Determine if utility gating should be applied to this layer."""
        if gating_mode == 'full':
            return True
        elif gating_mode == 'output_only':
            return 'linear_3' in layer_name
        elif gating_mode == 'hidden_only':
            return 'linear_1' in layer_name or 'linear_2' in layer_name
        elif gating_mode == 'hidden_and_output':
            return 'linear_1' in layer_name or 'linear_2' in layer_name or 'linear_3' in layer_name
        else:
            raise ValueError(f"Unknown gating_mode: {gating_mode}")

    def step(self):
        # First pass: compute global max utility (needed for all modes)
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

        # Collect utilities per layer
        all_scaled_utilities = []
        all_gradients = []
        all_weights = []
        all_raw_utilities = []

        layer_utilities = {}
        layer_gradients = {}
        layer_weights = {}
        layer_gating_applied = {}  # Track which layers got gating

        # Second pass: update parameters
        for group in self.param_groups:
            gating_mode = group["gating_mode"]

            for name, p in zip(group["names"], group["params"]):
                if p.grad is None or 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                # Collect for global statistics
                all_scaled_utilities.append(scaled_utility.flatten())
                all_gradients.append(p.grad.flatten())
                all_weights.append(p.data.flatten())
                all_raw_utilities.append((state["avg_utility"] / bias_correction).flatten())

                # Collect for per-layer statistics
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in layer_utilities:
                    layer_utilities[layer_name] = []
                    layer_gradients[layer_name] = []
                    layer_weights[layer_name] = []
                    layer_gating_applied[layer_name] = self._should_apply_gating(layer_name, gating_mode)

                layer_utilities[layer_name].append(scaled_utility.flatten())
                layer_gradients[layer_name].append(p.grad.flatten())
                layer_weights[layer_name].append(p.data.flatten())

                # Apply gating selectively
                if self._should_apply_gating(layer_name, gating_mode):
                    # Use utility gating
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * (1 - scaled_utility),
                        alpha=-1.0*group["lr"],
                    )
                else:
                    # Use fixed scaling at 0.5 (like SGD with matched LR)
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                        (p.grad.data + noise) * 0.5,
                        alpha=-1.0*group["lr"],
                    )

        # Compute statistics (same as standard UPGD)
        if all_scaled_utilities:
            scaled_utility_tensor = torch.cat(all_scaled_utilities)
            gradient_tensor = torch.cat(all_gradients)
            weight_tensor = torch.cat(all_weights)
            raw_utility_tensor = torch.cat(all_raw_utilities)

            total_params = scaled_utility_tensor.numel()

            # Global utility statistics
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

            self.utility_hist_0_20_pct = (self.utility_hist_0_20 / total_params) * 100
            self.utility_hist_20_40_pct = (self.utility_hist_20_40 / total_params) * 100
            self.utility_hist_40_44_pct = (self.utility_hist_40_44 / total_params) * 100
            self.utility_hist_44_48_pct = (self.utility_hist_44_48 / total_params) * 100
            self.utility_hist_48_52_pct = (self.utility_hist_48_52 / total_params) * 100
            self.utility_hist_52_56_pct = (self.utility_hist_52_56 / total_params) * 100
            self.utility_hist_56_60_pct = (self.utility_hist_56_60 / total_params) * 100
            self.utility_hist_60_80_pct = (self.utility_hist_60_80 / total_params) * 100
            self.utility_hist_80_100_pct = (self.utility_hist_80_100 / total_params) * 100
            self.utility_total_params = total_params

            # Gradient and weight histograms
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

            # Raw utility histogram
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

            # Per-layer statistics
            self.layer_stats = {}
            for layer_name in layer_utilities:
                layer_util_tensor = torch.cat(layer_utilities[layer_name])
                layer_grad_tensor = torch.cat(layer_gradients[layer_name])
                layer_weight_tensor = torch.cat(layer_weights[layer_name])
                layer_total = layer_util_tensor.numel()

                # 9-bin histogram
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
                    'gating_applied': layer_gating_applied[layer_name],  # Track gating status
                    'hist_0_20': hist_0_20,
                    'hist_20_40': hist_20_40,
                    'hist_40_44': hist_40_44,
                    'hist_44_48': hist_44_48,
                    'hist_48_52': hist_48_52,
                    'hist_52_56': hist_52_56,
                    'hist_56_60': hist_56_60,
                    'hist_60_80': hist_60_80,
                    'hist_80_100': hist_80_100,
                    'hist_0_20_pct': (hist_0_20 / layer_total) * 100,
                    'hist_20_40_pct': (hist_20_40 / layer_total) * 100,
                    'hist_40_44_pct': (hist_40_44 / layer_total) * 100,
                    'hist_44_48_pct': (hist_44_48 / layer_total) * 100,
                    'hist_48_52_pct': (hist_48_52 / layer_total) * 100,
                    'hist_52_56_pct': (hist_52_56 / layer_total) * 100,
                    'hist_56_60_pct': (hist_56_60 / layer_total) * 100,
                    'hist_60_80_pct': (hist_60_80 / layer_total) * 100,
                    'hist_80_100_pct': (hist_80_100 / layer_total) * 100,
                    'grad_mean': torch.abs(layer_grad_tensor).mean().item(),
                    'grad_std': torch.abs(layer_grad_tensor).std().item(),
                    'weight_mean': torch.abs(layer_weight_tensor).mean().item(),
                    'weight_std': torch.abs(layer_weight_tensor).std().item(),
                }
        else:
            # Initialize defaults
            self.layer_stats = {}

        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util

    def get_utility_stats(self):
        """Return utility statistics including per-layer gating status."""
        stats = {'utility/global_max': self.global_max_util}

        if hasattr(self, 'utility_L1'):
            stats.update({
                'utility/L1_norm': self.utility_L1,
                'utility/L2_norm': self.utility_L2,
                'utility/L4_norm': self.utility_L4,
                'utility/L5_norm': self.utility_L5,
                'utility/L10_norm': self.utility_L10,
            })

        # Add global utility histogram statistics (9 bins)
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

        # Add per-layer statistics including gating status
        if hasattr(self, 'layer_stats') and self.layer_stats:
            for layer_name, layer_stat in self.layer_stats.items():
                stats[f'layer/{layer_name}/utility_mean'] = layer_stat['mean']
                stats[f'layer/{layer_name}/utility_std'] = layer_stat['std']
                stats[f'layer/{layer_name}/gating_applied'] = 1.0 if layer_stat['gating_applied'] else 0.0

                # Add histogram stats
                for bin_name in ['0_20', '20_40', '40_44', '44_48', '48_52', '52_56', '56_60', '60_80', '80_100']:
                    stats[f'layer/{layer_name}/hist_{bin_name}'] = layer_stat[f'hist_{bin_name}']
                    stats[f'layer/{layer_name}/hist_{bin_name}_pct'] = layer_stat[f'hist_{bin_name}_pct']

        return stats

    def get_histogram_tensors(self):
        """Return tensor samples for histogram visualization."""
        return {}  # Can be implemented if needed
