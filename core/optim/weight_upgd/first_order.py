import torch
from torch.nn import functional as F

# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 (utility doesn't control gradient)
class FirstOrderNonprotectingGlobalUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderNonprotectingGlobalUPGD, self).__init__(params, defaults)

    def step(self):
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

        # Add epsilon to prevent division by zero
        global_max_util = torch.max(global_max_util, torch.tensor(1e-8))

        # Collect scaled utilities for norm computation
        all_scaled_utilities = []

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if p.grad is None or 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                # Collect scaled utilities for norm computation
                all_scaled_utilities.append(scaled_utility.flatten())

                # θ ← (1 - η·λ)θ - η·[∇L + σ·ε·(1 - u)]
                # where u = σ(Ū/max(Ū)) is scaled utility
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    p.grad.data
                    + noise * (1 - scaled_utility),
                    alpha=-1.0*group["lr"],
                )

        # Compute norms and histograms on scaled utilities
        if all_scaled_utilities:
            scaled_utility_tensor = torch.cat(all_scaled_utilities)
            self.utility_L1 = torch.norm(scaled_utility_tensor, p=1).item()
            self.utility_L2 = torch.norm(scaled_utility_tensor, p=2).item()
            self.utility_L4 = torch.norm(scaled_utility_tensor, p=4).item()
            self.utility_L5 = torch.norm(scaled_utility_tensor, p=5).item()
            self.utility_L10 = torch.norm(scaled_utility_tensor, p=10).item()

            # Compute histogram statistics (9 bins: [0, 0.2), [0.2, 0.4), [0.4, 0.44), [0.44, 0.48), [0.48, 0.52), [0.52, 0.56), [0.56, 0.6), [0.6, 0.8), [0.8, 1.0])
            total_params = scaled_utility_tensor.numel()
            self.utility_hist_0_20 = ((scaled_utility_tensor >= 0.0) & (scaled_utility_tensor < 0.2)).sum().item()
            self.utility_hist_20_40 = ((scaled_utility_tensor >= 0.2) & (scaled_utility_tensor < 0.4)).sum().item()
            self.utility_hist_40_44 = ((scaled_utility_tensor >= 0.4) & (scaled_utility_tensor < 0.44)).sum().item()
            self.utility_hist_44_48 = ((scaled_utility_tensor >= 0.44) & (scaled_utility_tensor < 0.48)).sum().item()
            self.utility_hist_48_52 = ((scaled_utility_tensor >= 0.48) & (scaled_utility_tensor < 0.52)).sum().item()
            self.utility_hist_52_56 = ((scaled_utility_tensor >= 0.52) & (scaled_utility_tensor < 0.56)).sum().item()
            self.utility_hist_56_60 = ((scaled_utility_tensor >= 0.56) & (scaled_utility_tensor < 0.6)).sum().item()
            self.utility_hist_60_80 = ((scaled_utility_tensor >= 0.6) & (scaled_utility_tensor < 0.8)).sum().item()
            self.utility_hist_80_100 = ((scaled_utility_tensor >= 0.8) & (scaled_utility_tensor <= 1.0)).sum().item()

            # Store percentages as well
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
        else:
            self.utility_L1 = 0.0
            self.utility_L2 = 0.0
            self.utility_L4 = 0.0
            self.utility_L5 = 0.0
            self.utility_L10 = 0.0
            self.utility_hist_0_20 = 0
            self.utility_hist_20_40 = 0
            self.utility_hist_40_44 = 0
            self.utility_hist_44_48 = 0
            self.utility_hist_48_52 = 0
            self.utility_hist_52_56 = 0
            self.utility_hist_56_60 = 0
            self.utility_hist_60_80 = 0
            self.utility_hist_80_100 = 0
            self.utility_hist_0_20_pct = 0.0
            self.utility_hist_20_40_pct = 0.0
            self.utility_hist_40_44_pct = 0.0
            self.utility_hist_44_48_pct = 0.0
            self.utility_hist_48_52_pct = 0.0
            self.utility_hist_52_56_pct = 0.0
            self.utility_hist_56_60_pct = 0.0
            self.utility_hist_60_80_pct = 0.0
            self.utility_hist_80_100_pct = 0.0
            self.utility_total_params = 0

        # Store global max utility for logging
        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util

    def get_utility_stats(self):
        """Return utility statistics for logging."""
        if not hasattr(self, 'global_max_util'):
            return {}

        stats = {
            'utility/global_max': self.global_max_util,
        }

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
            stats['utility/hist_40_44'] = self.utility_hist_40_44
            stats['utility/hist_44_48'] = self.utility_hist_44_48
            stats['utility/hist_48_52'] = self.utility_hist_48_52
            stats['utility/hist_52_56'] = self.utility_hist_52_56
            stats['utility/hist_56_60'] = self.utility_hist_56_60
            stats['utility/hist_60_80'] = self.utility_hist_60_80
            stats['utility/hist_80_100'] = self.utility_hist_80_100
            stats['utility/hist_0_20_pct'] = self.utility_hist_0_20_pct
            stats['utility/hist_20_40_pct'] = self.utility_hist_20_40_pct
            stats['utility/hist_40_44_pct'] = self.utility_hist_40_44_pct
            stats['utility/hist_44_48_pct'] = self.utility_hist_44_48_pct
            stats['utility/hist_48_52_pct'] = self.utility_hist_48_52_pct
            stats['utility/hist_52_56_pct'] = self.utility_hist_52_56_pct
            stats['utility/hist_56_60_pct'] = self.utility_hist_56_60_pct
            stats['utility/hist_60_80_pct'] = self.utility_hist_60_80_pct
            stats['utility/hist_80_100_pct'] = self.utility_hist_80_100_pct
            stats['utility/total_params'] = self.utility_total_params

        return stats

class FirstOrderNonprotectingLocalUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderNonprotectingLocalUPGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if p.grad is None or 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.sigmoid_(
                    F.normalize((avg_utility / bias_correction), dim=-1)
                )
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    p.grad.data + noise * (1 - scaled_utility), alpha=-1.0*group["lr"]
                )

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 (utility controls gradient)
class FirstOrderGlobalUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderGlobalUPGD, self).__init__(params, defaults)

    def step(self):
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

        # Add epsilon to prevent division by zero
        global_max_util = torch.max(global_max_util, torch.tensor(1e-8))

        # Collect scaled utilities, gradients, weights, and raw utilities for statistics
        all_scaled_utilities = []
        all_gradients = []
        all_weights = []
        all_raw_utilities = []

        # Per-layer tracking
        layer_utilities = {}  # {layer_name: [utilities]}
        layer_gradients = {}
        layer_weights = {}

        for group in self.param_groups:
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
                # Extract layer name from parameter name (e.g., "linear_1.weight" -> "linear_1")
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in layer_utilities:
                    layer_utilities[layer_name] = []
                    layer_gradients[layer_name] = []
                    layer_weights[layer_name] = []
                layer_utilities[layer_name].append(scaled_utility.flatten())
                layer_gradients[layer_name].append(p.grad.flatten())
                layer_weights[layer_name].append(p.data.flatten())

                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise)
                    * (1 - scaled_utility),
                    alpha=-1.0*group["lr"],
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

            # Scaled utility histogram (9 bins: [0, 0.2), [0.2, 0.4), [0.4, 0.44), [0.44, 0.48), [0.48, 0.52), [0.52, 0.56), [0.56, 0.6), [0.6, 0.8), [0.8, 1.0])
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
            
            # Store tensor samples for histogram visualization (sample up to 100k values to avoid memory issues)
            max_samples = 100000
            if total_params > max_samples:
                # Randomly sample
                indices = torch.randperm(total_params, device=scaled_utility_tensor.device)[:max_samples]
                self._hist_scaled_utility_sample = scaled_utility_tensor[indices].detach().cpu()
                self._hist_gradient_sample = gradient_tensor[indices].detach().cpu()
                self._hist_weight_sample = weight_tensor[indices].detach().cpu()
                self._hist_raw_utility_sample = raw_utility_tensor[indices].detach().cpu()
            else:
                # Store all values
                self._hist_scaled_utility_sample = scaled_utility_tensor.detach().cpu()
                self._hist_gradient_sample = gradient_tensor.detach().cpu()
                self._hist_weight_sample = weight_tensor.detach().cpu()
                self._hist_raw_utility_sample = raw_utility_tensor.detach().cpu()

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
                    'hist_0_20_pct': (hist_0_20 / layer_total) * 100,
                    'hist_20_40_pct': (hist_20_40 / layer_total) * 100,
                    'hist_40_44_pct': (hist_40_44 / layer_total) * 100,
                    'hist_44_48_pct': (hist_44_48 / layer_total) * 100,
                    'hist_48_52_pct': (hist_48_52 / layer_total) * 100,
                    'hist_52_56_pct': (hist_52_56 / layer_total) * 100,
                    'hist_56_60_pct': (hist_56_60 / layer_total) * 100,
                    'hist_60_80_pct': (hist_60_80 / layer_total) * 100,
                    'hist_80_100_pct': (hist_80_100 / layer_total) * 100,
                    # Gradient stats
                    'grad_mean': torch.abs(layer_grad_tensor).mean().item(),
                    'grad_std': torch.abs(layer_grad_tensor).std().item(),
                    # Weight stats
                    'weight_mean': torch.abs(layer_weight_tensor).mean().item(),
                    'weight_std': torch.abs(layer_weight_tensor).std().item(),
                }
        else:
            self.utility_L1 = 0.0
            self.utility_L2 = 0.0
            self.utility_L4 = 0.0
            self.utility_L5 = 0.0
            self.utility_L10 = 0.0
            self.utility_hist_0_20 = 0
            self.utility_hist_20_40 = 0
            self.utility_hist_40_44 = 0
            self.utility_hist_44_48 = 0
            self.utility_hist_48_52 = 0
            self.utility_hist_52_56 = 0
            self.utility_hist_56_60 = 0
            self.utility_hist_60_80 = 0
            self.utility_hist_80_100 = 0
            self.utility_hist_0_20_pct = 0.0
            self.utility_hist_20_40_pct = 0.0
            self.utility_hist_40_44_pct = 0.0
            self.utility_hist_44_48_pct = 0.0
            self.utility_hist_48_52_pct = 0.0
            self.utility_hist_52_56_pct = 0.0
            self.utility_hist_56_60_pct = 0.0
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
            # Clear histogram tensor samples
            if hasattr(self, '_hist_scaled_utility_sample'):
                delattr(self, '_hist_scaled_utility_sample')
                delattr(self, '_hist_gradient_sample')
                delattr(self, '_hist_weight_sample')
                delattr(self, '_hist_raw_utility_sample')
            # Clear layer stats
            self.layer_stats = {}

        # Store global max utility for logging
        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util

    def get_utility_stats(self):
        """Return utility statistics for logging."""
        if not hasattr(self, 'global_max_util'):
            return {}

        stats = {
            'utility/global_max': self.global_max_util,
        }

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
            stats['utility/hist_40_44'] = self.utility_hist_40_44
            stats['utility/hist_44_48'] = self.utility_hist_44_48
            stats['utility/hist_48_52'] = self.utility_hist_48_52
            stats['utility/hist_52_56'] = self.utility_hist_52_56
            stats['utility/hist_56_60'] = self.utility_hist_56_60
            stats['utility/hist_60_80'] = self.utility_hist_60_80
            stats['utility/hist_80_100'] = self.utility_hist_80_100
            stats['utility/hist_0_20_pct'] = self.utility_hist_0_20_pct
            stats['utility/hist_20_40_pct'] = self.utility_hist_20_40_pct
            stats['utility/hist_40_44_pct'] = self.utility_hist_40_44_pct
            stats['utility/hist_44_48_pct'] = self.utility_hist_44_48_pct
            stats['utility/hist_48_52_pct'] = self.utility_hist_48_52_pct
            stats['utility/hist_52_56_pct'] = self.utility_hist_52_56_pct
            stats['utility/hist_56_60_pct'] = self.utility_hist_56_60_pct
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
        """Return tensor samples for histogram visualization in WandB."""
        if not hasattr(self, '_hist_scaled_utility_sample'):
            return {}
        
        return {
            'histograms/scaled_utility': self._hist_scaled_utility_sample,
            'histograms/gradient': self._hist_gradient_sample,
            'histograms/weight': self._hist_weight_sample,
            'histograms/raw_utility': self._hist_raw_utility_sample,
        }


class FirstOrderLocalUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderLocalUPGD, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if p.grad is None or 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                # Retrieve the exponential moving average of utility from the optimizer state
                # This tracks the historical importance of each parameter for the learning task
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.sigmoid_(
                    F.normalize((avg_utility / bias_correction), dim=-1)
                )
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1 - scaled_utility), alpha=-1.0*group["lr"]
                )

