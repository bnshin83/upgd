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

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if p.grad is None or 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                # Collect for statistics
                all_scaled_utilities.append(scaled_utility.flatten())
                all_gradients.append(p.grad.flatten())
                all_weights.append(p.data.flatten())
                all_raw_utilities.append((state["avg_utility"] / bias_correction).flatten())

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

        return stats


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
                if 'gate' in name:
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
                    
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise)
                    * (1 - scaled_utility),
                    alpha=-group["lr"],
                )


class FirstOrderLocalUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderLocalUPGD, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
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
                    (p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"]
                )
