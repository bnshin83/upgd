import torch

class SGDCurvatureGating(torch.optim.Optimizer):
    """
    SGD with curvature-based gating instead of utility-based gating.

    Gating factor: g = sigmoid(-(κ - τ) / s)
    where:
        κ = input curvature
        τ = curvature threshold
        s = curvature scale

    High curvature → low gating factor → strong protection
    Low curvature → high gating factor → normal updates
    """
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.9999, sigma=0.0,
                 curvature_threshold=0.01, curvature_scale=0.01, beta_curvature=0.9,
                 disable_regularization=False, disable_gating=False, **kwargs):
        """
        Args:
            lr: learning rate
            weight_decay: L2 weight decay
            beta_utility: momentum for utility tracking (for logging only)
            sigma: noise standard deviation
            curvature_threshold: threshold τ for curvature gating
            curvature_scale: scale s for sigmoid mapping
            beta_curvature: momentum for tracking average curvature
            disable_regularization: ignored (for compatibility with runner)
            disable_gating: ignored (for compatibility with runner)
            **kwargs: additional arguments (ignored)
        """
        names, params = zip(*params)
        defaults = dict(
            lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma,
            curvature_threshold=curvature_threshold, curvature_scale=curvature_scale,
            beta_curvature=beta_curvature, names=names
        )
        super(SGDCurvatureGating, self).__init__(params, defaults)

        # Curvature tracking
        self.current_input_curvature = 0.0
        self.avg_input_curvature = 0.0
        self.curvature_step = 0

        # Gating factor histogram bins (counts over time steps)
        self.gating_hist_0_20 = 0    # [0, 0.2): Very strong protection
        self.gating_hist_20_40 = 0   # [0.2, 0.4): Strong protection
        self.gating_hist_40_60 = 0   # [0.4, 0.6): Medium protection
        self.gating_hist_60_80 = 0   # [0.6, 0.8): Light protection
        self.gating_hist_80_100 = 0  # [0.8, 1.0]: Nearly full update

    def set_input_curvature(self, curvature):
        """Set the current input curvature value (computed externally)."""
        self.current_input_curvature = curvature
        self.curvature_step += 1

        # Update running average
        beta = self.param_groups[0]['beta_curvature']
        self.avg_input_curvature = beta * self.avg_input_curvature + (1 - beta) * curvature

        # Compute gating factor and update histogram
        gating = self.compute_gating_factor()
        if gating < 0.2:
            self.gating_hist_0_20 += 1
        elif gating < 0.4:
            self.gating_hist_20_40 += 1
        elif gating < 0.6:
            self.gating_hist_40_60 += 1
        elif gating < 0.8:
            self.gating_hist_60_80 += 1
        else:
            self.gating_hist_80_100 += 1

    def compute_gating_factor(self):
        """
        Compute gating factor from input curvature using inverse sigmoid.

        g = sigmoid(-(κ - τ) / s)

        Returns scalar gating factor in [0, 1]:
            - High curvature (κ >> τ) → g ≈ 0 (strong protection)
            - Low curvature (κ << τ) → g ≈ 1 (normal update)
            - κ = τ → g = 0.5
        """
        if self.curvature_step == 0:
            return 1.0  # No protection until we have curvature data

        threshold = self.param_groups[0]['curvature_threshold']
        scale = self.param_groups[0]['curvature_scale']

        # Inverse sigmoid: high curvature → low gating
        normalized = (self.current_input_curvature - threshold) / scale
        gating = torch.sigmoid(-torch.tensor(normalized)).item()

        return gating

    def step(self):
        # First pass: compute utility for logging/monitoring
        global_max_util = torch.tensor(-torch.inf)

        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name or p.grad is None:
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

        # Compute curvature-based gating factor (scalar, same for all parameters)
        gating_factor = self.compute_gating_factor()

        # Collect scaled utilities and gating factors for logging
        all_scaled_utilities = []
        all_gating_factors = []

        # Second pass: update parameters with curvature-based gating
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name or p.grad is None:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)

                # Collect for logging
                all_scaled_utilities.append(scaled_utility.flatten())
                all_gating_factors.append(torch.full_like(scaled_utility, gating_factor).flatten())

                # SGD update with curvature-based gating
                noise = torch.randn_like(p.grad) * group["sigma"] if group["sigma"] > 0 else 0
                p.data.add_(
                    (p.grad + noise + group['weight_decay'] * p.data) * gating_factor,
                    alpha=-group["lr"]
                )

        # Compute norms and histograms on scaled utilities (for monitoring)
        if all_scaled_utilities:
            scaled_utility_tensor = torch.cat(all_scaled_utilities)
            self.utility_L1 = torch.norm(scaled_utility_tensor, p=1).item()
            self.utility_L2 = torch.norm(scaled_utility_tensor, p=2).item()
            self.utility_L4 = torch.norm(scaled_utility_tensor, p=4).item()
            self.utility_L5 = torch.norm(scaled_utility_tensor, p=5).item()
            self.utility_L10 = torch.norm(scaled_utility_tensor, p=10).item()

            # Utility histogram
            total_params = scaled_utility_tensor.numel()
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

            # Gating histogram (all same value since curvature-based gating is global)
            gating_tensor = torch.cat(all_gating_factors)
            self.gating_hist_0_20 = ((gating_tensor >= 0.0) & (gating_tensor < 0.2)).sum().item()
            self.gating_hist_20_40 = ((gating_tensor >= 0.2) & (gating_tensor < 0.4)).sum().item()
            self.gating_hist_40_60 = ((gating_tensor >= 0.4) & (gating_tensor < 0.6)).sum().item()
            self.gating_hist_60_80 = ((gating_tensor >= 0.6) & (gating_tensor < 0.8)).sum().item()
            self.gating_hist_80_100 = ((gating_tensor >= 0.8) & (gating_tensor <= 1.0)).sum().item()

            self.gating_hist_0_20_pct = (self.gating_hist_0_20 / total_params) * 100
            self.gating_hist_20_40_pct = (self.gating_hist_20_40 / total_params) * 100
            self.gating_hist_40_60_pct = (self.gating_hist_40_60 / total_params) * 100
            self.gating_hist_60_80_pct = (self.gating_hist_60_80 / total_params) * 100
            self.gating_hist_80_100_pct = (self.gating_hist_80_100 / total_params) * 100
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

            self.gating_hist_0_20 = 0
            self.gating_hist_20_40 = 0
            self.gating_hist_40_60 = 0
            self.gating_hist_60_80 = 0
            self.gating_hist_80_100 = 0
            self.gating_hist_0_20_pct = 0.0
            self.gating_hist_20_40_pct = 0.0
            self.gating_hist_40_60_pct = 0.0
            self.gating_hist_60_80_pct = 0.0
            self.gating_hist_80_100_pct = 0.0

        # Store statistics for logging
        self.global_max_util = global_max_util.item() if isinstance(global_max_util, torch.Tensor) else global_max_util
        self.current_gating_factor = gating_factor

    def get_utility_stats(self):
        """Return utility and gating statistics for logging."""
        if not hasattr(self, 'global_max_util'):
            return {}

        # Compute gating histogram percentages (over time steps)
        total_steps = self.curvature_step
        if total_steps > 0:
            self.gating_hist_0_20_pct = (self.gating_hist_0_20 / total_steps) * 100
            self.gating_hist_20_40_pct = (self.gating_hist_20_40 / total_steps) * 100
            self.gating_hist_40_60_pct = (self.gating_hist_40_60 / total_steps) * 100
            self.gating_hist_60_80_pct = (self.gating_hist_60_80 / total_steps) * 100
            self.gating_hist_80_100_pct = (self.gating_hist_80_100 / total_steps) * 100
        else:
            self.gating_hist_0_20_pct = 0.0
            self.gating_hist_20_40_pct = 0.0
            self.gating_hist_40_60_pct = 0.0
            self.gating_hist_60_80_pct = 0.0
            self.gating_hist_80_100_pct = 0.0

        stats = {
            'utility/global_max': self.global_max_util,
            'gating/curvature_based': self.current_gating_factor,
            'curvature/current': self.current_input_curvature,
            'curvature/avg': self.avg_input_curvature,
            'gating/total_steps': total_steps,
        }

        # Add utility norms
        if hasattr(self, 'utility_L1'):
            stats['utility/L1_norm'] = self.utility_L1
            stats['utility/L2_norm'] = self.utility_L2
            stats['utility/L4_norm'] = self.utility_L4
            stats['utility/L5_norm'] = self.utility_L5
            stats['utility/L10_norm'] = self.utility_L10

        # Add utility histogram
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

        # Add gating histogram
        if hasattr(self, 'gating_hist_0_20'):
            stats['gating/hist_0_20'] = self.gating_hist_0_20
            stats['gating/hist_20_40'] = self.gating_hist_20_40
            stats['gating/hist_40_60'] = self.gating_hist_40_60
            stats['gating/hist_60_80'] = self.gating_hist_60_80
            stats['gating/hist_80_100'] = self.gating_hist_80_100
            stats['gating/hist_0_20_pct'] = self.gating_hist_0_20_pct
            stats['gating/hist_20_40_pct'] = self.gating_hist_20_40_pct
            stats['gating/hist_40_60_pct'] = self.gating_hist_40_60_pct
            stats['gating/hist_60_80_pct'] = self.gating_hist_60_80_pct
            stats['gating/hist_80_100_pct'] = self.gating_hist_80_100_pct

        return stats
