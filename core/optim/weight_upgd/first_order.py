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
                    p.grad.data
                    + noise * (1 - scaled_utility),
                    alpha=-group["lr"],
                )

# Sangbin: Faster, vectorized version of FONPG_UPGD
class FastFirstOrderNonprotectingGlobalUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FastFirstOrderNonprotectingGlobalUPGD, self).__init__(params, defaults)

        # 🚀 Group parameters by type for vectorized operations
        self.gate_params = []
        self.non_gate_params = []

        # Separate parameters into gate and non-gate lists
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    self.gate_params.append(p)
                else:
                    self.non_gate_params.append(p)

    def step(self, closure=None):           
        group = self.param_groups[0] # Assuming one param group
        beta_utility = group['beta_utility']
        lr = group['lr']
        weight_decay = group['weight_decay']
        sigma = group['sigma']

        # === 1. Gather Tensors and Pre-compute Scalars ===
        grads = []
        avg_utilities = []
        all_utilities = []
        params_data = []
        bias_corrections = []

        for p in self.non_gate_params:
            if p.grad is None:
                continue
            
            params_data.append(p.data)
            grads.append(p.grad.data)
            
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["avg_utility"] = torch.zeros_like(p.data)   
            state['step'] += 1
            avg_utilities.append(state['avg_utility'])
            bias_corrections.append(1 - beta_utility ** state['step'])

        # === 2. Calculate Global Max Utility (Vectorized) === 🚀
        # Calculate current utility: u_t = -g_t * p_t
        current_utilities = torch._foreach_mul(grads, params_data)
        torch._foreach_mul_(current_utilities, -1.0)

        # Update EMA of utility: avg_u = beta * avg_u + (1-beta) * u_t
        torch._foreach_mul_(avg_utilities, beta_utility)
        torch._foreach_add_(avg_utilities, current_utilities, alpha=1 - beta_utility)

        # Find the maximum utility across all parameters
        # We find the max of each tensor, stack them, and find the global max.
        max_utils_tensor = torch.stack([u.max() for u in avg_utilities])
        global_max_util = max_utils_tensor.max()

        # === 3. Update Parameters in Bulk (Vectorized) === 🚀
        
        # Calculate scaled utility: s = sigmoid((avg_u / bias_correction) / global_max)
        scaled_utilities = [u.clone() for u in avg_utilities]
        torch._foreach_div_(scaled_utilities, bias_corrections)
        # Add a small epsilon for numerical stability
        torch._foreach_div_(scaled_utilities, global_max_util + 1e-8)
        torch._foreach_sigmoid_(scaled_utilities)
        
        # Generate noise for all parameters at once
        noise_list = [torch.randn_like(p) for p in params_data]
        torch._foreach_mul_(noise_list, sigma)
        
        # Calculate noise term: noise * (1 - scaled_utility)
        one_minus_scaled = [torch.sub(1.0, s) for s in scaled_utilities]
        torch._foreach_mul_(noise_list, one_minus_scaled)
        
        # Perform the final SGD update step with weight decay
        # p = p * (1 - lr * wd) - lr * (g + noise_term)
        torch._foreach_mul_(params_data, 1.0 - lr * weight_decay)
        torch._foreach_add_(params_data, grads, alpha=-lr)
        torch._foreach_add_(params_data, noise_list, alpha=-lr)


class FirstOrderNonprotectingLocalUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderNonprotectingLocalUPGD, self).__init__(params, defaults)

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
                    p.grad.data + noise * (1 - scaled_utility), alpha=-group["lr"]
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
