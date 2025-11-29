import torch

class ShrinkandPerturb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.01, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, sigma=sigma, weight_decay=weight_decay, names=names)
        super(ShrinkandPerturb, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                # θ ← (1 - η·λ)θ - η·[∇L + σ·ε]
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad + torch.randn_like(p.grad) * group["sigma"], alpha=-group["lr"])
