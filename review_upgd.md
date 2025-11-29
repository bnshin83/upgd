

# UPGD Implementation Review

This document provides a comprehensive review of the local UPGD (Utility-based Perturbed Gradient Descent) implementation compared to the original repository design.

## Overview

The codebase contains multiple UPGD optimizer variants with both weight-based and feature-based implementations, including input-aware extensions that incorporate curvature-driven modulation. This review analyzes implementation differences, identifies bugs, and provides recommendations for improvement.

## What's Implemented Locally

### Weight-based UPGD Variants
First- and second-order implementations in global/local and "nonprotecting" variants:
```1:42:/scratch/gautschi/shin283/upgd/core/optim/weight_upgd/first_order.py
import torch
from torch.nn import functional as F

# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 (utility doesn't control gradient)
class FirstOrderNonprotectingGlobalUPGD(torch.optim.Optimizer):
    ...
```

```80:150:/scratch/gautschi/shin283/upgd/core/optim/weight_upgd/second_order.py
class SecondOrderLocalUPGD(torch.optim.Optimizer):
    ...
class SecondOrderGlobalUPGD(torch.optim.Optimizer):
    ...
```

### Feature-based UPGD Variants
UPGD computed on "gate" layers and applied to subsequent parameters:
```1:76:/scratch/gautschi/shin283/upgd/core/optim/feature_upgd/first_order.py
class FirstOrderNonprotectingLocalUPGD(torch.optim.Optimizer):
    ...
class FirstOrderLocalUPGD(torch.optim.Optimizer):
    ...
```

### Input-aware Extensions
Curvature-driven modulation and learner wrappers:
```111:236:/scratch/gautschi/shin283/upgd/core/optim/weight_upgd/input_aware.py
class InputAwareFirstOrderGlobalUPGD(torch.optim.Optimizer):
    ...
    def step(self):
        ...
        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
            (p.grad.data + regularization + noise) * gating_factor,
            alpha=-group["lr"]
        )
```

```11:49:/scratch/gautschi/shin283/upgd/core/learner/input_aware_upgd.py
class InputAwareFirstOrderGlobalUPGDLearner(Learner):
    ...
    def compute_input_curvature(self, model, input_batch, targets, criterion, return_per_sample=False):
        ...
```

## Comparison with Original Implementation

### High-Level Differences

**Original README Implementation:**
- Gates the entire (grad + noise) by utility
- Uses a factor of 2 in the step size
```python
# original readme
p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
    (p.grad.data + noise) * (1-scaled_utility),
    alpha=-2.0*group["lr"],
)
```

**Local Implementation (Variation 2):**
- Matches the gating-of-(grad+noise) approach
- Uses `alpha=-lr` instead of `-2*lr`
```100:111:/scratch/gautschi/shin283/upgd/core/optim/weight_upgd/first_order.py
bias_correction = 1 - group["beta_utility"] ** state["step"]
noise = torch.randn_like(p.grad) * group["sigma"]
scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
    (p.grad.data + noise)
    * (1 - scaled_utility),
    alpha=-group["lr"],
)
```

**"Nonprotecting" Variants:**
- Only gate the noise, not the gradient
- Deliberate deviation from README behavior

## Issues and Findings

### 🐛 Critical Bugs

#### 1. Feature-UPGD: Stale Bias-Correction Bug
In several feature-upgd classes, the second pass uses a `bias_correction` variable that is not recomputed in that pass; it relies on a stale value from the first pass, which can mismatch the current `state["step"]`.

**Affected Code:**
```151:156:/scratch/gautschi/shin283/upgd/core/optim/feature_upgd/first_order.py
for group in self.param_groups:
    for name, p in zip(reversed(group["names"]), reversed(group["params"])):
        state = self.state[p]
        if 'gate' in name:
            gate_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
            continue
```

Same pattern in `feature_upgd/second_order.py`:
```161:166:/scratch/gautschi/shin283/upgd/core/optim/feature_upgd/second_order.py
if 'gate' in name:
    gate_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
    continue
```

**Fix:** Recompute `bias_correction = 1 - beta_utility ** state["step"]` in the second pass (per param).

#### 2. Input-aware Run: Wrong Curvature Call Signature
The run calls `compute_input_curvature(loss, input)` but the learner expects `(model, input_batch, targets, criterion, ...)`.

**Affected Code:**
```88:101:/scratch/gautschi/shin283/upgd/core/run/input_aware_run.py
if self.is_input_aware and step % self.compute_curvature_every == 0:
    with torch.enable_grad():
        # Compute input curvature before backward pass
        curvature = self.learner.compute_input_curvature(loss, input)
        self.learner.update_optimizer_curvature(curvature)
        curvatures_per_step.append(curvature)
        ...
```

**Fix:** Correct usage should pass `model=self.learner.network`, `inputs=input`, `targets=target`, `criterion=criterion`.

### ⚠️ Potential Numeric Edge Cases

1. **Division by Zero Risk:** All UPGD variants divide by `global_max_util` with no epsilon; if it ever becomes 0, scaled utility can explode. Consider adding a small epsilon.

2. **Missing Gradient Checks:** No `if p.grad is None: continue` checks; safer to skip those params.

3. **Deprecated Pattern:** Use of `.data` works but `with torch.no_grad(): p.add_...` is the recommended pattern.

### 📋 Behavioral Differences vs Original README

1. **Step Size Factor:** README uses `alpha=-2*lr`, local uses `-lr`. If strict parity is desired, adjust.

2. **Gating Strategy:** Some first/second-order "nonprotecting" classes only gate noise, not gradient; that's intentional but differs from the README's single presented form.

3. **Gate Parameter Handling:** Weight-UPGD skips parameters with `'gate'` in the name, while Feature-UPGD consumes gate params to modulate following params; the original README does not discuss gating layers explicitly (your design extends it).

## Recommendations

### 🔧 Immediate Fixes Required
1. **Recompute `bias_correction`** per param in feature-upgd second passes
2. **Fix `compute_input_curvature` invocation** in `input_aware_run.py`

### 🛡️ Robustness Improvements
3. **Add epsilon** to the `global_max_util` divisor to prevent division by zero
4. **Add gradient checks:** `if p.grad is None: continue`
5. **Clamp gating factor** in input-aware variants to [0, 1] range to avoid negative values when `lambda_max` grows

### 📝 Documentation & Alignment
6. **Decide on step size factor:** Align to README's `-2*lr` or document the rationale for using `-lr`
7. **Update code style:** Replace `.data` usage with `with torch.no_grad(): p.add_...` pattern

## References

### External Links
- **Original Repository:** [mohmdelsayed/upgd](https://github.com/mohmdelsayed/upgd.git)

### Code References
Example local implementations referenced in this review:
```73:111:/scratch/gautschi/shin283/upgd/core/optim/weight_upgd/first_order.py
# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 (utility controls gradient)
class FirstOrderGlobalUPGD(torch.optim.Optimizer):
    ...
    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
        (p.grad.data + noise)
        * (1 - scaled_utility),
        alpha=-group["lr"],
    )
```

```239:359:/scratch/gautschi/shin283/upgd/core/optim/weight_upgd/input_aware.py
class InputAwareSecondOrderGlobalUPGD(torch.optim.Optimizer):
    ...
    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
        (p.grad.data + regularization + noise) * gating_factor,
        alpha=-group["lr"]
    )
```

---

**Note:** The two critical bugs (feature-upgd bias_correction and input_aware_run signature) require immediate attention for correct functionality.