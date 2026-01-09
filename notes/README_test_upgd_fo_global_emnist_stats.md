# UPGD FO Global EMNIST Stats

This shows how to run the baseline UPGD first-order global statistics job on EMNIST using the SLURM script `test_upgd_fo_global_emnist_stats.sh`, what artifacts are produced, and how to generate PDF plots (dead units, gradient norms, plasticity, etc.) from the logged JSON.

### What the script does
- Submits a single-GPU SLURM job that runs the standard statistics runner:
  - `core/run/run_stats.py --task label_permuted_emnist_stats --learner upgd_fo_global --lr 0.01 --sigma 0.001 --network fully_connected_relu_with_hooks --n_samples 1000000`
- Saves all per-task statistics (losses, accuracies, plasticity, dead units, gradient norms, weight norms, etc.) as a JSON file under the logs directory.

### Script location
- `test_upgd_fo_global_emnist_stats.sh`

### Prerequisites
- Working CUDA and Python modules on your cluster (the script loads them).
- Virtual environment at `/scratch/gautschi/shin283/upgd/.upgd` with project requirements installed.
- `PYTHONPATH` includes the project root.

### How to run
```bash
# From the repository root
sbatch test_upgd_fo_global_emnist_stats.sh

# Monitor job
tail -f /scratch/gautschi/shin283/upgd/logs/<JOB_ID>_test_upgd_fo_global_emnist_stats.out
 tail -f /scratch/gautschi/shin283/upgd/logs/<JOB_ID>_test_upgd_fo_global_emnist_stats.err
```

### Command executed by the job
```bash
python3 core/run/run_stats.py \
  --task label_permuted_emnist_stats \
  --learner upgd_fo_global \
  --seed 0 \
  --lr 0.01 \
  --sigma 0.001 \
  --network fully_connected_relu_with_hooks \
  --n_samples 1000000
```

### Output artifacts
- JSON logs are written by `core/logger.py` with the following directory schema:
  - `logs/{task}/{learner}/{network}/{optimizer_hps_key_val_pairs}/{seed}.json`
- Concretely, for the above command (values may differ depending on `network.name` and hps):
  - `/scratch/gautschi/shin283/upgd/logs/label_permuted_emnist_stats/upgd_fo_global/{network}/lr_0.01_sigma_0.001/0.json`

Notes:
- `{network}` is taken from the learner's `network.name`. If you pass `fully_connected_relu_with_hooks`, the saved folder name may be the underlying network's name (e.g., `fully_connected_relu`).
- The JSON includes arrays for:
  - `losses`, `accuracies` (classification tasks),
  - `plasticity_per_task`,
  - `n_dead_units_per_task`, `weight_l2_per_task`, `weight_l1_per_task`,
  - `grad_l2_per_task`, `grad_l1_per_task`, `grad_l0_per_task`, and more.

### Generate PDF plots (dead units, gradient norms, plasticity, …)
Use the provided plotting utility to convert a single JSON file into a set of PDFs saved alongside the JSON (or to a custom output directory).

```bash
python3 /scratch/gautschi/shin283/upgd/plot_experiment_stats.py \
  --json-file \
  "/scratch/gautschi/shin283/upgd/logs/label_permuted_emnist_stats/upgd_fo_global/{network}/lr_0.01_sigma_0.001/0.json"
```

Generated PDFs:
- `learning_curves.pdf`
- `plasticity.pdf`
- `dead_units.pdf`
- `weight_norms.pdf`
- `gradient_norms.pdf`
- `gradient_sparsity.pdf`
- `summary.pdf`

Optional: specify a different output directory
```bash
python3 /scratch/gautschi/shin283/upgd/plot_experiment_stats.py \
  --json-file "/path/to/0.json" \
  --output-dir "/path/to/output/plots"
```

### Troubleshooting
- No JSON output:
  - Check SLURM `.out/.err` files for Python errors.
  - Ensure `PYTHONPATH` is set and the venv is activated inside the script.
- Missing PDFs:
  - Run the plotting command manually (above); ensure `matplotlib` is installed in the venv.
- Path mismatch (`label_permuted_emnist` vs `label_permuted_emnist_stats`):
  - PDFs are produced from the JSON you point to; verify the exact `{task}` subfolder created by your run and use that path in the plotting command.

---

## Precise end-to-end flow (what runs under the hood)

This section ties the baseline "UPGD FO Global" stats run to the code and the theory (Elsayed & Mahmood; HesScale notes).

### 1) Initialization
- Runner: `core/run/run_stats.py` → class `RunStats`.
- Constructs:
  - `task = tasks[task]()` (e.g., `label_permuted_emnist_stats`) with an internal `change_freq` (e.g., 2,500) defining task boundaries.
  - `learner = learners[learner](networks[network], optim_kwargs)`; for `upgd_fo_global` this is the first‑order UPGD learner (no BackPACK/HesScale extension).
  - `criterion` from `task.criterion` (extended with BackPACK only if `learner.extend` is True; baseline FO is typically False).
  - `optimizer = learner.optimizer(learner.parameters, **learner.optim_kwargs)`.
  - Metrics containers for per‑step and per‑task statistics.

### 2) Per‑sample loop (streaming setting)
For each time step `i` in `range(n_samples)`:
- Pull a sample: `input, target = next(task)` (GPU‑moved tensors).
- Zero grads, forward, compute `loss = criterion(output, target)`.
- Backward and `optimizer.step()`.
- Record per‑step statistics (see below), then every `task.change_freq` steps aggregate to per‑task arrays and reset per‑step buffers.

### 3) What UPGD FO Global does (optimizer behavior)
- UPGD core idea (Elsayed & Mahmood, "UPGD"): protect high‑utility weights, perturb low‑utility ones.
- First‑order global utility approximation used in code:
  - Maintain an EMA trace per weight: `avg_utility ← β * avg_utility + (1-β) * (-grad * weight)`.
  - Compute a global maximum utility (per step) and scale each weight's utility via a sigmoid to `[0,1]`.
  - Update rule (conceptually):
    - `w ← w - lr * (grad + noise) * (1 - scaled_utility)`
    - Important weights (utility≈1) change little; unimportant weights (utility≈0) get gradient + noise (maintain plasticity).
- Contrast to second‑order: HesScale (from `README_hessian.md`) supplies per‑parameter Hessian diagonals for second‑order variants; FO global does not use HesScale.

### 4) Metrics computed (definitions match plots)
All computed in `core/run/run_stats.py` and aggregated every task.
- Losses: mean of per‑step loss within task (`losses_per_task`).
- Accuracy (classification): fraction correct per step, averaged per task (`accuracies`).
- Plasticity (`plasticity_per_task`): for each step, after the update, recompute loss on the same input; per‑step plasticity is `clamp(1 - loss_after / max(loss_before,1e-8), 0, 1)`; averaged per task.
- Dead units (`n_dead_units_per_task`): learner/network tracks `network.activations` (unit‑level zero activations); the fraction of dead units is `dead_count / network.n_units`, averaged per task.
- Weight norms: `weight_l2_per_task`, `weight_l1_per_task` (per‑step sums across parameters; aggregated per task).
- Gradient norms: `grad_l2_per_task`, `grad_l1_per_task` (per‑step norms across parameters; aggregated per task).
- Gradient sparsity: `grad_l0_per_task` = fraction of zero gradient entries per step, averaged per task.

These arrays map to PDFs generated by `plot_experiment_stats.py`:
- `learning_curves.pdf`: loss + accuracy curves.
- `plasticity.pdf`, `dead_units.pdf`, `weight_norms.pdf`, `gradient_norms.pdf`, `gradient_sparsity.pdf`.
- `summary.pdf`: small multiples for a quick overview.

### 5) Logging schema
- Logger path format (`core/logger.py`):
  - `logs/{task}/{learner}/{network}/{optimizer_hps_kv_pairs}/{seed}.json`.
- The filename segment `optimizer_hps_kv_pairs` is created by joining key=value pairs from `learner.optim_kwargs` (e.g., `lr_0.01_sigma_0.001`).

### 6) Relation to the papers/notes
- UPGD (Elsayed & Mahmood): the FO global variant in this run uses the first‑order utility trace `-∂L/∂w * w` with EMA and global scaling to gate both gradient and perturbation.
- HesScale recap (`README_hessian.md`): diagonal Hessians w.r.t. parameters (exact last layer; diagonal backprop earlier) are used in second‑order variants, not in FO global baseline.
- Streaming learning setup (Label‑Permuted EMNIST): hundreds of task shifts; this runner measures online accuracy, plasticity, dead units, and gradient/weight norms across tasks, matching diagnostics used in the paper.

### 7) Baseline vs Input‑Aware (for parity)
- Baseline (this file): `learner=upgd_fo_global`.
- Input‑Aware counterpart adds input‑curvature gating (see `README_CURVATURE_TRACKING_GUIDE.md`); to mimic baseline with the input‑aware runner, set `--lambda_max 0.0` (so λ(x)=0) and optionally increase `--compute_curvature_every` to reduce overhead.

---

## FAQ
- How often are tasks aggregated?
  - Every `task.change_freq` steps (e.g., 2,500 for EMNIST label‑permuted), statistics are averaged into per‑task arrays and per‑step buffers are cleared.
- Why does JSON live under the learner name?
  - The logger nests by `{task}/{learner}/{network}` to keep configurations comparable across optimizers/networks.
- Can I compare to other baselines (SGD/Adam/EWC/SI/MAS/PGD)?
  - Yes—use the corresponding `test_*_emnist_stats_seed0.sh` scripts; the same plotting tool works on their JSONs.

---

## Formal equations (utility and plasticity)

Let $L(\boldsymbol{w}; \boldsymbol{x}, y)$ be the loss for parameters $\boldsymbol{w}$, sample $(\boldsymbol{x}, y)$. Denote per-parameter gradient $\boldsymbol{g} = \nabla_{\boldsymbol{w}} L$, learning rate $\alpha$, weight decay $\lambda$ (decoupled), utility momentum $\beta$, and perturbation noise $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$. Sigmoid is $\sigma(z) = 1/(1+e^{-z})$.

### First-order per-parameter utility (FO, global)
- Instantaneous utility (element-wise):
$$
u_t = -\, \boldsymbol{g}_t \odot \boldsymbol{w}_t
$$
- Exponential moving average (EMA) and bias-corrected trace:
$$
\boldsymbol{U}_t = \beta\, \boldsymbol{U}_{t-1} + (1-\beta)\, u_t,\qquad \hat{\boldsymbol{U}}_t = \frac{\boldsymbol{U}_t}{1-\beta^{t}}
$$
- Global scaling and gating (global maximum):
$$
\eta_t = \max_{i}\, \hat{U}_{t,i},\qquad \bar{U}_{t,i} = \sigma\!\left( \frac{\hat{U}_{t,i}}{\eta_t} \right) \in (0,1)
$$
- UPGD FO Global update (gradient and noise both gated):
$$
\boldsymbol{w}_{t+1} = (1 - \alpha\lambda)\,\boldsymbol{w}_t - \alpha\, (\boldsymbol{g}_t + \boldsymbol{\xi}_t) \odot (\mathbf{1} - \bar{\boldsymbol{U}}_t)
$$

### First-order (local) scaling
For local variants, replace the global max scaling by tensor-wise normalization $\mathcal{N}(\cdot)$:
$$
\bar{\boldsymbol{U}}_t = \sigma\!\big(\, \mathcal{N}(\hat{\boldsymbol{U}}_t)\,\big)
$$

### Non-protecting FO variants
Gradient is not gated; only noise is attenuated by $1-\bar{\boldsymbol{U}}_t$:
$$
\boldsymbol{w}_{t+1} = (1 - \alpha\lambda)\,\boldsymbol{w}_t - \alpha\, \boldsymbol{g}_t - \alpha\, \boldsymbol{\xi}_t \odot (\mathbf{1} - \bar{\boldsymbol{U}}_t)
$$

### Second-order per-parameter utility (HesScale)
With a diagonal Hessian approximation $\boldsymbol{h} = \operatorname{diag}(H_{\boldsymbol{w}})$ from HesScale (exact at last layer; diagonal backprop otherwise), the instantaneous second-order utility is
$$
u^{(2)}_t = \frac{1}{2}\, \boldsymbol{h}_t \odot (\boldsymbol{w}_t \odot \boldsymbol{w}_t) - \boldsymbol{g}_t \odot \boldsymbol{w}_t
$$
It is then inserted into the same EMA, scaling, and gating as above to obtain $\bar{\boldsymbol{U}}_t$ and the update.

### Plasticity (per-step, per-sample)
Let $L_{\text{before}}$ be the loss used for the backward pass at step $t$, and $L_{\text{after}}$ be the loss recomputed on the same sample after the update. With a small $\epsilon > 0$ (e.g., $10^{-8}$) for numerical stability:
$$
p_t = \operatorname{clip}\!\left(\, 1 - \frac{L_{\text{after}}}{\max(L_{\text{before}},\, \epsilon)}\,,\; 0,\; 1\,\right)
$$
Per-task plasticity is the average of $p_t$ over the task window (every `change_freq` steps).

---

# Curvature Calculation and Tracking
<!-- based on the theoretical framework from `input_aware.md` -->
This guide explains how to calculate and track input curvature in the input-aware UPGD implementation

**Updated Implementation**: This now reflects the finite differences methodology from `post_run_analysis_modified2.py` instead of the original Hutchinson trace estimator, ensuring consistency with the existing loss-of-plasticity research codebase.

## Mathematical Foundation

### What It Is Computing

The input curvature measures how "sharp" the loss landscape is with respect to input perturbations:

$$\text{Input Curvature} = \operatorname{Tr}(H_x^2) = \sum_{i,j} \left(\frac{\partial^2 \mathcal{L}}{\partial x_i \partial x_j}\right)^2$$

Where $H_x$ is the Hessian matrix of the loss with respect to the input $x$.

### Why Trace of Squared Hessian?

From Garg et al. (2024), $\operatorname{Tr}(H_x^2)$ identifies "memorized" samples:
- **High curvature** → Sample lies in a sharp region → Likely memorized/hard
- **Low curvature** → Sample lies in a smooth region → Likely generalizable

## Implementation Details

### 1. Finite Differences Method (Following Existing Codebase)

Following the methodology from `post_run_analysis_modified2.py`, we use finite differences with Rademacher random vectors to estimate input curvature:

```python
def compute_input_curvature_finite_diff(model, inputs, targets, criterion, h=1e-3, niter=10, temp=1.0):
    """
    Compute input curvature using finite differences method.
    
    This follows the approach used in the existing loss-of-plasticity codebase,
    using Rademacher random vectors and finite differences.
    
    Mathematical basis:
    Approximate curvature using gradient norm differences:
    curvature ≈ ||∇L(x + hv) - ∇L(x)||
    
    where v are Rademacher (±1) random vectors.
    """
    device = inputs.device
    model.eval()
    
    num_samples = inputs.shape[0]
    regr = torch.zeros(num_samples, device=device)
    
    # Perturb each input in niter random directions
    for _ in range(niter):
        # Generate Rademacher random vector (±1)
        v = torch.randint_like(inputs, high=2, device=device) * 2 - 1  # Rademacher (±1)
        v = h * v.float()  # Scale perturbation
        
        with torch.enable_grad():
            # Forward pass on perturbed and original inputs
            outputs_pos = model(inputs + v)
            outputs_orig = model(inputs)
            
            # Compute losses
            loss_pos = criterion(outputs_pos / temp, targets)
            loss_orig = criterion(outputs_orig / temp, targets)
            
            # Compute gradient difference (finite difference approximation)
            grad_diff = torch.autograd.grad(loss_pos - loss_orig, inputs, create_graph=False)[0]
        
        # Accumulate gradient norm per sample
        regr += grad_diff.reshape(num_samples, -1).norm(dim=1)
    
    # Return average curvature across samples and iterations
    return regr.mean().item() / niter
```

### Key Steps Explained

1. **Rademacher Vector Sampling**: $v \in \{-1, +1\}^d$ (discrete random directions)
2. **Perturbation**: Scale by step size $h$ to get $x' = x + hv$
3. **Loss Difference**: Compute $\Delta L = L(f(x + hv)) - L(f(x))$
4. **Gradient Difference**: $\nabla_x \Delta L$ using automatic differentiation
5. **Norm Accumulation**: $\|\nabla_x \Delta L\|_2$ as curvature proxy
6. **Average**: Over multiple random directions and samples

## Practical Usage

### 2. Integration in Training Loop

```python
# Actual implementation in run_stats_with_curvature.py
for i in range(self.n_samples):
    input, target = next(self.task)
    input, target = input.to(self.device), target.to(self.device)
    optimizer.zero_grad()
    
    output = self.learner.predict(input)
    loss = criterion(output, target)
    
    # Compute input curvature for input-aware learners
    current_curvature = 0.0
    current_lambda = 0.0
    if self.is_input_aware and i % self.compute_curvature_every == 0:
        try:
            # Compute input curvature using the new finite differences method
            current_curvature = self.learner.compute_input_curvature(
                model=self.learner.network, 
                input_batch=input, 
                targets=target, 
                criterion=criterion
            )
            
            # Update optimizer with curvature
            self.learner.update_optimizer_curvature(current_curvature)
            
            # Get current lambda value from optimizer
            if hasattr(optimizer, 'compute_lambda'):
                current_lambda = optimizer.compute_lambda()
                
            input_curvature_per_step.append(current_curvature)
            lambda_values_per_step.append(current_lambda)
            
            print(f"Step {i}: Input curvature = {current_curvature:.6f}, Lambda = {current_lambda:.6f}")
            
        except Exception as e:
            print(f"Warning: Could not compute curvature at step {i}: {e}")
            input_curvature_per_step.append(0.0)
            lambda_values_per_step.append(0.0)
    
    # Continue with regular training step
    loss.backward()
    optimizer.step()
```

### 3. Dynamic Regularization

The curvature is used to compute dynamic protection strength:

```python
def compute_lambda(self):
    """Convert input curvature to protection strength."""
    # Sigmoid mapping: λ(x) = λ_max * sigmoid((tr(H_x^2) - τ) / α)
    threshold = self.param_groups[0]['curvature_threshold']
    lambda_max = self.param_groups[0]['lambda_max']
    lambda_scale = self.param_groups[0]['lambda_scale']
    
    normalized_curvature = (self.current_input_curvature - threshold) / lambda_scale
    lambda_value = lambda_max * torch.sigmoid(torch.tensor(normalized_curvature)).item()
    
    return lambda_value
```

## Tracking and Analysis

### 4. What Gets Tracked

The enhanced statistics collection tracks:

```python
# Per-step tracking
input_curvature_per_step = []     # Raw tr(H_x^2) values
lambda_values_per_step = []       # Dynamic λ(x) values

# Per-task aggregation
input_curvature_per_task = []     # Average curvature per task
lambda_values_per_task = []       # Average protection per task  
avg_curvature_per_task = []       # EMA curvature from optimizer
```

### 5. Logged Metrics

The following are saved to logs for analysis:

- `input_curvature_per_task`: Average input curvature per task
- `lambda_values_per_task`: Average protection strength per task
- `avg_curvature_per_task`: Running average from optimizer
- `compute_curvature_every`: Computation frequency

## Hyperparameters for Curvature Tracking

### Core Parameters

```bash
--curvature_threshold 0.07      # τ: Threshold for "high curvature"
--lambda_max 1.0                # Maximum protection strength
--lambda_scale 1.0              # α: Curvature-to-lambda mapping scale
--hutchinson_samples 10         # Number of random directions (niter) for estimation
--compute_curvature_every 1     # Frequency of curvature computation
```

### Parameter Selection Guidelines

1. **`curvature_threshold`**: 
   - Start with dataset-specific values (0.01-1.0)
   - Lower → more samples considered "high curvature"
   - Higher → fewer samples trigger protection

2. **`hutchinson_samples`** (now `niter` in finite differences):
   - More directions → better estimation, higher cost
   - 1-3 directions often sufficient for relative ranking
   - 10+ directions for precise absolute values
   - Each direction requires 2 forward passes (perturbed + original)

3. **`compute_curvature_every`**:
   - 1 → compute every step (accurate but expensive)
   - 10 → compute every 10 steps (cheaper but less precise)

4. **Finite differences specific parameters** (hardcoded in implementation):
   - `h=1e-3`: Perturbation step size (follows existing codebase)
   - `temp=1.0`: Temperature scaling for softmax (for numerical stability)

## Running with Curvature Tracking

### Command Line Example

```bash
python3 core/run/run_stats_with_curvature.py \
    --task label_permuted_emnist_stats \
    --learner input_aware_upgd_fo_global \
    --seed 0 \
    --lr 0.01 \
    --sigma 0.001 \
    --network fully_connected_relu_with_hooks \
    --n_samples 1000000 \
    --curvature_threshold 0.07 \
    --lambda_max 1.0 \
    --hutchinson_samples 10 \
    --compute_curvature_every 1
```

### Using the SLURM Script

```bash
sbatch test_inputaware_emnist_stats.sh
```

## Interpreting Results

### Expected Patterns

1. **High Curvature Samples**:
   - Mislabeled examples
   - Boundary/ambiguous samples  
   - Rare/long-tail samples

2. **Curvature Evolution**:
   - Early training: High variability
   - Later training: More stable, lower values
   - Task changes: Temporary spikes

3. **Lambda Response**:
   - High curvature → λ ≈ λ_max → Strong protection
   - Low curvature → λ ≈ 0 → Free learning

### Analysis Workflow

1. **Load logged data** from `logs/` directory
2. **Plot curvature over time** to see evolution
3. **Correlate with performance** metrics (accuracy, plasticity)
4. **Identify high-curvature samples** for manual inspection
5. **Analyze lambda distribution** to understand protection patterns

## Computational Considerations

### Cost Analysis

- **Base cost**: 1 forward + 1 backward pass
- **Curvature cost**: +1 forward + n_samples × (1 gradient + 1 HVP)
- **Total overhead**: ≈ 2-4× base cost depending on n_samples

### Optimization Strategies

1. **Reduce frequency**: `compute_curvature_every > 1`
2. **Fewer samples**: `hutchinson_samples = 1-3` for ranking
3. **Batch processing**: Compute curvature on mini-batches
4. **Early stopping**: Skip curvature in later stages if stable

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `hutchinson_samples` or `compute_curvature_every`
2. **NaN values**: Check for numerical instability in loss computation
3. **Zero curvature**: Verify `input.requires_grad_(True)` is set
4. **Performance impact**: Increase `compute_curvature_every` to reduce cost

### Debug Tips

```python
# Add to track intermediate values
print(f"Loss: {loss.item():.6f}")
print(f"Input grad norm: {torch.norm(grad_loss).item():.6f}")
print(f"HVP norm: {torch.norm(hvp).item():.6f}")
print(f"Curvature: {current_curvature:.6f}")
```

### Curvature-Specific Formal Equations

#### Input-Aware Utility Gating
The input-aware UPGD variant modifies the standard utility with curvature-based protection:

$$
\bar{\boldsymbol{U}}_{t,i} = \sigma\!\left( \frac{\hat{U}_{t,i}}{\eta_t} \right) \cdot \left(1 + \lambda(x_t)\right)
$$

where $\lambda(x_t)$ is the input-dependent protection strength.

#### Sigmoid Protection Mapping
The curvature-to-protection mapping uses a sigmoid transformation:

$$
\lambda(x_t) = \lambda_{\max} \cdot \sigma\!\left( \frac{\operatorname{Tr}(H_{x_t}^2) - \tau}{\alpha} \right)
$$

where:
- $\tau$ is the curvature threshold (`curvature_threshold`)
- $\alpha$ is the mapping scale (`lambda_scale`) 
- $\lambda_{\max}$ is the maximum protection strength (`lambda_max`)

#### Finite Differences Approximation
The input curvature is approximated using finite differences with Rademacher vectors:

$$
\operatorname{Tr}(H_x^2) \approx \frac{1}{N} \sum_{j=1}^{N} \left\| \nabla_x \left[ L(f(x + h \boldsymbol{v}_j)) - L(f(x)) \right] \right\|_2
$$

where $\boldsymbol{v}_j \in \{-1, +1\}^d$ are Rademacher random vectors and $h$ is the perturbation step size.

This comprehensive tracking allows you to test the core hypothesis from `input_aware.md`: that memorization of hard samples (high curvature) drives loss of plasticity, and that input-aware protection can mitigate this effect.
