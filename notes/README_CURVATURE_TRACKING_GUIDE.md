# Curvature Calculation and Tracking Guide

This guide explains how to calculate and track input curvature in the input-aware UPGD implementation, based on the theoretical framework from `input_aware.md`.

**Updated Implementation**: This guide now reflects the finite differences methodology from `post_run_analysis_modified2.py` instead of the original Hutchinson trace estimator, ensuring consistency with the existing loss-of-plasticity research codebase.

## Mathematical Foundation

### What We're Computing

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

This comprehensive tracking allows you to test the core hypothesis from `input_aware.md`: that memorization of hard samples (high curvature) drives loss of plasticity, and that input-aware protection can mitigate this effect.