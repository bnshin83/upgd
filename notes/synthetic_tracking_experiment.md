# Synthetic Parameter Tracking Experiment for UPGD

**Date:** February 3, 2026  
**Purpose:** A simple synthetic experiment to evaluate UPGD's ability to track non-stationary targets.

---

## Mathematical Formulation

### 1. Parameter Evolution (Linear Dynamical System)

The true parameters evolve according to linear dynamics:

$$\theta_{t+1} = A \cdot \theta_t + \zeta_t$$

Where:
- $\theta_t = \begin{bmatrix} \phi_t \\ w_t \end{bmatrix} \in \mathbb{R}^{d}$ is the true parameter vector
- $A \in \mathbb{R}^{d \times d}$ is the transition matrix
- $\zeta_t \sim \mathcal{N}(0, \sigma_\zeta^2 I)$ is optional process noise

**Example A matrix (2D case):**
$$A = \begin{bmatrix} 1 & 3 \\ -2 & 1 \end{bmatrix}$$

> [!NOTE]
> The eigenvalues of $A$ determine the dynamics:
> - $|\lambda| < 1$: contracting (stable)
> - $|\lambda| = 1$: oscillatory
> - $|\lambda| > 1$: expanding (unstable)

---

### 2. Observation Model (Linear Regression)

At each time step $t$, we observe data samples $(m_i, y_i)$ where:

$$y_i = m_i^\top \theta_t + \epsilon_i$$

The feature vector decomposes as:
$$m_i = \begin{bmatrix} m_{\phi,i} \\ m_{w,i} \end{bmatrix}$$

So the observation becomes:
$$y_i = m_{\phi,i}^\top \phi_t + m_{w,i}^\top w_t + \epsilon_i$$

Where:
- $m_i \in \mathbb{R}^d$ is the input/feature vector (sampled randomly)
- $y_i \in \mathbb{R}$ is the scalar target
- $\epsilon_i \sim \mathcal{N}(0, \sigma_\epsilon^2)$ is observation noise

---

## Experimental Design

### Objective
Evaluate how well different optimizers can **track** the evolving true parameters $\theta_t$ over time.

### Optimizers to Compare
| Optimizer | Description |
|-----------|-------------|
| **SGD** | Baseline stochastic gradient descent |
| **Adam** | Adaptive learning rate baseline |
| **UPGD (Full)** | Utility-based plasticity on all parameters |
| **UPGD (Output Only)** | Utility-based plasticity on output layer only |
| **S&P** | Shrink and Perturb baseline |

### Loss Function
Mean squared error between predictions and observations:
$$\mathcal{L}(\hat{\theta}_t) = \frac{1}{n} \sum_{i=1}^{n} (m_i^\top \hat{\theta}_t - y_i)^2$$

Where $\hat{\theta}_t$ is the learner's current parameter estimate.

---

## Metrics

1. **Parameter Tracking Error**: $\|\hat{\theta}_t - \theta_t\|_2$
2. **Prediction Error (MSE)**: $\mathbb{E}[(m^\top \hat{\theta}_t - y)^2]$
3. **Utility Distribution**: How UPGD assigns utility to parameters
4. **Adaptation Speed**: How quickly the learner recovers after a parameter shift

---

## Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Dimension | $d$ | 10 | Size of parameter vector |
| Batch size | $n$ | 32 | Samples per time step |
| Learning rate | $\eta$ | 0.01 | Optimizer step size |
| Process noise | $\sigma_\zeta$ | 0.0 | Std of parameter evolution noise |
| Observation noise | $\sigma_\epsilon$ | 0.1 | Std of observation noise |
| Total steps | $T$ | 10000 | Number of time steps |
| A matrix type | - | rotating | Type of dynamics (rotating, contracting, etc.) |

---

## Implementation Notes

### Data Generation (per step)
```python
# Parameter evolution
theta_true = A @ theta_true + noise_zeta

# Generate batch of observations
M = np.random.randn(batch_size, d)  # features
y = M @ theta_true + noise_epsilon   # targets
```

### Learning Update (per step)
```python
# Forward pass
y_pred = M @ theta_hat

# Compute loss and gradients
loss = 0.5 * np.mean((y_pred - y)**2)
grad = M.T @ (y_pred - y) / batch_size

# Optimizer step
theta_hat = optimizer.step(theta_hat, grad)
```

---

## Why This Experiment for UPGD?

1. **Non-stationarity**: The true parameters keep changing, testing plasticity
2. **Simplicity**: Linear regression is well-understood, easy to analyze
3. **Interpretability**: Can directly measure distance to ground truth
4. **Controllable difficulty**: Adjust $A$ matrix to control rate of change
5. **Utility relevance**: Parameters become "stale" as dynamics evolve

### Key Hypothesis
UPGD should maintain better tracking performance than SGD/Adam because:
- It identifies which parameters are "useful" vs "stale"
- It can reallocate learning capacity to track changing parameters
- Standard optimizers may get stuck with outdated parameter estimates

---

## Variations to Explore

1. **Abrupt changes**: Periodically reset $\theta_t$ to test adaptation
2. **Partial observability**: Only observe subset of parameters
3. **Different A dynamics**: Rotating, expanding, contracting
4. **Varying noise levels**: Test robustness
5. **Different dimensions**: Scale from 2D to high-dimensional

---

## TODO

- [ ] Implement basic experiment script
- [ ] Add UPGD optimizer integration
- [ ] Set up logging (WandB/TensorBoard)
- [ ] Run baseline comparisons
- [ ] Analyze utility distributions
- [ ] Generate plots and visualizations
