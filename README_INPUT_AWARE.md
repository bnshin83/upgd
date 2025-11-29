# Input-Aware UPGD Implementation

This implementation adds input-aware optimization to the UPGD codebase based on the theoretical framework described in `input_aware.md`. The key idea is to modulate parameter protection based on input-space curvature, protecting important parameters more strongly when processing "hard" (high-curvature) samples.

## Theory Overview

The implementation combines two complementary measures of curvature:
- **Parameter curvature** (from UPGD/HesScale): Per-weight utility/importance 
- **Input curvature** (from Garg et al.): Per-sample memorization score via $\operatorname{Tr}(H_x^2)$

High input curvature indicates "hard" samples that could damage specialized knowledge. The optimizer dynamically adjusts protection strength based on this signal.

## Files Created

### 1. Core Optimizer Implementation
- **`core/optim/weight_upgd/input_aware.py`**
  - `hutchinson_trace_estimator()`: Estimates $\operatorname{Tr}(H_x^2)$ using Hutchinson's method
  - `InputAwareFirstOrderGlobalUPGD`: First-order input-aware UPGD optimizer
  - `InputAwareSecondOrderGlobalUPGD`: Second-order version using HesScale

### 2. Learner Classes
- **`core/learner/input_aware_upgd.py`**
  - `InputAwareFirstOrderGlobalUPGDLearner`: Wrapper for first-order optimizer
  - `InputAwareSecondOrderGlobalUPGDLearner`: Wrapper for second-order optimizer
  - Methods for computing and updating input curvature

### 3. Modified Run Classes
- **`core/run/input_aware_run.py`**
  - `InputAwareRun`: Modified training loop that computes input curvature
  - Handles curvature computation frequency and logging
- **`core/run/run_stats_with_curvature.py`**
  - `RunStatsWithCurvature`: Enhanced statistics collection with curvature tracking
  - Logs input curvature values, lambda values, and average curvature per task

### 4. Experiment Script
- **`experiments/label_permuted_emnist_input_aware.py`**
  - Complete experiment setup with grid search
  - Includes input-aware specific hyperparameters
  - Comparison with baseline methods

### 5. Test Scripts
- **`test_input_aware.py`**
  - Unit tests for Hutchinson estimator
  - Integration tests for optimizer
  - Verification of the complete pipeline
- **`test_inputaware_emnist_stats.sh`**
  - SLURM script for running input-aware experiments with curvature tracking
  - Uses `run_stats_with_curvature.py` for comprehensive statistics collection

## Key Features

### Dynamic Regularization
The optimizer computes a dynamic regularization strength $\lambda(x)$ based on input curvature:

$$\lambda(x) = \lambda_{\text{max}} \cdot \text{sigmoid}\left(\frac{\operatorname{Tr}(H_x^2) - \tau}{\alpha}\right)$$

Where:
- $\tau$: Curvature threshold
- $\lambda_{\text{max}}$: Maximum regularization strength  
- $\alpha$: Scaling factor

### Protection Mechanism
When processing high-curvature samples:
1. Input curvature $\operatorname{Tr}(H_x^2)$ is computed via Hutchinson estimator
2. Dynamic $\lambda$ is calculated based on curvature
3. Important parameters (high utility) are protected proportionally to $\lambda$
4. Low-utility parameters remain plastic for new learning

### Curvature Tracking
The enhanced statistics collection (`run_stats_with_curvature.py`) tracks:
1. **Input curvature per sample**: $\operatorname{Tr}(H_x^2)$ values computed via Hutchinson estimator
2. **Dynamic lambda values**: Current regularization strength $\lambda(x)$ per sample
3. **Average curvature**: Running exponential moving average of input curvature
4. **Curvature statistics per task**: Aggregated curvature metrics for analysis

Key tracked metrics:
- `input_curvature_per_task`: Average input curvature per task
- `lambda_values_per_task`: Average protection strength per task  
- `avg_curvature_per_task`: Optimizer's running average curvature per task

## Hyperparameters

### Standard UPGD Parameters
- `lr`: Learning rate
- `beta_utility`: Momentum for utility tracking
- `sigma`: Noise standard deviation
- `weight_decay`: L2 regularization

### Input-Aware Specific Parameters
- `curvature_threshold`: Threshold for "high curvature" samples
- `lambda_max`: Maximum protection strength
- `lambda_scale`: Curvature-to-lambda mapping scale
- `beta_curvature`: Momentum for curvature tracking
- `hutchinson_samples`: Number of random vectors for trace estimation
- `compute_curvature_every`: Frequency of curvature computation

## Usage

### Basic Usage
```python
from core.learner.input_aware_upgd import InputAwareFirstOrderGlobalUPGDLearner
from core.network.fcn_relu import FullyConnectedReLU

learner = InputAwareFirstOrderGlobalUPGDLearner(
    network=FullyConnectedReLU(),
    optim_kwargs={
        'lr': 0.01,
        'beta_utility': 0.99,
        'sigma': 0.1,
        'curvature_threshold': 1.0,
        'lambda_max': 0.5,
        'hutchinson_samples': 3
    }
)
```

### Running Experiments
```bash
# Generate experiment scripts
python experiments/label_permuted_emnist_input_aware.py

# Run all experiments
cd generated_cmds_input_aware/label_permuted_emnist/
./run_label_permuted_emnist.sh

# Or run specific learner
./generate_label_permuted_emnist_input_aware_upgd_fo_global.sh
```

### Running with Curvature Tracking
```bash
# Run with comprehensive curvature statistics
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

# Or use the provided SLURM script
sbatch test_inputaware_emnist_stats.sh
```

## Implementation Details

### Curvature Computation
The Hutchinson estimator approximates $\operatorname{Tr}(H_x^2)$ efficiently:
1. Sample random vector $v \sim \mathcal{N}(0, I)$
2. Compute Hessian-vector product $Hv$ via autodiff
3. Estimate: $\operatorname{Tr}(H_x^2) \approx \mathbb{E}[\|Hv\|^2]$

### Integration with UPGD
The input-aware component modulates the existing UPGD update:
- Low curvature → $\lambda \approx 0$ → Standard UPGD behavior
- High curvature → $\lambda \approx \lambda_{\text{max}}$ → Strong protection of important weights

### Computational Overhead
- Hutchinson estimation adds one extra forward-backward pass
- Can be computed every N steps to reduce overhead
- Second-order version reuses HesScale computations

## Experimental Setup

The provided experiment script compares:
1. Standard UPGD (first-order, global)
2. Non-protecting UPGD variant
3. **Input-aware UPGD (first-order)**
4. **Input-aware UPGD (second-order with HesScale)**
5. Baseline methods (SGD, PGD, Shrink & Perturb)

Grid search explores:
- Different curvature thresholds
- Regularization strengths
- Hutchinson sample counts
- Computation frequencies

## Expected Benefits

1. **Better plasticity-stability tradeoff**: Protects knowledge selectively
2. **Reduced catastrophic forgetting**: Preserves important rare patterns
3. **Adaptive learning**: Adjusts protection dynamically per sample
4. **Computational efficiency**: Protection only when needed

## Testing

Run the test suite to verify installation:
```bash
python test_input_aware.py
```

This tests:
- Hutchinson estimator accuracy
- Optimizer functionality
- Full integration pipeline

## Citation

This implementation is based on:
- Elsayed & Mahmood (2022): HesScale for parameter curvature
- Garg et al. (2024): Input curvature for memorization detection
- UPGD framework: Utility-based gradient perturbation

## Notes

- Input curvature computation requires gradient tracking on inputs
- Second-order version requires BackPACK extensions
- Curvature can be computed less frequently to save computation
- Protection strength should be tuned based on task difficulty

## summary
I'll help you implement the input-aware optimization strategy from `input_aware.md` into the label_permuted_emnist experiment. Let me first explore the codebase structure to understand the existing implementation.

[6 tools called]

Now let me look at how the UPGD optimizers are implemented to understand how to add input-aware functionality:

[1 tool called]

Now let me check how the second-order UPGD works since it might be using curvature information already:

[1 tool called]

Now let me check how EWC is implemented to understand how to add regularization terms:

[2 tools called]

Now I'll create the implementation for input-aware UPGD. Let me create the necessary files:

[3 tools called]

Now let me create the learner class:

[3 tools called]

Now I need to create a modified run class that handles input-aware learners:

[2 tools called]

Now let me update the utils to register the new learners:

[5 tools called]

Now let me create the updated experiment script with input-aware learners:

[3 tools called]

Now let me check for any linting errors and create a simple test script:

[1 tool called]

Now let me create a simple test script to verify the implementation works:

[2 tools called]

Let me create a summary README documenting the implementation:

[1 tool called]

## Summary

I've successfully implemented the input-aware optimization strategy from your `input_aware.md` document into the UPGD codebase. Here's what was created:

### 🎯 **Core Implementation**

1. **Input-Aware Optimizers** (`core/optim/weight_upgd/input_aware.py`)
   - `InputAwareFirstOrderGlobalUPGD`: Combines UPGD's utility-based protection with input curvature gating
   - `InputAwareSecondOrderGlobalUPGD`: Uses HesScale for parameter curvature + input curvature
   - `hutchinson_trace_estimator`: Efficiently computes Tr(H_x^2) for memorization scoring

2. **Learner Classes** (`core/learner/input_aware_upgd.py`)
   - Wrappers that handle input curvature computation during training
   - Methods to update optimizer with current sample's curvature

3. **Modified Training Loop** (`core/run/input_aware_run.py`)
   - Computes input curvature before parameter updates
   - Handles frequency of computation and logging

### 🔧 **Key Features**

The implementation realizes the theoretical framework from `input_aware.md`:

- **Dynamic Protection**: $\lambda(x) = \lambda_{\text{max}} \cdot \text{sigmoid}((\operatorname{Tr}(H_x^2) - \tau) / \alpha)$
  - Low curvature (easy samples) → $\lambda \approx 0$ → Free learning
  - High curvature (hard samples) → $\lambda \approx \lambda_{\text{max}}$ → Protect important weights

- **Curvature-Gated Updates**: Combines:
  - Parameter utility (from UPGD/HesScale): Which weights are important
  - Input curvature (Hutchinson estimator): When protection is needed

### 📊 **Experiment Setup**

Created `experiments/label_permuted_emnist_input_aware.py` with:
- Grid search over input-aware hyperparameters:
  - `curvature_threshold`: [0.1, 1.0, 10.0]
  - `lambda_max`: [0.1, 0.5, 1.0]
  - `hutchinson_samples`: [1, 3]
  - `compute_curvature_every`: [1, 10]
- Comparison with standard UPGD and baseline methods

### 🚀 **Usage**

```python
# Run experiments
python experiments/label_permuted_emnist_input_aware.py

# This generates scripts in generated_cmds_input_aware/
cd generated_cmds_input_aware/label_permuted_emnist/
./run_label_permuted_emnist.sh
```

### 💡 **How It Works**

1. **Per-sample**: Compute input curvature $\operatorname{Tr}(H_x^2)$ via Hutchinson
2. **Dynamic $\lambda$**: Map curvature to protection strength
3. **Selective protection**: High-utility weights × high-curvature samples = strong protection
4. **Maintain plasticity**: Low-utility weights remain adaptable

This implementation directly addresses the hypothesis from `input_aware.md`: memorization of hard samples causes loss of plasticity by concentrating curvature on specific weights. The input-aware approach protects these critical weights only when processing risky (high-curvature) samples, maintaining plasticity otherwise.

**Note**: The mathematical formulation above is specific to input-aware UPGD variants, which use multiplicative weight decay `θₜ(1-αβ)`. Other optimizers in this codebase (SGD, Adam) use additive weight decay `θₜ - αβθₜ`, which is mathematically equivalent but implemented differently.



## Core Update Rule for Input-Aware UPGD

The input-aware UPGD parameter update combines utility-based protection with dynamic input-curvature gating. The update rule is:

$$
\theta_{t+1} = \theta_t (1 - \alpha \beta) - \alpha \left[ (g_t + R_t + \xi_t) \odot (1 - u_t \lambda(x_t)) \right]
$$

Where:
- $\theta_t$: parameters at step $t$
- $\alpha$: learning rate
- $g_t$: gradient
- $R_t$: regularization term
- $\xi_t$: exploration noise
- $u_t$: normalized utility (importance)
- $\lambda(x_t)$: dynamic protection strength based on input curvature
- $\beta$: weight decay coefficient (multiplicative form used in UPGD variants)

### Comparison to Standard First-Order UPGD

For reference, the standard first-order UPGD update (without input-awareness) is:

$$
\theta_{t+1} = \theta_t (1 - \alpha \beta) - \alpha \left[ (g_t + \xi_t) \odot (1 - u_t) \right]
$$

- No input curvature gating: $\lambda(x_t)$ is absent (effectively always 1).
- No regularization term $R_t$ (no explicit pull toward $\theta_0$).
- Gating is based only on parameter utility $u_t$.

### Key Differences and Why Both Mechanisms Are Needed:

1. **Regularization Term ($R_t$):**  
   - *What it does:* Input-aware UPGD introduces an explicit regularization term, $R_t = \lambda(x_t) u_t (\theta_t - \theta_0)$, which pulls important parameters back toward their initial values.
   - *When it acts:* This regularization is only strong when the input is "risky" (i.e., has high curvature, so $\lambda(x_t)$ is large) **and** the parameter is important ($u_t$ is large).
   - *Why it's needed:* This prevents important parameters from drifting too far from their original, task-specialized values, especially when encountering hard or ambiguous samples.

   - *Reference: Regularizing toward random parameters.*  
     With L2 Init, regularization is applied toward the specific fixed parameters $\theta_0$ sampled at initialization. Alternatively, following a procedure more similar to Shrink & Perturb, a new set of parameters could be sampled at each time step. In this case, $\phi_t$ is sampled from the same distribution as $\theta_0$, and the regularization term becomes $\|\theta_t - \phi_t\|_2^2$. As shown in [Kumar et al., 2024](./@kumar-et-al-2024---_maintaining-plasticity-in-continual-learnin-c3a4728c-fc72-4464-8d0e-09331754cf74.md), Figure 4, performance is compared between L2 Init and this variant (L2 Init + Resample) on Permuted MNIST, Random Label MNIST, and 5+1 CIFAR using the Adam optimizer. The best regularization strength for each method is selected using the same hyper-parameter sweep as for L2 Init. Results indicate that regularizing toward the initial parameters rather than sampling a new set of parameters at each time step yields much better performance.

   - *Choice of norm.*  
     While L2 Init uses the L2 norm, the L1 norm of the difference between the parameters and their initial values can also be used. This approach, called L1 Init, uses the following loss function:
     $$
     L_{\text{reg}}(\theta) = L_{\text{train}}(\theta) + \lambda \|\theta - \theta_0\|_1
     $$
     In [Kumar et al., 2024](./@kumar-et-al-2024---_maintaining-plasticity-in-continual-learnin-c3a4728c-fc72-4464-8d0e-09331754cf74.md), Figure 3, the performance of L2 Init and L1 Init is compared on Permuted MNIST, Random Label MNIST, and 5+1 CIFAR with the Adam optimizer. While L1 Init mitigates plasticity loss, performance is worse on Permuted MNIST and 5+1 CIFAR.

2. **Lambda-Gated Update Dampening:**  
   - *What it does:* The update is multiplicatively dampened by $(1 - u_t \lambda(x_t))$ instead of just $(1 - u_t)$ as in standard UPGD.
   - *Why it's needed:* This means that the optimizer reduces the magnitude of **all** updates (gradient, regularization, and noise) for important parameters, but only when the input is risky. This selective dampening preserves knowledge by making important parameters less responsive to new, potentially disruptive information.

3. **Dynamic, Input-Dependent Protection ($\lambda(x_t)$):**  
   - *What it does:* The protection strength $\lambda(x_t)$ is computed dynamically for each input, based on its curvature.
   - *Why it's needed:* This allows the optimizer to adapt on-the-fly: strong protection is applied only when needed (e.g., for outliers, mislabeled, or boundary samples), and plasticity is maintained for easy or typical samples.

**Why both regularization and gating are necessary:**  
- *Regularization* ($R_t$) provides a directional pull, actively restoring important parameters toward their initial state when at risk.
- *Lambda-gated dampening* controls the overall magnitude of updates, making important parameters less plastic only when the input is risky.
- **Together**, these mechanisms ensure that the optimizer can both prevent catastrophic forgetting (by protecting knowledge) and remain adaptable (by not over-regularizing on easy inputs).


---
### Key Components

**1. Input Curvature Estimation**

Using finite differences (see `core/optim/weight_upgd/input_aware.py`):

$$
\text{Curvature}(x) = \frac{1}{N} \sum_{i=1}^{N} \left\| \nabla_x [L(f(x + hv_i)) - L(f(x))] \right\|
$$

Where:
- $v_i \sim \text{Rademacher}(\pm 1)$ (random directions)
- $h = 10^{-3}$ (perturbation size)
- $N = 10$ (number of directions/samples)

**2. Dynamic Protection Strength**

$$
\lambda(x_t) = \lambda_{\max} \cdot \sigma\left(\frac{\text{Curvature}(x_t) - \tau}{\lambda_{\text{scale}}}\right)
$$

Where:
- $\tau$: curvature threshold
- $\lambda_{\max}$: maximum regularization strength.
  - **Typical range:** 0.1–5.0 (default: 1.0)
  - **Smaller values (0.1–0.5):** Light regularization, less protection.
  - **Medium values (0.5–2.0):** Moderate protection.
  - **Larger values (2.0–5.0):** Strong protection, but may risk over-regularization.
- $\lambda_{\text{scale}}$: scaling factor controlling the sensitivity of the sigmoid.
  - **Typical range:** 0.01–10.0 (default: 0.1)
  - **Smaller $\lambda_{\text{scale}}$ (0.01–0.1):** Very sensitive, steeper sigmoid, protection rapidly switches from off to on as curvature crosses $\tau$ (almost binary gating).
  - **Medium $\lambda_{\text{scale}}$ (0.1–1.0):** Balanced sensitivity, smooth but responsive transition.
  - **Larger $\lambda_{\text{scale}}$ (1.0–10.0):** Gentler sigmoid, more gradual transition in protection strength as curvature increases.
  - **Interpretation:** Lower values make the optimizer highly responsive to small changes in curvature; higher values make protection ramp up more slowly.
- **Typical combinations:**
  - *Conservative:* $\lambda_{\max}=0.5$, $\lambda_{\text{scale}}=1.0$ (gentle, gradual protection)
  - *Balanced:* $\lambda_{\max}=1.0$, $\lambda_{\text{scale}}=0.1$ (moderate, responsive)
  - *Aggressive:* $\lambda_{\max}=2.0$, $\lambda_{\text{scale}}=0.05$ (strong, very sensitive)
- $\sigma(\cdot)$: sigmoid function

**3. Utility Computation**

- *First-order version (InputAwareFirstOrderGlobalUPGD):*
  $$
  u_t^{(i)} = \text{sigmoid}\left(\frac{\bar{U}_t^{(i)}}{\max_j \bar{U}_t^{(j)}}\right)
  $$
  $$
  \bar{U}_t^{(i)} = \beta_u \bar{U}_{t-1}^{(i)} + (1-\beta_u)(-g_t^{(i)} \theta_t^{(i)})
  $$

- *Second-order version (InputAwareSecondOrderGlobalUPGD):*
  $$
  \bar{U}_t^{(i)} = \beta_u \bar{U}_{t-1}^{(i)} + (1-\beta_u)\left(\frac{1}{2}H_{ii}(\theta_t^{(i)})^2 - g_t^{(i)} \theta_t^{(i)}\right)
  $$
  Where $H_{ii}$ is the Hessian diagonal from HesScale.

**4. Regularization Term and Protection Mechanism**

$$
R_t = \lambda(x_t) \cdot u_t \odot (\theta_t - \theta_0)
$$

The input-aware UPGD employs **dual protection mechanisms**:

- **λ(x_t) - The "Gating Signal":**
  - Dynamic, input-dependent, ranges from 0 to $\lambda_{\max}$
  - Low curvature input → $\lambda \approx 0$ → "Safe sample, allow plasticity"
  - High curvature input → $\lambda \approx \lambda_{\max}$ → "Risky sample, protect knowledge"

- **R_t - The "Protection Force":**
  - Pulls parameters toward their initial values $\theta_0$
  - Strength: $\lambda(x_t) \times u_t \times (\theta_t - \theta_0)$

  - *Alternative regularization targets:*  
    The regularization term can be generalized to pull toward a different set of parameters. For example, at each time step, $\phi_t$ could be sampled from the same distribution as $\theta_0$ and $R_t = \lambda(x_t) u_t (\theta_t - \phi_t)$ used, i.e., regularizing toward random parameters at each step ([Kumar et al., 2024](./@kumar-et-al-2024---_maintaining-plasticity-in-continual-learnin-c3a4728c-fc72-4464-8d0e-09331754cf74.md), Fig. 4). However, regularizing toward the fixed initialization $\theta_0$ (L2 Init) is found to be more effective in practice.

  - *Alternative norms:*  
    The regularization can also use the L1 norm, i.e., $R_t = \lambda(x_t) u_t \cdot \text{sign}(\theta_t - \theta_0)$, as in L1 Init ([Kumar et al., 2024](./@kumar-et-al-2024---_maintaining-plasticity-in-continual-learnin-c3a4728c-fc72-4464-8d0e-09331754cf74.md), Fig. 3). While L1 regularization can mitigate plasticity loss, it generally underperforms L2 Init on challenging benchmarks.

- **Combined:**
  1. **Additive Regularization**: $R_t$ adds a restorative force toward $\theta_0$ (only when $\lambda(x_t) > 0$)
  2. **Multiplicative Gating**: The update is scaled by $(1 - u_t \lambda(x_t))$, reducing the update for important parameters on risky samples.

**Example Scenarios:**
- **Safe sample** ($\lambda \approx 0$): $R_t \approx 0$, gating $\approx 1$ → Normal gradient descent (like standard UPGD)
- **Risky sample + Important parameter** ($\lambda \approx \lambda_{\max}$, $u_t \approx 1$): Strong pull toward $\theta_0$ and heavily dampened update → Parameter preservation

  **5. Exploration Noise**

  $$
  \xi_t \sim \mathcal{N}(0, \sigma^2 I)
  $$

  ---
  ## Complete Algorithm

  The input-aware UPGD alternates between:

  1. **Curvature computation** (every $K$ steps):
     $$
     \text{Curvature}(x_t) = \text{finite\_diff\_estimate}(x_t)
     $$
  2. **Protection strength update:**
     $$
     \lambda(x_t) = \lambda_{\max} \sigma\left(\frac{\text{Curvature}(x_t) - \tau}{\alpha_s}\right)
     $$
  3. **Parameter update** (UPGD-specific multiplicative weight decay):
     $$
     \theta_{t+1} = (1-\alpha\beta)\theta_t - \alpha[(g_t + \lambda(x_t) u_t (\theta_t - \theta_0) + \xi_t) \odot (1 - u_t \lambda(x_t))]
     $$

  ---
  ## Adaptive Behavior

  - **Low curvature** ($\text{Curvature}(x_t) < \tau$): $\lambda(x_t) \approx 0$ → Reduces to standard UPGD (no extra protection, full plasticity)
  - **High curvature** ($\text{Curvature}(x_t) \gg \tau$): $\lambda(x_t) \approx \lambda_{\max}$ → Strong protection of important weights, both via regularization and gating

  ---
  ### 🔍 **Summary of Differences: Input-Aware UPGD vs. Standard UPGD**

  | Aspect                | Standard UPGD                                 | Input-Aware UPGD                                              |
  |-----------------------|-----------------------------------------------|---------------------------------------------------------------|
  | Gating                | $(1 - u_t)$                                   | $(1 - u_t \lambda(x_t))$                                      |
  | Regularization        | None                                          | $R_t = \lambda(x_t) u_t (\theta_t - \theta_0)$                |
  | Input Curvature       | Not used                                      | Used to modulate protection ($\lambda(x_t)$)                  |
  | Plasticity/Protection | Fixed by $u_t$                                | Dynamic, depends on both $u_t$ and input curvature            |
  | Behavior on "easy"    | Always plastic for low-utility params         | Same as UPGD (if $\lambda(x_t) \approx 0$)                    |
  | Behavior on "hard"    | Always protects important params              | Only protects important params on risky (high-curvature) input|

  This dual mechanism ensures that important parameters are only protected when the input is likely to cause catastrophic forgetting, maintaining plasticity otherwise.

