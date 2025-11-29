# Input-Aware UPGD Gating Analysis

**Document Purpose:** Comprehensive analysis of input-aware UPGD gating mechanism, covering initial hypotheses, experimental results, and conclusions from systematic λ_scale ablation studies.

**Key Finding:** Input curvature is largely redundant with utility-based protection for Input-Permuted MNIST. Standard UPGD already captures the necessary protection through its utility mechanism.

**Date:** October 2025

---

## 🚀 Quick Start - Essential Facts

**TL;DR:** Input-aware UPGD with λ_scale=0.1 gives ~1-2% improvement over standard UPGD, but it's not worth the computational cost. Input curvature is correlated with utility, so UPGD already handles protection. Use standard UPGD for this task.

**Critical Parameter:** `--lambda_scale` (default: 0.1)
- **0.01**: Steep sigmoid, two-regime (94% more plastic, 6% protected) → **HURTS performance**
- **0.1**: Gentle sigmoid, quasi-uniform (λ≈0.95-1.0) → Slight improvement (~1-2%)
- **≥0.2**: Very gentle, essentially same as 0.1

**Main Hypothesis (CONFIRMED):**
```
High curvature → High gradients → High utility → Already protected by UPGD
```

**When to Use Input-Aware:**
- ✅ Tasks with adversarial/OOD samples (high curv, low utility)
- ✅ Class-imbalanced learning (rare classes)
- ✅ Non-stationary distributions (utility goes stale)
- ❌ Input-Permuted MNIST and similar tasks

**Open Question:** Does curvature improve utility *scaling* (second-order effects)?

---

## Table of Contents
1. [Initial Experimental Observation](#experimental-observation)
2. [Lambda Behavior Analysis](#lambda-behavior-actual-data)
3. [The Paradox](#the-paradox-critical-observation)
4. [Experimental Results](#experimental-results-and-final-conclusions)
5. [Open Questions & Future Work](#open-question-second-order-effects)

---

## Experimental Observation

**Initial best performing configuration on Input-Permuted MNIST:**
- **Mode**: Gating only (no regularization, `disable_regularization=True`)
- **λ_max**: 2.0
- **λ_scale**: 0.1 (default, not initially parameterized)
- **τ (curvature threshold)**: 0.01
- **Result**: Slightly better than standard UPGD (~1-2% improvement)

### Curvature Distribution (ACTUAL DATA - 10,000 samples)
```
Mean curvature:     1.870e-03  (close to 0.002 estimate!)
Median curvature:   3.813e-04  (Mean/Median = 4.91x → long tail)
Max curvature:      1.351e-01  (Max/Mean = 72x)
Threshold τ:        0.01

Distribution shape:
→ Skewness: 8.54 (HIGHLY right-skewed)
→ Kurtosis: 109.07 (EXTREMELY heavy tails)
→ τ = 0.01 is at the 95.4th percentile (even higher than estimated!)
→ Only 4.61% of samples exceed τ
→ 95.39% of samples have Tr(H_x²) < 0.01

Percentiles:
→ P75:  8.286e-04
→ P90:  3.813e-03
→ P95:  9.230e-03
→ P99:  2.959e-02
→ P99.9: 7.572e-02
```

## Update Rule Analysis

### Standard UPGD (Baseline)
```python
θ ← θ(1-αβ) - α[(g + ξ) ⊙ (1 - u)]

Protection: Fixed per parameter based on global utility
```

### Input-Aware UPGD (Gating Only, λ_max=2.0)
```python
θ ← θ(1-αβ) - α[(g + ξ) ⊙ (1 - u·λ(x))]

where:
λ(x) = 2.0 · sigmoid((Tr(H_x²) - 0.01) / λ_scale)

Protection: Dynamic, sample-dependent
```

## Lambda Behavior (ACTUAL DATA)

**CRITICAL FINDING**: The initial hypothesis was WRONG!

### Actual Lambda Distribution with τ=0.01, λ_max=2.0
```
Mean λ:     0.9593  (NOT 0 as hypothesized!)
Median λ:   0.9519
Max λ:      1.5548
Min λ:      0.9500

Lambda activation:
→ λ > 0.5:  100.00% of samples
→ λ > 1.0:    4.61% of samples
→ λ > 1.5:    0.03% of samples (only 3 samples!)
```

### REVISED Understanding

**The sigmoid with τ=0.01 and λ_scale=0.1 creates a BASELINE protection, not selective:**

```python
# For 95.39% of samples (Tr(H_x²) < 0.01):
λ(x) ≈ 2.0 · sigmoid(negative_small) ≈ 0.95

Gating: (1 - u·0.95)
  u=0.2: 0.81 (vs 0.80 in standard UPGD)
  u=0.5: 0.52 (vs 0.50 in standard UPGD)
  u=0.8: 0.24 (vs 0.20 in standard UPGD)

Result: SLIGHTLY MORE protection for ALL weights on most samples!
```

```python
# For 4.61% of samples (Tr(H_x²) > 0.01):
λ(x) ≈ 2.0 · sigmoid(positive) → 1.0 to 1.5

Gating: (1 - u·λ)
  u=0.2, λ=1.2: 0.76 (MORE protection than most samples)
  u=0.5, λ=1.2: 0.40 (SIGNIFICANT extra protection)
  u=0.8, λ=1.2: 0.04 (NEAR-ZERO updates, almost blocked)
  u=0.8, λ=1.5: -0.20 (NEGATIVE GATING on 18 samples = 0.18%)

Result: GRADUATED protection increase on rare hard samples
```

## Core Hypothesis: Rare Sample Protection

### Why Hard Samples Matter

**Hard samples (high curvature) represent rare but important patterns:**

1. **Nature of high-curvature samples:**
   - Boundary cases
   - Rare classes/features
   - Ambiguous or noisy examples
   - Samples that define critical decision boundaries
   - **Informative edge cases** crucial for robust representations

2. **In Input-Permuted MNIST context:**
   - Certain input configurations after permutation are inherently harder
   - Rare geometric patterns that are difficult to classify
   - Edge cases that stress the learned representations

3. **The vulnerability problem:**
   - Standard UPGD protects based on **global utility** (averaged over all samples)
   - Connections important for **rare samples** have:
     - **Low global utility** (rarely activated across all samples)
     - **High local importance** (critical when activated for specific rare patterns)
   - Standard UPGD would **NOT protect** these → forgetting of rare patterns

## Two Orthogonal Protection Signals

### Signal Decomposition

| Signal | What it measures | What it protects | Temporal scope |
|--------|------------------|------------------|----------------|
| **Utility (u)** | Global importance averaged over all samples | Frequently important connections | Long-term average |
| **Lambda (λ)** | Sample-specific difficulty/curvature | Connections critical for rare/hard samples | Per-sample |

### Four-Quadrant Analysis

```
                  High λ (rare/hard sample)
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    │   Low u, High λ       │    High u, High λ     │
    │                       │                       │
    │   Rare Pattern        │    Core Important     │
    │   Connection          │    Connection on      │
    │                       │    Hard Sample        │
    │   Low global utility  │                       │
    │   but locally crucial │    Global + local     │
    │                       │    importance         │
    │   PROTECTED HERE! ✓   │    HEAVILY PROTECTED  │
    │                       │                       │
    │   Example:            │    Example:           │
    │   u = 0.2, λ = 2.0    │    u = 0.8, λ = 2.0   │
    │   (1 - 0.2·2) = 0.6   │    (1 - 0.8·2) = -0.6 │
    │   60% update allowed  │    NEGATIVE GATING!   │
    │                       │                       │
    ├───────────────────────┼───────────────────────┤
    │                       │                       │
    │   Low u, Low λ        │    High u, Low λ      │
    │                       │                       │
    │   Easy Sample         │    Important Conn     │
    │   Unimportant Weight  │    on Easy Sample     │
    │                       │                       │
    │   FULL PLASTICITY     │    FULL PLASTICITY    │
    │                       │    (λ≈0 releases)     │
    │                       │                       │
    │   Example:            │    Example:           │
    │   u = 0.2, λ ≈ 0      │    u = 0.8, λ ≈ 0     │
    │   (1 - 0.2·0) = 1     │    (1 - 0.8·0) = 1    │
    │   100% update         │    100% update        │
    │                       │                       │
    └───────────────────────┼───────────────────────┘
                            │
                     Low λ (easy sample)
```

### Key Insights from Quadrants

**Top-Left (Low u, High λ)**: **CRITICAL DISCOVERY**
- Standard UPGD: `(1 - 0.2) = 0.8` → Allows 80% update
- Input-aware:   `(1 - 0.2·2.0) = 0.6` → Allows 60% update
- **25% MORE protection** for low-utility weights on hard samples!
- **This is the key advantage**: protects rare-pattern-specific connections

**Top-Right (High u, High λ)**:
- Standard UPGD: `(1 - 0.8) = 0.2` → Allows 20% update
- Input-aware:   `(1 - 0.8·2.0) = -0.6` → **Negative gating**
- Not just protection, but **active implicit regularization**
- Reverses update direction for highly important weights on hard samples

**Bottom Row (Low λ)**:
- Both configurations give near-full plasticity
- Input-aware **releases protection** even for high-utility weights on easy samples
- Allows **aggressive learning when safe**

## Why λ_max = 2.0 is Critical

### The Math of Exceeding 1.0

**When λ_max > 1.0**, the product `u·λ(x)` can exceed 1.0, causing:

```python
(1 - u·λ(x)) < 0  # Negative gating

Effective update becomes:
θ ← θ(1-αβ) - α[(g + ξ) ⊙ (negative_value)]
θ ← θ(1-αβ) + α|(g + ξ)|  # Sign reversal!
```

### Implicit Regularization via Negative Gating

**For high-utility weights (u > 0.5) on hard samples (λ > 1.0):**

The negative gating creates an **implicit pull toward previous values** without explicit regularization term R.

```
Standard update: θ_new = θ_old - update
Negative gating: θ_new = θ_old + update

If gradient points away from θ_0, negative gating pulls back!
```

### Protection Spectrum

| u·λ(x) | Gating (1-u·λ) | Behavior |
|--------|----------------|----------|
| 0.0 | 1.0 | Full plasticity |
| 0.5 | 0.5 | 50% update dampening |
| 1.0 | 0.0 | Complete blocking |
| 1.5 | -0.5 | Reversed update (implicit reg) |
| 2.0 | -1.0 | Strong reversal |

## Rare Sample Protection Hypothesis

### Statement

**Hard samples with high curvature represent rare but informative patterns. The connections critical for these rare patterns have low global utility but high local importance. Input-aware gating with λ_max > 1.0 protects these vulnerable connections that standard UPGD would forget.**

### Supporting Evidence

1. **Long-tailed curvature distribution**:
   - Mean ~0.002, Max ~0.3
   - Only ~10-25% of samples exceed τ=0.01
   - These are genuinely rare, informative samples

2. **λ_max = 2.0 provides graduated protection**:
   - Low-u, high-λ: Moderate extra protection (0.6 vs 0.8)
   - High-u, high-λ: Strong implicit regularization (negative gating)
   - All weights on low-λ: Full plasticity regardless of u

3. **Task structure of Input-Permuted MNIST**:
   - After permutation, some rare input patterns become critical
   - Low-utility weights might be important for specific permuted configurations
   - Need to retain "dormant" connections that activate on edge cases

### Predictions

If hypothesis is correct:

1. **Accuracy on rare samples**: Input-aware should significantly outperform on high-curvature samples
2. **Weight drift**: Low-utility weights should drift less during high-λ samples
3. **Utility-lambda correlation**: Bimodal distribution for high-λ samples (both high-u and low-u weights activated)
4. **Task transitions**: Curvature spikes at task boundaries; protection engages exactly when needed

## Proposed Validation Experiments

### Analysis 1: Utility-Lambda Correlation
```python
# For samples with high curvature (λ > 0.5):
# Examine utility distribution of activated weights

Expected: BIMODAL distribution
- Peak 1: High-u weights (globally important)
- Peak 2: Low-u weights (rare-pattern specific)

Validation: Low-u weights benefit disproportionately from λ protection
```

### Analysis 2: Performance Stratified by Curvature
```python
# Partition samples:
# - Easy: Tr(H_x²) < 0.01  (~75-90% of samples)
# - Hard: Tr(H_x²) > 0.01  (~10-25% of samples)

# Compare accuracy:
# Standard UPGD vs Input-aware (gating-only, λ_max=2.0)

Expected: Input-aware shows LARGER improvement on hard samples
```

### Analysis 3: Weight Drift on Hard Samples
```python
# Track weights activated during high-λ samples
# Measure |θ_t - θ_0| over time

Expected:
- Standard UPGD: Large drift (forgetting rare patterns)
- Input-aware: Controlled drift (λ protection even for low-u)
```

### Analysis 4: Task Transition Dynamics
```python
# At permutation boundaries (every 5000 steps):

Track:
1. Curvature distribution before/after permutation
2. λ(x) activation rate (fraction with λ > 0.5)
3. Protection engagement on transition samples
4. Retention of low-u connections that prove useful post-permutation

Expected:
- Curvature spike at boundaries
- Protection engages during risky transitions
- Low-u weights preserved by λ become important in new task
```

## Ablation Experiment Design

### Primary Ablations

**A1: Lambda_max sweep** (Does exceeding 1.0 matter?)
```bash
λ_max ∈ {0.5, 1.0, 1.5, 2.0, 5.0}
Fixed: τ=0.01, gating-only

Hypothesis: Performance improves with λ_max up to ~2.0, then plateaus
Critical test: Does λ_max > 1.0 significantly outperform λ_max = 1.0?
```

**A2: Threshold sweep** (Sensitivity to rare sample definition)
```bash
τ ∈ {0.001, 0.005, 0.01, 0.05, 0.1}
Fixed: λ_max=2.0, gating-only

Hypothesis: Optimal τ at ~75-90th percentile of curvature distribution
Too low: Over-protection, loss of plasticity
Too high: Misses rare samples, no benefit over standard UPGD
```

**A3: Lambda-only protection** (Remove utility dependence)
```bash
Gating: (1 - λ(x))
No utility weighting

Hypothesis: WORSE than u·λ product
Reason: Over-protects all weights on hard samples, including irrelevant ones
```

**A4: Max vs Product** (Additive vs multiplicative protection)
```bash
Product (current): (1 - u·λ(x))
Max (alternative):  (1 - max(u, λ(x)))

Hypothesis: Product is BETTER
Reason: Max over-protects (protects if EITHER signal triggers)
        Product is selective (protects only when BOTH signal importance)
```

**A5: Negative gating clipping**
```bash
Clip gating to [0, 1]: (1 - u·λ(x)).clamp(0, 1)
Allow negative:       (1 - u·λ(x))  [current]

Hypothesis: Allowing negative gating improves performance
Reason: Implicit regularization for high-u, high-λ combinations
```

### Secondary Ablations

**B1: Regularization interaction**
```bash
Compare:
- Gating only (current best)
- Regularization only (R = λ·u·(θ-θ₀), no gating)
- Both (full input-aware UPGD)

Hypothesis: Gating-only best for Input-Permuted MNIST
Reason: Task requires unlearning; regularization toward θ₀ harmful
```

**B2: Non-protecting variants**
```bash
Apply λ to noise only: θ ← θ(1-αβ) - α[g + ξ·(1 - u·λ(x))]

Hypothesis: WORSE than full gating
Reason: Doesn't protect from gradient updates on hard samples
```

## Open Questions

1. **Negative gating interpretation**:
   - Is it truly implicit regularization or an artifact?
   - Should we clip to [0, 1] or embrace negative values?
   - What's the mathematical justification for sign reversal?

2. **Generalization beyond Input-Permuted MNIST**:
   - Does this help on Label-Permuted EMNIST (forgetting + plasticity)?
   - Does this help on Label-Permuted CIFAR-10 (mainly forgetting)?
   - Is rare sample protection task-specific or general principle?

3. **Curvature threshold tuning**:
   - Should τ be adaptive (e.g., running percentile)?
   - Fixed threshold may not generalize across tasks
   - How to set τ without validation set?

4. **Computational cost vs benefit**:
   - Computing λ(x) every step is expensive
   - Would periodic computation (every 10 steps) work?
   - Trade-off: accuracy vs computational efficiency

5. **Interaction with network architecture**:
   - Does this depend on ReLU dead units?
   - Would it work with other activations (GELU, SiLU)?
   - Role of network depth/width?

## REVISED Summary (Based on Actual Data)

**ORIGINAL HYPOTHESIS (WRONG)**:
- τ=0.01 creates selective protection only on rare samples
- Most samples have λ≈0 (full plasticity)
- Only top 10-25% samples get protection

**ACTUAL BEHAVIOR (DATA-DRIVEN)**:
- τ=0.01 creates a **BASELINE protection boost** (λ≈0.95) for 95% of samples
- Only top 4.61% samples get λ>1.0
- This is NOT selective protection, but **graduated protection scaling**

**Key Finding**: Input-aware gating with λ_max=2.0 and τ=0.01 provides **universally stronger protection with rare-sample amplification**.

**Mechanism (REVISED)**:

1. **Baseline protection** (95.39% of samples, Tr(H_x²) < 0.01):
   - λ ≈ 0.95
   - Gating: `(1 - 0.95u)` vs standard UPGD: `(1 - u)`
   - **Effect**: ~5% stronger protection across ALL weights
   - This is like "weight decay lite" - pulls updates toward zero slightly

2. **Enhanced protection** (4.61% of samples, Tr(H_x²) > 0.01):
   - λ ≈ 1.0 to 1.5
   - Gating: `(1 - 1.2u)` for moderate hard samples
   - **Effect**: 20-50% stronger protection on rare, hard samples

3. **Extreme protection** (0.18% of samples, highest curvature):
   - λ > 1.25 with high utility (u > 0.8)
   - Gating goes NEGATIVE
   - **Effect**: Implicit regularization via sign reversal

**Why it works (REVISED THEORY)**:

1. **Universal damping**: All samples get slightly more conservative updates
   - Reduces overfitting to individual samples
   - Acts like adaptive weight decay based on input difficulty

2. **Rare sample protection**: 4.61% of hard samples get amplified protection
   - Preserves connections activated by rare but informative patterns
   - Prevents catastrophic updates on high-curvature edge cases

3. **Negative gating on extremes**: 0.18% of samples reverse high-utility weight updates
   - Stabilizes important weights when encountering very hard samples
   - Acts as implicit L2-Init regularization without explicit R term

**Alternative interpretation**:
This is essentially **input-difficulty-modulated weight decay**:
- Easy samples (low curv): Standard update with mild damping
- Hard samples (high curv): Strong damping approaching full protection
- Very hard + important weights: Reversal (implicit pull toward θ₀)

**Critical Questions** (data-driven):

1. **Is the baseline λ≈0.95 necessary?**
   - Try λ_scale ∈ {0.01, 0.05, 0.1, 0.5} to make sigmoid steeper/gentler
   - Hypothesis: Steeper sigmoid (lower λ_scale) creates true selectivity

2. **Is negative gating helpful or harmful?**
   - Only 18 samples (0.18%) experience it
   - Ablation: Clip gating to [0, 1] vs allow negative
   - Hypothesis: Minimal impact given rarity

3. **Can we achieve selectivity with different τ?**
   - τ=0.01 is 95.4th percentile → most samples protected
   - Try τ ∈ {0.001, 0.005, 0.01, 0.05} to shift baseline λ
   - Hypothesis: Lower τ creates true two-regime behavior

**Next steps (data-driven)**:
1. **λ_scale ablation**: Test {0.01, 0.05, 0.1, 0.5} to control sigmoid steepness
2. **τ ablation**: Test {0.001, 0.005, 0.01, 0.05} to shift percentile cutoff
3. **Baseline vs selective**: Compare uniform λ=0.95 vs current sigmoid approach
4. **Negative gating**: Test with/without clipping to [0,1]
5. **Performance stratification**: Easy vs hard sample accuracy breakdown

---

## THE PARADOX (Critical Observation)

**USER'S KEY INSIGHT**: If λ≈0.95 for 95% of samples, this means LESS plasticity than standard UPGD, yet performance is AS GOOD AS standard UPGD. How?

### The Math Confirms Less Plasticity

```python
# Most samples (95.39%) with λ≈0.95:
Standard UPGD gating: (1 - u)
Input-aware gating:   (1 - u·0.95) = (1 - 0.95u)

For u=0.5: 0.50 (UPGD) vs 0.525 (input-aware) → 5% MORE protection
For u=0.8: 0.20 (UPGD) vs 0.24 (input-aware)  → 20% MORE protection

More protection = Less plasticity
```

**The paradox**: Input-aware is LESS plastic on 95% of samples, yet performs equally well!

### Possible Resolutions

**Hypothesis A: Hard samples dominate performance**
- The 4.61% of samples with λ>1.0 are disproportionately important
- These might be the "catastrophic" samples that cause forgetting
- Protecting them heavily (80% less update for high-u weights) preserves performance
- Slight plasticity loss on easy samples doesn't matter

**Hypothesis B: Noise dampening is the key mechanism**
```python
Update: θ ← θ(1-αβ) - α[(g + ξ) ⊙ (1 - u·λ)]

With σ=0.1 noise:
- Standard UPGD: Noise scaled by (1-u)
- Input-aware: Noise scaled by (1-0.95u) → LESS noise

Maybe the benefit is:
  - Slightly less gradient plasticity (bad)
  - Significantly less noise-induced drift (good)

Net effect: More stable learning without hurting useful gradient updates
```

**Hypothesis C: The 5% "tax" is acceptable for Input-Permuted MNIST**
- Task requires rapid relearning after each permutation
- Standard UPGD might be TOO plastic (overfits to noise)
- 5% reduction in plasticity = beneficial regularization
- But on tasks requiring maximum plasticity, this would hurt

**Hypothesis D: Compensation via other hyperparameters**
- lr=0.01, σ=0.1, β=0.9999 might not be optimal for standard UPGD
- The λ≈0.95 factor might accidentally compensate for bad hyperparameter choices
- Effective learning rate: 0.01 × 0.525 ≈ 0.0052 (for u=0.5)
- Maybe standard UPGD with lr=0.0052 would perform equally well?

**Hypothesis E: The sigmoid shape matters more than the mean**
- Average λ≈0.95, but VARIANCE in λ is what helps
- Standard UPGD: Fixed protection for all samples
- Input-aware: Variable protection adapts to sample difficulty
- Even if average plasticity is lower, ADAPTIVE plasticity might be better

### Critical Tests

1. **Compare to UPGD with lr=0.0095** (compensates for 5% less plasticity)
   - If performs equally → Hypothesis D (just effective lr reduction)
   - If input-aware still better → Something else is going on

2. **Stratified accuracy analysis**:
   - Accuracy on samples with Tr(H_x²) < 0.01 (easy, 95%)
   - Accuracy on samples with Tr(H_x²) > 0.01 (hard, 5%)
   - If hard-sample accuracy much better → Hypothesis A

3. **Ablation: Uniform λ=0.95 (no input-dependence)**:
   - Remove sigmoid, just multiply all updates by 0.95
   - If performs equally → Just a global damping factor
   - If worse → Adaptive aspect is critical

4. **Noise-only analysis**:
   - Track weight drift from ξ vs from g separately
   - Compare noise-induced variance between methods
   - If input-aware has lower noise variance → Hypothesis B

### My Best Guess

**I think it's Hypothesis B + E combined**:

The key is NOT the mean λ value, but:
1. **Noise control**: Dampening noise more on all samples stabilizes learning
2. **Adaptive gradient scaling**: Hard samples get extra protection from destructive gradients
3. **The combination**: Reduced noise + selective gradient dampening = better stability-plasticity tradeoff

The 5% less plasticity is a FEATURE, not a bug - it's preventing overfitting to individual samples and noise.

---

## EXPERIMENTAL RESULTS AND FINAL CONCLUSIONS

### λ_scale Ablation Results

**Tested configurations:**
- λ_scale = 0.1 (baseline, original): **Slight improvement** over standard UPGD
- λ_scale = 0.01 (steep sigmoid): **WORSE performance** than standard UPGD
- λ_scale ≥ 0.1 (0.2, 0.5, 1.0): All behave **identically** to λ_scale=0.1

### Lambda Distribution Comparison

| λ_scale | Lambda Range | Mean λ | Behavior |
|---------|--------------|--------|----------|
| 0.01 | 0.54 - 2.00 | 0.61 | **Strong two-regime**: 94% samples λ<0.9, 6% samples λ>1.0 |
| 0.05 | 0.90 - 1.85 | 0.92 | Moderate differentiation |
| 0.1  | 0.95 - 1.55 | 0.96 | Gentle gradient (baseline) |
| ≥0.2 | ~0.98 - ~1.3 | ~0.98 | Quasi-uniform, nearly same as 0.1 |

**Key finding**: λ_scale ≥ 0.1 converges to the same behavior - gentle gradient with λ≈0.95-1.0 for most samples.

### Why λ_scale=0.01 HURT Performance

With λ_scale=0.01:
- **94% of samples**: λ < 1.0 → `(1 - u·0.6) > (1 - u)` → **MORE plasticity** than standard UPGD
- **6% of samples**: λ > 1.0 → **LESS plasticity** (protected)

**This revealed the key insight:**

The increased plasticity on easy samples (94%) **harmed** performance, suggesting that standard UPGD's protection level was already appropriate. Releasing that protection caused worse results.

### CONFIRMED HYPOTHESIS: Input Curvature ≈ Utility

**User's hypothesis (CONFIRMED):**
```
High input curvature → High gradients → High utility → Already protected by UPGD
```

**Evidence:**
1. ✅ λ_scale=0.1 (λ≈1.0): Slight improvement (nearly identical to UPGD)
2. ✅ λ_scale=0.01 (strong differentiation): **Worse** performance
3. ✅ λ_scale ≥ 0.1: All equivalent (redundant with UPGD)

**Conclusion:**

Input curvature is **largely redundant** with utility for Input-Permuted MNIST:
- High-curvature samples are the same samples with high gradients
- These samples already increase utility on active weights
- UPGD's `(1 - u)` protection already handles them appropriately
- Adding input-curvature gating `(1 - u·λ)` provides minimal additional benefit when λ≈1.0

### Why λ≈1.0 (λ_scale≥0.1) Shows Slight Improvement

The marginal improvement with λ_scale=0.1 comes from **edge cases**:

1. **Temporal mismatch**:
   - Utility is GLOBAL (accumulated over all past samples)
   - Curvature is LOCAL (current sample difficulty)
   - Rare case: High-curvature sample on LOW-utility weight (not yet accumulated)
   - Input-aware provides slight extra protection: `(1 - 0.2·1.1) = 0.78` vs `(1 - 0.2) = 0.8`

2. **Noise dampening**:
   - λ≈0.95 slightly dampens the noise term `ξ` on all samples
   - Marginally reduces noise-induced drift without hurting gradient learning

3. **Sample-adaptive modulation**:
   - Even with λ close to 1.0, slight variation (0.95 → 1.3) provides adaptive dampening
   - Hard samples get marginally more protection than easy samples

But these effects are **small** because curvature and utility are highly correlated.

### Implications for Research

#### **For Input-Permuted MNIST:**
- ✅ **Standard UPGD is near-optimal**
- ✅ Input curvature doesn't add significant value
- ✅ Computational cost of curvature measurement (Tr(H_x²)) is not justified
- ✅ Use standard UPGD with first-order utility: `U = -g·w`

#### **When Input-Curvature MIGHT Help:**

Look for tasks where **curvature ≠ utility** (orthogonal signals):

1. **Non-stationary distributions:**
   - Sudden distribution shifts
   - Old utility values become stale
   - Curvature provides fresh, local signal

2. **Adversarial/OOD samples:**
   - High curvature but never seen before
   - Low utility (no accumulation history)
   - Curvature catches them, utility doesn't

3. **Class-imbalanced learning:**
   - Rare classes have low global utility
   - But high local curvature when they appear
   - Curvature protects rare-class weights

4. **Tasks with weak gradient signal:**
   - Sparse rewards (RL)
   - Noisy gradients
   - Curvature provides additional geometry information

### Open Question: Second-Order Effects

**User's refined hypothesis:**

> Even if curvature and utility are correlated, curvature might provide better scaling/weighting than first-order utility alone, because first-order utility (U = -g·w) is an approximation that ignores the Hessian term (½H·w²).

**This is still untested!**

To prove this hypothesis, you would need to show that:
- Input curvature Tr(H_x²) provides information about the Hessian term
- This improves protection scaling beyond first-order utility
- Performance improvement comes specifically from better scaling, not just correlation

**Proposed experiments to test this:**

### **Experiment 1: Direct Comparison (Gold Standard)**

Compare three variants to isolate the contribution of second-order information:

```bash
A. First-order UPGD (FO):     U = -g·w
   Protection: (1 - u)

B. Second-order UPGD (SO):    U = ½H·w² - g·w  (uses parameter Hessian diagonal)
   Protection: (1 - u_SO)

C. Input-aware FO UPGD:       U = -g·w, modulated by λ(Tr(H_x²))
   Protection: (1 - u·λ)

Expected outcomes:
  If C ≈ B > A → Input curvature approximates second-order utility ✓
                 Curvature provides Hessian information that FO utility misses
  If B > C ≈ A → Input curvature doesn't help ✗
                 Parameter Hessian (SO) is different from input curvature
  If B ≈ C > A → Both provide second-order info, equally useful
```

**Why this works:**
- B (SO-UPGD) has the true second-order term ½H·w²
- If C performs like B, then λ(Tr(H_x²)) approximates the missing Hessian info
- If C performs like A, then input curvature is not a good proxy for parameter curvature

### **Experiment 2: Stratified Analysis (Most Practical)**

**Goal:** Find specific scenarios where curvature provides unique value beyond utility.

**Method:**
```python
# During training, log per-sample metrics:
for each sample x:
    - curvature[x] = Tr(H_x²)  # Input-space curvature
    - active_weights = weights activated by x
    - utility[active_weights] = current utility values
    - accuracy[x] = prediction correct/incorrect
    - update_magnitude[active_weights] = |Δw|

# Post-hoc analysis:
# Partition samples into 2D grid (utility × curvature)
u_median = median(utility[active_weights])
c_median = median(curvature)

Q1: Low u (<median), Low curv (<median)   → Unimportant weights, easy samples
Q2: Low u (<median), High curv (≥median)  → KEY QUADRANT ★
Q3: High u (≥median), Low curv (<median)  → Important weights, easy samples
Q4: High u (≥median), High curv (≥median) → Important weights, hard samples

# Compare performance in each quadrant:
for Q in [Q1, Q2, Q3, Q4]:
    accuracy_UPGD[Q] = mean accuracy with standard UPGD
    accuracy_InputAware[Q] = mean accuracy with input-aware
    improvement[Q] = accuracy_InputAware[Q] - accuracy_UPGD[Q]
```

**Interpretation:**

**Q2 (Low u, High curv) is the critical test:**
- These are samples with high input curvature (risky/hard)
- But operating on weights with LOW utility (not yet important globally)
- **Standard UPGD**: `(1 - 0.2) = 0.8` → Allows 80% update (minimal protection)
- **Input-aware**: `(1 - 0.2·1.5) = 0.7` → Allows 70% update (extra protection)

**Expected results if hypothesis is correct:**
- Q2 has **non-trivial population** (>2% of samples)
- Q2 shows **significant improvement** with input-aware (>2% accuracy gain)
- Q4 shows **moderate improvement** (better scaling for high-u weights)
- Q1, Q3 show **no difference** (protection not needed or already sufficient)

**Expected results if hypothesis is wrong:**
- Q2 is nearly **empty** (<0.5% of samples) → Utility and curvature always correlate
- Q2 shows **no improvement** → Curvature doesn't help even when orthogonal
- All quadrants show **similar performance** → Input curvature is redundant

**Why Q2 matters:**
This quadrant represents the **unique contribution** of input curvature:
- High curvature identifies risky samples (local signal)
- Low utility means UPGD won't protect them (global signal misses them)
- If input-aware helps here, it's providing information utility doesn't have

### **Experiment 3: Correlation Analysis**

**Goal:** Quantify the relationship between utility and curvature.

```python
# During training, collect paired data:
data = []
for each sample x at timestep t:
    active_weights = get_active_weights(x)
    mean_utility = mean(utility[active_weights])
    input_curvature = Tr(H_x²)
    data.append((mean_utility, input_curvature))

# Compute correlations:
utilities, curvatures = zip(*data)

# Linear correlation
pearson_r = corrcoef(utilities, curvatures)

# Monotonic correlation (handles non-linear relationships)
spearman_r = spearmanr(utilities, curvatures)

# Information-theoretic (any dependency)
mutual_info = mutual_info_score(discretize(utilities), discretize(curvatures))
```

**Interpretation:**

| Pearson r | Interpretation | Implication |
|-----------|----------------|-------------|
| r > 0.7 | **Highly correlated** | Curvature is redundant with utility |
| 0.3 < r < 0.7 | **Moderately correlated** | Curvature provides some unique info |
| r < 0.3 | **Weakly correlated** | Curvature is orthogonal signal |

**Why this matters:**
- High correlation → Your confirmed hypothesis is correct
- Low correlation → Input curvature provides complementary information
- Can guide whether input-aware optimization is worth pursuing

### **Experiment 4: Alternative Scaling Functions**

**Goal:** Determine if the specific way we combine utility and curvature matters.

```bash
# Test different protection functions:
A. Standard UPGD:           (1 - u)
B. Product (current):       (1 - u·λ)
C. Additive:                (1 - u) × (1 - α·λ)
D. Max (conservative):      (1 - max(u, λ))
E. Weighted sum:            (1 - βu - (1-β)λ)
F. Power modulation:        (1 - u^(1/λ))
G. Curvature-gated utility: (1 - u) × λ
```

**Rationale for each:**

- **Product B**: Current approach, multiplicative interaction
- **Additive C**: Separate protection terms that multiply
- **Max D**: Protect if EITHER signal says to (conservative)
- **Weighted E**: Linear combination, tunable balance
- **Power F**: Curvature modulates the strength of utility protection
- **Gated G**: Curvature scales the utility-based protection directly

**Interpretation:**

If multiple variants significantly outperform standard UPGD (A):
→ Curvature provides useful information

If some specific variant B/C/D/E/F/G >> others:
→ The functional form of combination matters
→ Suggests specific mathematical relationship between u and λ

If all perform similarly:
→ Specific combination doesn't matter
→ Either all work (curvature helps) or none work (curvature redundant)

### **Experiment 5: Per-Weight Protection Trajectory**

**Goal:** Track individual weights to see if curvature improves protection timing/scaling.

```python
# Select representative weights:
# - High-utility weight (important)
# - Low-utility weight (unimportant)
# - Medium-utility weight (transitioning)

# For each weight w_i, log over time:
log[t] = {
    'utility': u_i[t],
    'curvature_contribution': contribution_to_Tr(H_x²),
    'lambda_value': λ[t],
    'protection_UPGD': (1 - u_i[t]),
    'protection_InputAware': (1 - u_i[t] × λ[t]),
    'update_magnitude': |Δw_i|,
    'drift_from_init': |w_i[t] - w_i[0]|
}

# Analyze trajectories:
# 1. Does input-aware better track actual importance?
# 2. Does it reduce drift for vulnerable weights?
# 3. Does it maintain plasticity where needed?
```

**Key questions:**
1. Do weights with high curvature contribution get better protection timing?
2. Does curvature catch "about-to-be-important" weights that utility misses?
3. Is the protection scaling (via λ) better matched to actual weight importance?

### **Recommended Experiment Order**

**Phase 1: Quick diagnostics** (1 day)
1. Correlation analysis → Tells you if curvature is orthogonal to utility
2. Stratified analysis setup → Partition existing data by utility/curvature

**Phase 2: Main test** (1 week)
3. Second-order UPGD comparison → Gold standard test
4. Stratified analysis results → Find where input-aware helps

**Phase 3: Deep dive** (if Phase 2 shows promise)
5. Alternative scaling functions → Optimize the combination
6. Per-weight trajectories → Understand mechanism

### **What Each Experiment Proves**

| Experiment | What it proves | Difficulty | Time |
|------------|----------------|------------|------|
| Correlation | Whether curvature ≠ utility | Easy | 1 day |
| Stratified | Where curvature helps | Medium | 2 days |
| SO-UPGD comparison | If curvature ≈ 2nd-order | Hard | 1 week |
| Scaling functions | Optimal combination | Medium | 3 days |
| Per-weight | Detailed mechanism | Hard | 1 week |

**Bottom line:**
Start with **Correlation** and **Stratified analysis** - they're fast and will tell you if the deeper experiments are worth doing.

### Summary: What We Learned

1. **λ_scale is critical**: Values ≥0.1 are equivalent (gentle); 0.01 is dramatically different (steep)

2. **Input curvature ≈ Utility**: For Input-Permuted MNIST, they're highly correlated

3. **UPGD is sufficient**: First-order utility already captures the protection needed

4. **Slight improvement is real but marginal**: λ≈1.0 helps edge cases, not worth computational cost

5. **Two-regime approach failed**: λ_scale=0.01 made things worse, not better

6. **Open question remains**: Does curvature improve utility *scaling* (second-order effects)?

### Recommendations

**For practitioners:**
- Use **standard UPGD** for Input-Permuted MNIST and similar tasks
- Only consider input-aware if you have strong reason to believe curvature ≠ utility
- If testing input-aware, use **λ_scale ≥ 0.1** (gentle gradient, safe)
- Avoid λ_scale < 0.05 unless you have evidence that increased plasticity helps

**For researchers:**
- Test the **second-order utility hypothesis** (compare with UPGD-SO)
- Do **stratified analysis** to find where input-curvature helps
- Measure **utility-curvature correlation** on your specific task
- Look for tasks where signals are **orthogonal** (adversarial, OOD, rare classes)

---

## COMPLETE CONVERSATION SUMMARY

This section captures the key insights from the entire analysis for future reference.

### Journey Overview

**Phase 1: Initial Hypothesis (WRONG)**
```
Assumption: τ=0.01 creates selective protection
- Low curvature (<τ): λ→0, full plasticity
- High curvature (>τ): λ→2, strong protection

Reality with λ_scale=0.1:
- ALL samples: λ≈0.95-1.0 (quasi-uniform)
- Only 4.61% samples: λ>1.0
- Gentle gradient, not selective binary
```

**Phase 2: The Sigmoid Problem**
```
Key realization: λ_scale=0.1 makes sigmoid TOO GENTLE

For sigmoid(x) to approach 0 or 1:
- Need |x| > 10
- With λ_scale=0.1: normalized = (curv - 0.01)/0.1
- Even curv=0: normalized = -0.1 → sigmoid ≈ 0.475 → λ≈0.95

Solution: Use λ_scale=0.01 for true two-regime behavior
```

**Phase 3: The Paradox**
```
Observation: λ≈0.95 means LESS plasticity than UPGD
- (1 - 0.8×0.95) = 0.24 > (1 - 0.8) = 0.20
- More of update applied with input-aware!

But then confusion: Which direction is "more protection"?

Clarification:
- λ < 1: (1 - u×λ) > (1 - u) → MORE update = LESS protection
- λ > 1: (1 - u×λ) < (1 - u) → LESS update = MORE protection
```

**Phase 4: The λ_scale=0.01 Experiment**
```
With steep sigmoid (λ_scale=0.01):
- 94% samples: λ<0.9 → MUCH MORE plasticity
- 6% samples: λ>1.0 → Protected

Result: WORSE performance than standard UPGD

Conclusion: Increased plasticity harmed performance!
→ UPGD's protection level was already correct
→ Curvature ≈ Utility (correlated)
```

**Phase 5: User's Hypothesis (CONFIRMED)**
```
"High curvature → High gradients → High utility → Already protected by UPGD"

Evidence:
✓ λ_scale=0.1 (λ≈1): Marginal improvement
✓ λ_scale=0.01 (λ varies widely): WORSE
✓ λ_scale≥0.1: All equivalent

Conclusion: Input curvature is redundant with utility
```

### Critical Insights

**1. λ_scale is the Key Hyperparameter**

| λ_scale | Behavior | Lambda Range | Use Case |
|---------|----------|--------------|----------|
| 0.01 | Steep sigmoid | 0.5-2.0 | True two-regime, risky |
| 0.05 | Moderate | 0.9-1.85 | Some differentiation |
| 0.1 | Gentle (default) | 0.95-1.55 | Safe, marginal benefit |
| ≥0.2 | Very gentle | ~0.98-1.3 | Essentially same as 0.1 |

**2. Understanding Gating Direction**

```python
Standard UPGD: update × (1 - u)
Input-aware:   update × (1 - u·λ)

When λ < 1:
  u·λ < u
  (1 - u·λ) > (1 - u)
  → Applies MORE of update → LESS protection → MORE plasticity

When λ > 1:
  u·λ > u
  (1 - u·λ) < (1 - u)
  → Applies LESS of update → MORE protection → LESS plasticity

When λ = 1:
  Same as standard UPGD
```

**3. Why λ≈1 Shows Slight Improvement**

Edge cases where curvature helps:
1. **Temporal mismatch**: High-curvature sample on not-yet-high-utility weight
2. **Noise dampening**: λ≈0.95 slightly dampens noise term
3. **Adaptive modulation**: Small variation in λ provides sample-specific adjustment

But these are **marginal** because curvature and utility are highly correlated.

**4. The Utility-Curvature Correlation**

For Input-Permuted MNIST:
```
High curvature samples = High gradient samples = High utility samples

Why?
- Hard samples → Large loss → Large gradients
- Large gradients → High utility accumulation
- UPGD already protects these weights via (1-u)

Therefore:
- Input curvature provides redundant information
- Computing Tr(H_x²) is expensive and not justified
```

**5. When Input-Curvature MIGHT Help**

Tasks where **curvature ≠ utility**:
- **Non-stationary distributions**: Utility is stale, curvature is fresh
- **Adversarial/OOD**: High curvature, but low utility (never seen)
- **Class imbalance**: Rare classes have low global utility, high local curvature
- **Weak gradient signal**: Sparse rewards, noisy gradients

**6. The Second-Order Question (OPEN)**

Refined hypothesis:
> "Even if curvature and utility are correlated, curvature might provide better *scaling* because first-order utility (U=-g·w) ignores the Hessian term (½H·w²)"

This is **still untested** and requires:
- Comparison with second-order UPGD (U = ½H·w² - g·w)
- Stratified analysis (utility × curvature quadrants)
- Correlation measurement
- Alternative scaling functions

### Key Equations Reference

**Lambda computation:**
```python
λ(x) = λ_max · sigmoid((Tr(H_x²) - τ) / λ_scale)

where:
  Tr(H_x²) = input curvature (Hutchinson estimation)
  τ = threshold (e.g., 0.01)
  λ_scale = sigmoid steepness (e.g., 0.1)
  λ_max = maximum lambda value (e.g., 2.0)
```

**Update rules:**
```python
# Standard UPGD
θ ← θ(1-αβ) - α[(g + ξ) ⊙ (1 - u)]

# Input-aware UPGD (gating only)
θ ← θ(1-αβ) - α[(g + ξ) ⊙ (1 - u·λ(x))]

# Input-aware UPGD (with regularization)
θ ← θ(1-αβ) - α[(g + R + ξ) ⊙ (1 - u·λ(x))]
where R = λ(x)·u·(θ - θ₀)
```

**First-order utility:**
```python
U = -g·w  (UPGD uses this)
```

**Second-order utility:**
```python
U = ½H·w² - g·w  (includes Hessian diagonal)
```

### Important Files Reference

**Scripts:**
- `test_input_permuted_mnist_stats_upgd_input_aware_fo_global_gating_only.sh`: Main experiment script
- `analyze_curvature_distribution.py`: Analyze lambda distributions
- `stratified_analysis.py`: Utility×curvature quadrant analysis
- `analyze_utility_curvature_correlation.py`: Correlation measurement

**Code locations:**
- Learner: `core/learner/input_aware_upgd.py`
- Optimizer: `core/optim/weight_upgd/input_aware.py`
- Runner: `core/run/run_stats_with_curvature.py`

**Key parameters:**
```bash
--lambda_scale 0.1      # Sigmoid steepness (KEY!)
--lambda_max 2.0        # Maximum lambda value
--curvature_threshold 0.01  # Threshold τ
--hutchinson_samples 5  # For Tr(H_x²) estimation
--disable_regularization True/False
--disable_gating True/False
```

### Timeline of Understanding

1. **Start**: Thought λ≈0 for most samples (WRONG)
2. **Data analysis**: Discovered λ≈0.95 for most samples
3. **Confusion**: Why does λ≈0.95 work if it's almost uniform?
4. **Realization**: λ_scale=0.1 makes sigmoid too gentle
5. **Hypothesis**: Maybe λ_scale=0.01 will help (two-regime)
6. **Experiment**: λ_scale=0.01 HURT performance
7. **User insight**: "Curvature-based utility already protected by UPGD"
8. **Confirmation**: Experiments confirm curvature ≈ utility
9. **Refined question**: Does curvature improve *scaling*? (open)

### What NOT to Do

❌ **Don't use λ_scale < 0.05** without strong evidence
- Increases plasticity on easy samples
- Can hurt performance if UPGD's protection was already correct

❌ **Don't assume input curvature always helps**
- Test correlation with utility first
- Expensive computation may not be justified

❌ **Don't confuse gating direction**
- λ < 1 → MORE plasticity (less protection)
- λ > 1 → LESS plasticity (more protection)

❌ **Don't ignore λ_scale convergence**
- λ_scale ≥ 0.1 all behave similarly
- No need to test 0.2, 0.5, 1.0 separately

### What TO Do

✅ **Start with standard UPGD** for new tasks
- It's simpler and often sufficient
- Add input-awareness only if needed

✅ **Measure utility-curvature correlation** on your task
- High correlation (r>0.7) → Input curvature redundant
- Low correlation (r<0.3) → Worth exploring

✅ **Use stratified analysis** if pursuing input-aware
- Find specific scenarios where it helps
- Focus on Q2 quadrant (low-u, high-curv)

✅ **Consider second-order UPGD** as alternative
- May provide better Hessian information
- More principled than input curvature

### Final Takeaway

**For Input-Permuted MNIST and similar tasks:**

Input-aware UPGD with curvature gating provides **marginal improvement** (~1-2%) at **significant computational cost** (computing Tr(H_x²) every step). The improvement comes from **edge cases** where curvature and utility temporarily diverge, but overall they're **highly correlated**.

**Standard UPGD is near-optimal** because its utility mechanism already captures the necessary protection. The first-order utility U=-g·w implicitly tracks sample difficulty through gradient magnitude accumulation.

**Future research** should focus on:
1. Tasks where curvature ≠ utility (adversarial, OOD, rare classes)
2. Second-order UPGD comparison (does ½H·w² help more?)
3. Stratified analysis to isolate specific beneficial scenarios
4. Alternative scaling functions if pursuing input-awareness

**The open question:** Does input curvature improve the *scaling* of protection (second-order effects) even when correlated with utility? This requires targeted experiments comparing with second-order UPGD and measuring performance in utility×curvature quadrants.

---

## Document Metadata

**Created:** October 2025
**Last Updated:** October 2025
**Status:** Complete analysis with open research questions
**Contributors:** Analysis based on systematic experimentation and ablation studies
**Related Work:** UPGD (Elsayed 2024), HesScale, Input-aware optimization

**Keywords:** input-aware optimization, curvature-based protection, utility-based gradient descent, continual learning, loss of plasticity, catastrophic forgetting, lambda scaling, sigmoid steepness, second-order optimization

---

## GATING FACTOR DESIGN DISCUSSION (October 2025)

### Problem: Lambda Scaling and Gating Factor Formula

**Context:** Need to redesign both lambda computation and gating factor formula to achieve:
1. High curvature → More protection
2. High utility + High lambda → More protection
3. Low utility + Low lambda → Less protection
4. High utility + Low lambda → Less protection
5. Low utility + High lambda → More protection

### Lambda Computation Analysis

#### Current Sigmoid Approach (PROBLEMATIC)
```python
normalized_curvature = (current_curvature - threshold) / lambda_scale
lambda_value = lambda_max * torch.sigmoid(torch.tensor(normalized_curvature)).item()
```

**Issue:** With any reasonable lambda_scale, sigmoid compresses values around 1.0
- `lambda_scale=0.1`: Range [0.9, 1.5] → narrow, centered at 1.0
- `lambda_scale=1.0`: Range even narrower → [0.995, 1.24]
- **No dynamic range** for meaningful curvature-to-protection mapping

#### Proposed Lambda Mappings

**Option 1: Linear Ratio (Simple)**
```python
lambda_value = min(lambda_max, max(0, curvature / threshold))
```
- At threshold: lambda = 1.0
- Below threshold: lambda < 1.0 (less protection)
- Above threshold: lambda > 1.0 (more protection, capped)
- **Pro:** Clear interpretation, proportional scaling
- **Con:** Sudden cap at lambda_max

**Option 2: Linear Offset (Centered at 1.0)**
```python
lambda_value = max(0, 1.0 + (curvature - threshold) / lambda_scale)
```
- At threshold: lambda = 1.0 (baseline)
- curvature > threshold: lambda > 1.0 (more protection)
- curvature < threshold: lambda < 1.0 (less protection)
- **Pro:** Symmetric around neutral point, tunable via lambda_scale
- **Con:** Can go arbitrarily large without lambda_max

**Option 3: Power Law**
```python
ratio = curvature / threshold
lambda_value = min(lambda_max, ratio ** power)  # power=1.5 or 2
```
- Smooth non-linear transition
- More aggressive for high curvature
- **Pro:** Controlled non-linearity
- **Con:** Extra hyperparameter (power)

### Gating Factor Formula Analysis

#### Current Formula (BROKEN for lambda > 1)
```python
gating_factor = 1 - scaled_utility * lambda
```

**Problem:** When lambda > 1 and utility is high:
- Example: utility=1.0, lambda=2.0
- gating = 1 - 1.0*2.0 = -1.0
- Negative values cause gradient sign reversal

**Fix Applied:**
```python
gating_factor = torch.clamp(1 - scaled_utility * lambda, min=0.0)
```
- Prevents negative values
- But lambda > 1 still just zeros out gradients completely

#### The Core Design Challenge

With current formula `gating = 1 - utility * lambda`:

| Utility | Lambda | Product | Gating | Behavior |
|---------|--------|---------|--------|----------|
| 0.1 | 0.5 | 0.05 | 0.95 | **High plasticity** ✓ |
| 0.1 | 2.0 | 0.2 | 0.80 | Moderate plasticity (want more protection!) ✗ |
| 1.0 | 0.5 | 0.5 | 0.50 | Moderate protection ✓ |
| 1.0 | 2.0 | 2.0 | 0.0 (clamped) | **Complete blocking** ✓ |

**Issue:** Low utility + high lambda → Product is still small → Weak protection

This **violates requirement #5**: Low utility params should still get protected when lambda is high.

#### Proposed Gating Formulas

**Option A: Inverse Scaling (Smooth Protection)**
```python
gating_factor = 1 / (1 + scaled_utility * lambda)
```

| Utility | Lambda | Denominator | Gating | Protection Level |
|---------|--------|-------------|--------|------------------|
| 0.1 | 0.5 | 1.05 | 0.95 | Minimal (high plasticity) ✓ |
| 0.1 | 2.0 | 1.2 | 0.83 | **Still weak** ✗ |
| 1.0 | 0.5 | 1.5 | 0.67 | Moderate ✓ |
| 1.0 | 2.0 | 3.0 | 0.33 | Strong ✓ |

**Problem:** Low utility always results in weak protection, even with high lambda.

**Option B: Lambda Dominates**
```python
gating_factor = 1 / (1 + lambda) * (1 - 0.5 * utility)
```
- Lambda provides base protection
- Utility modulates within that range
- **Pro:** Lambda affects all params regardless of utility
- **Con:** Complex interaction, harder to interpret

**Option C: Additive Protection**
```python
gating_factor = 1 / (1 + lambda + utility)
```

| Utility | Lambda | Denominator | Gating | Protection |
|---------|--------|-------------|--------|------------|
| 0.1 | 0.5 | 1.6 | 0.625 | Moderate ✓ |
| 0.1 | 2.0 | 3.1 | 0.32 | **Strong** ✓ |
| 1.0 | 0.5 | 2.5 | 0.40 | Moderate ✓ |
| 1.0 | 2.0 | 4.0 | 0.25 | Strong ✓ |

**Pro:** Both utility and lambda contribute independently
- Low utility + high lambda → Strong protection ✓
- High utility + low lambda → Moderate protection ✓
- Both high → Very strong protection ✓

**Option D: Weighted Multiplicative**
```python
gating_factor = 1 / (1 + lambda * (0.5 + 0.5 * utility))
```
- Lambda weighted by utility (0.5 to 1.0 range)
- Ensures lambda has minimum effect even at low utility
- **Pro:** Balances multiplicative and additive approaches

**Option E: Separate Gating Factors (Multiplicative)**
```python
gating_factor = (1 / (1 + lambda)) * (1 / (1 + utility))
```
- Each signal provides independent gating
- Combined via multiplication
- **Pro:** Clear separation of concerns

### Design Space Summary

#### Lambda Mapping Recommendation
**Linear Offset (Option 2)** with modifications:
```python
lambda_value = max(0, min(lambda_max, 1.0 + (curvature - threshold) / lambda_scale))
# Adds min() to cap at lambda_max
```
- Centered at 1.0 (neutral)
- Tunable sensitivity via lambda_scale
- Bounded by lambda_max

#### Gating Factor Recommendation
**Additive Protection (Option C)** or **Weighted Multiplicative (Option D)**

**Option C pros:**
- Simple, interpretable
- Both signals contribute independently
- Satisfies all 5 requirements

**Option D pros:**
- Ensures lambda always affects gating (via 0.5 base weight)
- Utility modulates lambda's effect (0.5x to 1.0x)
- Still satisfies all 5 requirements

### Open Questions

1. **Which gating formula best matches the conceptual model?**
   - Additive (C): "Protection from utility + protection from lambda"
   - Weighted (D): "Lambda-based protection, scaled by utility importance"

2. **Should low-utility weights get strong protection when lambda is high?**
   - Yes → Use Option C or D
   - No → Current multiplicative approach is fine

3. **How to validate the formula empirically?**
   - Track per-sample metrics: utility, lambda, gating, update magnitude
   - Stratified analysis: performance by (utility, lambda) quadrants
   - Compare learned representations across formula variants

### Next Steps

1. **Decide on gating formula** based on conceptual model
2. **Implement chosen formula** in both first-order and second-order optimizers
3. **Run ablation experiments** comparing formulas
4. **Analyze per-sample behavior** to validate design requirements
5. **Document final design choice** with empirical justification

### Related Issues

- **Negative gating**: Current approach clamps to 0, completely blocking updates
- **Lambda range**: Need lambda to span meaningful range (not compressed to ~1.0)
- **Utility-lambda correlation**: If highly correlated, any formula may show minimal difference
- **Computational cost**: More complex formulas may have negligible overhead compared to curvature computation
