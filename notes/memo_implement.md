# Memorization Implementation: Feasibility Assessment

This document assesses the feasibility of implementing memorization experiments to connect the "tiny utility tail fraction" to memorization/forgetting dynamics in continual learning.

---

## 1. The Core Hypothesis

The theory roadmap proposes this connection:

```
Tiny tail set I_tau = {i : u_hat_i >= tau * u_max}  <-->  "Consolidation set" protecting memorized content
         (~0.1-1% of params)                              (fragile, example-specific reliance)
```

**The claim**: Parameters in the high-utility tail are those that the network "needs to keep" because they encode hard/atypical examples that would otherwise be forgotten. Output-only gating works because it protects this set in the head (where label mappings live), without over-protecting representation parameters.

### Key Definitions (Ravikumar/Garg Framework)

| Proxy | Definition | CL Interpretation |
|-------|------------|-------------------|
| **CSL** | Cumulative Sample Loss: `sum_t loss(theta_t, z)` | Instability indicator; high CSL = frequently misclassified |
| **CSG** | Cumulative Sample Gradient: `sum_t ||grad_x loss(theta_t, z)||^2` | Learning difficulty; high CSG = model struggles |
| **Curv** | Input loss curvature: `tr(grad^2_x loss(theta, z))` | Fragility/sharpness; high Curv = memorization-like reliance |

---

## 2. What Already Exists (No New Code Needed)

| Metric | Status | Location |
|--------|--------|----------|
| `hist_52_56_pct` per layer | [x] Logged | JSON logs (`layer_utility_histogram_per_step`) |
| `raw_utility_max` per layer | [x] Logged | JSON logs (`layer_utility_max_per_step`) |
| Head vs hidden tail ratios | [x] Computed | 3.8-98x depending on dataset |
| Accuracy/loss over time | [x] Logged | JSON logs |
| Curvature function | [x] Exists | `compute_input_curvature_finite_diff()` in `input_aware.py` |

**Current findings (from reruns)**:
- CIFAR-10: output/hidden tail ratio ~ 3.8-5.0
- EMNIST: output/hidden tail ratio ~ 2.3-5.4
- Mini-ImageNet: output/hidden tail ratio ~ 1.3-2.1
- Input-MNIST: output/hidden tail ratio ~ 76-98

---

## 3. Feasibility Tiers

### Tier 1: Very Easy (~1 day)

| Experiment | Code Effort | What It Shows |
|------------|-------------|---------------|
| **Forgetting events (C3.1)** | ~50 lines | Track per-example correct->incorrect flips. Shows if forgetting concentrates in a subset. |
| **CSL accumulation** | ~20 lines | Sum `loss(x)` at each boundary. Free proxy for instability. |
| **Buffer storage** | ~30 lines | Store B_k (256 examples) per task for evaluation. |

**Implementation notes**:
- No curvature computation required
- Just forward passes at boundaries
- Adds minimal overhead to training

### Tier 2: Moderate (~2-3 days)

| Experiment | Code Effort | What It Shows |
|------------|-------------|---------------|
| **Curv vs forgetting (C2)** | ~100 lines | Does curvature increase as examples are forgotten? |
| **Curv-stratified retention** | ~150 lines | Do high-curv examples benefit most from output-only? |
| **CSG computation** | ~50 lines | Track `||grad_x loss||^2` per example at boundaries. |

**Implementation notes**:
- Reuse `compute_input_curvature_finite_diff()`
- Compute on 64 samples per buffer with niter=3 -> ~0.5s per buffer
- Call only at boundaries (every 2500 steps)

### Tier 3: Heavier (~1 week)

| Experiment | Code Effort | What It Shows |
|------------|-------------|---------------|
| **Linear probe recoverability (C3.3)** | ~200 lines | Does representation still contain info that head forgot? |
| **Bridge plot (C3.5)** | ~100 lines | Direct correlation: head tail mass <-> forgetting rate |
| **Representation drift probe** | ~150 lines | Track feature statistics drift across tasks. |

---

## 4. The Gap: Parameter-Level vs Example-Level

**Honest assessment**: The roadmap hypothesizes tail<->memorization but doesn't prove it.

| Evidence FOR | Evidence UNCLEAR |
|--------------|------------------|
| Tail is head-localized -> head mappings change under label permutation | We don't know if high-utility params correspond to "hard examples" |
| Clamping tail hurts accuracy -> tail matters for performance | Utility = gradient * param, not example-level difficulty |
| Output-only wins -> protecting head tail specifically helps | Tail might just be "recently active" params, not "memorization" |

**The conceptual gap**:
- **Utility** is a *parameter-level* statistic (`u_hat_i = -g_i * theta_i`)
- **Memorization** is an *example-level* phenomenon (fragile reliance on specific inputs)
- The bridge experiments (C3.2, C3.5) are designed to connect them

---

## 5. Recommendation for Current Paper

### For the UPGD Paper (sufficient as-is)

1. Keep Propositions A & B as stated:
   - **Proposition A**: Output-only advantage under target shift
   - **Proposition B**: Tail/top-k equivalence via global-max normalization

2. Use existing metrics:
   - Head tail dominance (hist_52_56_pct ratios)
   - Clamping sensitivity (43% accuracy drop)
   - Head raw_utility_max dominance

3. **Language recommendation**:
   - Say "selective consolidation" or "protection of high-utility parameters"
   - **Don't claim memorization** without the bridge experiments

### For Follow-up / Extended Version

1. Implement Tier 1 first (forgetting events + CSL)
2. Run the bridge plot (C3.5) -- directly tests if tail mass correlates with reduced forgetting
3. If positive correlation found, proceed with curvature experiments

---

## 6. Minimal Viable Memorization Experiment

The simplest experiment to test the tail<->memorization link:

```python
# === BUFFER STORAGE ===
# At each task boundary k (every 2500 steps):
#   1. Store buffer B_k containing 256 examples with labels as in task k
#   2. Record head tail mass: hist_52_56_pct['linear_3']

# === EVALUATION ===
# At each subsequent boundary t > k:
#   3. For each stored buffer B_j (j < current_task):
#      - Evaluate accuracy(B_j; theta_t) and loss(B_j; theta_t)
#      - Track per-example correctness: correct[j][example_id][t] = 1 or 0
#   4. Count forgetting events: flips from correct->incorrect

# === BRIDGE PLOT ===
# x-axis: head tail mass (hist_52_56_pct in linear_3) at time t
# y-axis: forgetting event rate on old buffers at time t
#
# Prediction: NEGATIVE correlation (more tail -> less forgetting)
```

**Estimated effort**: ~100 lines of code in `run_stats_with_curvature.py`

---

## 7. Implementation Sketch

### 7.1 Buffer Storage (add to `run_stats_with_curvature.py`)

```python
# Initialize
past_task_buffers = {}  # {task_id: {'inputs': tensor, 'targets': tensor}}
buffer_size = 256
forgetting_events = {}  # {task_id: {example_idx: [0,1,1,0,...]}}

# At task boundary k:
if step % task_switch_interval == 0:
    task_id = step // task_switch_interval
    # Sample buffer from current batch or recent data
    past_task_buffers[task_id] = {
        'inputs': current_inputs[:buffer_size].clone(),
        'targets': current_targets[:buffer_size].clone()
    }
    forgetting_events[task_id] = {i: [] for i in range(buffer_size)}
```

### 7.2 Forgetting Event Tracking

```python
# At each boundary, evaluate all past buffers:
for buf_id, buf in past_task_buffers.items():
    with torch.no_grad():
        outputs = model(buf['inputs'])
        preds = outputs.argmax(dim=1)
        correct = (preds == buf['targets']).cpu().numpy()

    # Track per-example correctness
    for i, c in enumerate(correct):
        prev = forgetting_events[buf_id][i]
        if len(prev) > 0 and prev[-1] == 1 and c == 0:
            # Forgetting event: was correct, now incorrect
            forgetting_event_count += 1
        forgetting_events[buf_id][i].append(int(c))
```

### 7.3 Bridge Plot Data

```python
# At each boundary, record:
bridge_data.append({
    'step': step,
    'head_tail_mass': utility_stats['layer_stats']['linear_3']['hist_52_56_pct'],
    'head_raw_umax': utility_stats['layer_stats']['linear_3']['raw_utility_max'],
    'forgetting_rate': forgetting_event_count / total_evaluated,
    'mean_old_buffer_acc': mean([acc(B_j) for B_j in past_task_buffers])
})
```

---

## 8. Expected Outcomes

### If Hypothesis is Correct

1. **Forgetting events concentrate** in a subset of examples (heavy-tailed distribution)
2. **Negative correlation** between head tail mass and forgetting rate
3. **Output-only** shows:
   - Fewer forgetting events overall
   - Largest reduction on "hard" examples (high-curv/high-CSL stratum)
4. **Clamped variants** show:
   - More forgetting events
   - Weaker correlation (tail collapsed -> no selective protection)

### If Hypothesis is Wrong

1. Forgetting is uniform across examples (no concentration)
2. No correlation between tail mass and forgetting rate
3. Output-only advantage explained by something else (e.g., just reduced interference, not selective protection)

---

## 9. Decision Points

### Q1: Is this needed for the current paper?

**Probably not**. The current findings (tail dominance, output-only wins, clamping hurts) are sufficient for one paper. The memorization link is a compelling extension but not required.

### Q2: What's the minimal experiment if we proceed?

**Forgetting events + bridge plot** (Tier 1 + part of Tier 3). Takes ~2-3 days including plotting.

### Q3: When would curvature be worth the compute?

Only if the bridge plot shows a clear correlation. Then curvature adds *why* (fragile examples) to the *what* (tail protects against forgetting).

---

## 10. Files to Modify

| File | Changes |
|------|---------|
| `core/run/run_stats_with_curvature.py` | Add buffer storage, forgetting tracking, bridge data logging |
| `core/optim/weight_upgd/first_order.py` | (No changes needed - already logs tail stats) |
| `upgd_plots/scripts/plot_bridge.py` | New file: scatter plot of tail mass vs forgetting rate |

---

## 11. Summary Table

| Experiment | Effort | Risk | Reward | Priority |
|------------|--------|------|--------|----------|
| Forgetting events | Low | Low | Medium | High (do first) |
| CSL accumulation | Very Low | Low | Medium | High |
| Bridge plot | Low | Medium | High | High (key test) |
| Curv vs forgetting | Medium | Medium | High | Medium (after bridge) |
| Linear probe | Medium | Low | Medium | Low (optional) |
| Representation drift | Medium | Low | Low | Low (optional) |

---

## 12. Conclusion

The memorization implementation is **feasible** with moderate effort. The key insight is to start with cheap proxies (forgetting events, CSL) and the bridge plot before investing in curvature computation.

**Recommended path**:
1. Finalize UPGD paper with current tail/output-only findings
2. Implement Tier 1 experiments as exploratory analysis
3. If bridge plot is positive -> write follow-up paper on tail<->memorization connection
4. If bridge plot is negative -> the tail story is about "consolidation" not "memorization" specifically

The tiny utility fraction (~0.1-1%) is definitively important for performance (clamping proves this). Whether it specifically corresponds to "memorization" in the Ravikumar/Garg sense requires the bridge experiments to confirm.

---

## 13. Current Codebase Status

### Curvature Computation: NOT Active

The curvature code exists but is **dormant** in current experiments:

```python
# From run_stats_with_curvature.py:227
if self.is_input_aware and i % self.compute_curvature_every == 0:
    current_curvature = self.learner.compute_input_curvature(...)
```

- `is_input_aware` checks if learner name contains `'input_aware'` or `'curvature_gating'`
- Current learners (`upgd_fo_global`, `upgd_fo_global_outputonly`, etc.) -> `is_input_aware = False`
- Result: **curvature is never computed** in current experiments

### To Enable Curvature

Option A: Use input-aware learner variant
Option B: Add separate flag for curvature-on-buffers (recommended for memorization experiments)

---

## 14. The Role of the Tiny Fraction

### What We Know

The ~0.1-1% of parameters in the high-utility tail:
1. Are **head-localized** (output layer has 3.8-98x more tail mass than hidden layers)
2. Are **performance-critical** (clamping causes 43% accuracy drop)
3. Correspond to parameters with large `|gradient * parameter|` product

### What We Hypothesize (Unproven)

This tail set:
1. Encodes "hard/atypical" examples that require specific parameter configurations
2. Acts as a "consolidation set" that prevents catastrophic forgetting
3. Is the mechanism by which output-only gating succeeds (protecting decisive head mappings)

### The Bridge

To connect parameter-level (tail) to example-level (memorization):
- Track which examples are repeatedly forgotten
- Correlate forgetting rate with tail statistics
- If high tail mass -> low forgetting rate, the hypothesis is supported

This is what the "bridge plot" tests.
