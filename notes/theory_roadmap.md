# Theory roadmap: why output-only gating wins + why tails matter

This note turns the paper's two key empirical signatures into *provable* claims:

- **(S1) Output-only gating dominates** on target-shift (label permutation) streams.
- **(S2) Tail sensitivity**: performance is driven by a tiny fraction of extreme-utility parameters; clamping away the tail degrades strongly.

The goal is not to perfectly model deep nets, but to produce (i) a clean mathematical abstraction whose conclusions match (S1)–(S2), and (ii) falsifiable predictions that connect back to logged utility statistics.

---

## 1) Minimal formalization of UPGD-W (as used in this paper)

Write the per-parameter update in the paper as

θₜ₊₁,ᵢ = (1 − ηλ)θₜ,ᵢ − η(gₜ,ᵢ + σεₜ,ᵢ)aₜ,ᵢ,    aₜ,ᵢ ≔ 1 − sₜ,ᵢ ∈ (0,1)

where the gate sₜ,ᵢ is formed from the (bias-corrected) EMA utility ûₜ,ᵢ via **global-max normalization**:

sₜ,ᵢ = σ(ûₜ,ᵢ / max(u_max,t, ε)),    u_max,t ≔ maxⱼ ûₜ,ⱼ

Key structural takeaway:

- UPGD-W is **SGD + isotropic noise with a diagonal, data-dependent step-size** aₜ,ᵢ.
- Because of the normalization by u_max,t, most coordinates typically satisfy ûₜ,ᵢ/u_max,t ≈ 0 ⇒ sₜ,ᵢ ≈ 0.5 ⇒ aₜ,ᵢ ≈ 0.5, while a tiny set near the maximum gets s ≈ 1 ⇒ a ≈ 0.

---

## 2) A two-block model for target shift (explains output-only dominance)

### 2.1 Decomposition

Model the network as a representation + head:

f(x; θ) ≡ f(x; V, W) = W φ(x; V)

and assume **cross-entropy** training on a stream with piecewise-stationary tasks (task switches every U updates).

### 2.2 Target-shift regime (label permutations)

Formalize "label permutation" as:

- The *feature distribution* over φ(x; V) is roughly stable across tasks (or changes slowly).
- The *optimal head* W*ₖ changes abruptly across tasks k (often exactly by a permutation of class labels).

This captures the empirical setting where:

- the representation can remain useful, but
- the head must continually remap features to new labels.

### 2.3 Theorem target (what to prove)

Prove a statement of the form:

> **Proposition A (output-only gating advantage under target shift).**
> In a target-shift stream where the head optimum W*ₖ moves significantly across tasks while the representation optimum V* is approximately stable, applying utility-gated damping/noise to W while leaving V ungated yields lower *dynamic regret* (or faster tracking) than gating both W and V.

**Why this matches your results.**

- Gating in V reduces effective step-sizes in representation coordinates that must keep adapting to avoid plasticity loss and to maintain separability under cross-entropy.
- Head utilities are "closer to the loss" (higher SNR), so gating decisions are more reliable at W than at V.

### 2.4 Proof strategy (tractable but faithful)

Use a *linearized cross-entropy* analysis on the head:

1. Condition on the representation φ(x; V) at a time scale of one task.
2. Analyze online logistic regression / multiclass softmax on features z = φ(x; V).
3. Treat the gate aₜ on W as a diagonal preconditioner and show a tracking bound.

Then compare:

- **Output-only**: gating W, while V uses a constant scaling (your 0.5 baseline).
- **Full gating**: gating both W and V, which effectively reduces representation learning rate in "high utility" coordinates.

You can make this rigorous via:

- a dynamic regret bound for online convex optimization with time-varying comparator W*ₜ, and
- a (possibly simplified) assumption that representation drift required per task is non-negligible, so damping V hurts tracking/generalization.

### 2.5 Proof sketch + intuition (output-only advantage)

This is easiest if you treat the network as “features + head” and only make the head part fully formal.

**Setup (the part we can prove cleanly).**
Let zₜ = φ(xₜ; Vₜ) and consider the per-step head loss ℓₜ(W). For fixed features zₜ, multiclass logistic/softmax is convex in W, so standard online convex optimization tools apply. Model target shift as:

- tasks last U steps, and there is an “instantaneous best head” W*ₖ per task k;
- at a task boundary, W* jumps (label permutation ⇒ big jump), while zₜ’s distribution changes slowly.

The head update under UPGD-W is just preconditioned noisy SGD:

Wₜ₊₁ = Wₜ − η · Aₜ · (∇ℓₜ(Wₜ) + σ εₜ),

where Aₜ = diag(aₜ,i) and aₜ,i = 1 − σ(ûₜ,i / u_max,ₜ) ∈ (0,1).

**Key fact (dynamic regret shape).**
For preconditioned online gradient descent, a standard “tracking” bound has the qualitative form:

total regret  ≲  (initial error)/η  +  η·(gradient/noise term)  +  (drift term weighted by Aₜ).

The last term is the important one here: it scales with how fast the comparator sequence W* moves, and how much step-size Aₜ gives you in the relevant coordinates. Under label permutations, the drift of W* across tasks is large, so you want the head to keep “tracking capacity.”

**Where output-only wins.**
The only place “full gating” can hurt relative to “output-only” is the representation block V:

- In output-only, V is not additionally damped: it keeps a baseline learning rate (≈ constant 0.5 scaling in your implementation), so features can keep adapting/maintaining separability across tasks.
- In full gating, V is also damped in some coordinates (and perturbed differently), which can reduce feature quality zₜ and effectively make the head problem harder (worse gradients / worse conditioning / slower tracking).

Intuition: target shift primarily demands rapid remapping in the head. If you also throttle the representation, you pay a “feature drift / feature degradation” penalty that compounds the already-large head drift problem. Output-only avoids paying that extra penalty while still protecting a small subset of head parameters.

**What you write in the paper.**
State the setup assumptions (piecewise-stationary tasks, head optimum jumps, features slow drift), cite one dynamic regret lemma for OGD (or mirror descent) and then add a short paragraph explaining that gating V adds an extra error term through zₜ, which is not worth it in head-dominated non-stationarity.

### 2.6 Experiment → theory: what the reruns actually validate (and what they refine)

The reruns were designed to test the *mechanistic assumptions* behind Proposition A, not to re-prove the whole performance claim.

**What we measured.**
From the new per-step JSON logs, we measured (per layer ℓ ∈ {linear_1, linear_2, linear_3}):
- A high-utility “shoulder” mass: `hist_52_56_pct` from `layer_utility_histogram_per_step[ℓ]`
- A layerwise extreme: `raw_utility_max` from `layer_utility_max_per_step[ℓ]`
- A task-boundary modulation heuristic: ratio of `raw_utility_max` at the nearest boundary step to the median in the preceding window

**Why these are relevant to Proposition A.**
Proposition A needs two empirical premises to be believable:
- **(A-premise 1) Head-dominated non-stationarity:** the head experiences “sharper” utility statistics than hidden layers under target shift.
- **(A-premise 2) Output-only does not remove head selectivity:** even when only the head is gated, the head still concentrates the extreme/tail behavior; i.e. the mechanism is not an artifact of also gating hidden layers.

**What we observed (seed=2, last 20% of training; output layer vs mean(hidden layers)).**
- Across all four datasets, the output layer shows **larger high-utility shoulder mass** and **larger raw utility maxima** than hidden layers:
  - CIFAR-10 tail ratio ≈ 3.8–5.0; raw_u_max ratio ≈ 1.4–2.4
  - EMNIST tail ratio ≈ 2.3–5.4; raw_u_max ratio ≈ 2.5–5.5
  - Mini-ImageNet tail ratio ≈ 1.3–2.1; raw_u_max ratio ≈ 2.2–7.3
  - Input-MNIST tail ratio ≈ 76–98; raw_u_max ratio ≈ 2.6–3.5

Interpretation: this directly supports (A-premise 1) and (A-premise 2). The head concentrates the “interesting” part of the gate statistics regardless of whether hidden layers are gated.

**Implication for proofs (how to write assumptions more confidently).**
In the paper, you can now explicitly assume a **head-localized heavy upper tail** for the utility ratios (and/or the max–bulk gap), i.e. “the head contributes most of the extreme values that define u_max and most of the mass above the σ(0)=0.5 baseline region.” This justifies treating the head as the primary locus where global-max normalization meaningfully modulates step-sizes, while the representation is closer to constant-step.

**Refinement: boundary excursions are not universal.**
The earlier intuition “raw_u_max spikes at task boundaries” holds strongly for label-permuted CIFAR-10/EMNIST but is weaker (sometimes even dip-like under the crude ratio metric) for Mini-ImageNet, and nearly flat for Input-MNIST. This suggests:
- boundary-local modulation is a *property of the task stream + representation stability*, not a guaranteed property of max-normalization itself.

This is useful for the theory writeup: you can state the boundary-excursion prediction as **conditional** (strongest when the head optimum jumps while features remain usable).

---

## 3) Tail sensitivity from global-max normalization (explains clamping results)

### 3.1 Key observation: global-max normalization forces "winner-take-most" gating

Because u_max,t = maxᵢ ûₜ,ᵢ, the ratio rₜ,ᵢ = ûₜ,ᵢ/u_max,t ∈ [0,1] is typically:

- near **1** only for a very small set of coordinates (those close to the maximum),
- near **0** for the vast majority.

After the sigmoid, this means:

- most parameters get s ≈ σ(0) = 0.5 ⇒ a ≈ 0.5,
- only the extreme-utility tail gets s ≈ 1 ⇒ a ≈ 0.

So UPGD-W behaves like:

> **"Protect a tiny extreme-utility set; apply roughly constant noisy SGD to everyone else."**

### 3.2 Theorem target (tail mass controls algorithmic behavior)

> **Proposition B (effective top-k gating).**
> Fix a threshold τ ∈ (0,1). Let
> ℐₜ(τ) = {i : ûₜ,ᵢ ≥ τ·u_max,t}.
> Under mild conditions on the utility distribution, |ℐₜ(τ)| ≪ N while the complement satisfies aₜ,ᵢ ≈ 0.5. Therefore the algorithm's deviation from constant-step noisy SGD is controlled primarily by the tail mass |ℐₜ(τ)| and the gap between u_max,t and the bulk.

This proposition gives an immediate explanation for clamping:

- Clamping (especially to [0.48, 0.52]) removes or shrinks the set ℐₜ(τ) and reduces the max–bulk gap.
- The gate collapses toward a near-constant factor, destroying selective protection and causing the large accuracy drop you observed.

### 3.3 Proof strategy (order statistics / EVT-lite)

You do *not* need full extreme value theory. A simple route:

1. Assume {ûₜ,ᵢ}ᵢ₌₁ᴺ have a distribution with a non-trivial upper tail (even sub-Gaussian is fine).
2. Use a concentration inequality or a quantile bound to show u_max,t is well-separated from the median/mean as N grows.
3. Show that for most i, rₜ,ᵢ is small and thus sₜ,ᵢ ≈ 0.5.

Empirically, you already see this: ~99.7–100% of parameters in the central bin.

### 3.4 Proof sketch + intuition (tails ⇒ “effective top‑k protection”)

The entire story is driven by one normalization: rₜ,i = ûₜ,i / u_max,ₜ ∈ [0,1].

**Tail set.**
Fix τ ∈ (0,1) and define the “near-max” (tail) indices:
Iₜ(τ) = { i : ûₜ,i ≥ τ · u_max,ₜ }.

**Bulk is almost constant-step.**
For i ∉ Iₜ(τ), we have rₜ,i ≤ τ, and empirically most ratios are near 0 (because u_max is far above the bulk). Since σ(0)=0.5, this means:
sₜ,i = σ(rₜ,i) ≈ 0.5  ⇒  aₜ,i = 1 − sₜ,i ≈ 0.5
for almost all coordinates. So most parameters behave like “noisy SGD with a constant 0.5 multiplier.”

**Only the extreme tail behaves differently.**
The only coordinates that get meaningfully different step-sizes are those with rₜ,i close to 1, i.e., i ∈ Iₜ(τ). Those are the ones pushed toward s≈1, a≈0 (“frozen / protected”). In other words:

UPGD-W ≈ “baseline noisy SGD”  +  “extra protection on a tiny tail set.”

That is exactly why the *tail mass* |Iₜ(τ)| (or its layerwise fraction) is the right control knob: it quantifies how far you are from a constant-step optimizer.

**Why clamping breaks performance.**
Clamping compresses the ratio distribution rₜ,i (and/or shrinks the max–bulk gap). That collapses gate variability: fewer coordinates reach r≈1, and the algorithm becomes closer to constant-step noisy SGD everywhere. If performance depends on selectively protecting a tiny set (your experiments suggest it does), then removing that tail inevitably hurts.

**What you write in the paper.**
Define rₜ,i and Iₜ(τ), state the “bulk ≈ 0.5 multiplier, tail ≈ frozen” observation in one paragraph, and then connect clamping to “gate variance collapses ⇒ selective protection disappears.”

### 3.5 Experiment → theory: what the reruns validate about “tails” (and what metric we still need)

The core claim in Proposition B is about the near-max set Iₜ(τ) = {i : ûₜ,i ≥ τ·u_max,ₜ}. Our current logs do not store p(|Iₜ(τ)|) directly, but the reruns still strengthen the theory in three concrete ways.

**(B-support 1) Layerwise max domination is real (the “u_max comes from the head”).**
`raw_utility_max` is consistently larger in `linear_3` than in hidden layers (ratios above). This supports the structural picture that the global max is typically sourced from the head (or at least the head contains the strongest extremes), which is exactly the condition under which global-max normalization becomes “head-controlled.”

**(B-support 2) Even a shoulder-tail proxy localizes in the head.**
`hist_52_56_pct` is not a strict near-max statistic, but it measures how much mass is pushed above the σ(0)=0.5 baseline region (recall s=σ(r), so crossing r≈0.5 moves you into the high-s regime). The fact that this shoulder mass is head-dominated across datasets suggests:
- the ratio distribution rₜ,i is not only head-extreme at the very top, it is head-shifted in its upper tail region more broadly.

This is consistent with a “top‑k protection” mechanism: the head has a heavier upper tail, hence it contributes disproportionately to the set of coordinates whose gates deviate from the baseline.

**(B-support 3) Dataset dependence hints at when tail sensitivity matters most.**
Mini-ImageNet shows weaker head dominance in the shoulder mass than CIFAR-10/EMNIST, while still showing strong head dominance in raw_u_max. A plausible reading is:
- the existence of extremes (u_max) is not enough; the *mass near the upper tail* determines how many parameters are meaningfully protected.
This aligns with Proposition B’s emphasis on tail mass, not just the maximum.

**Missing (easy to add later): true τ-tail mass.**
To make Proposition B fully tight empirically, add one extra logged time series per layer:
- p_ℓ,t(τ) = (1/N_ℓ)·|{i ∈ ℓ : ûₜ,i ≥ τ·u_max,ₜ}| for τ ∈ {0.9, 0.99}
This would let the experiments match the statement of Proposition B exactly.

**Implication for proofs (what the experiments let you “take as given”).**
For Proposition B, the reruns justify assuming that (i) layerwise maxima satisfy u_max,head ≫ u_max,hidden often, and (ii) the head’s utility-ratio distribution has a heavier upper tail (in the sense that even fixed-bin “upper shoulder” mass is larger in the head). In the proof sketch, you can therefore model global-max normalization as effectively using a head-controlled scale, and analyze the resulting behavior as “baseline step-size for most coordinates + selective suppression on a small head-dominated set.”

---

## 4) What statistics to extract from your saved utility logs (to "close the loop")

From the raw utilities per step (or per task boundary), compute:

- **Tail mass**: for each layer ℓ, p_ℓ,t(τ) = (1/N_ℓ)|{i ∈ ℓ : ûₜ,ᵢ ≥ τ·u_max,t}| for τ ∈ {0.9, 0.99}.
- **Max–bulk gap**: u_max,t / q₀.₅,t or u_max,t / q₀.₉,t (layer-wise and global).
- **Gate entropy**: entropy of {aₜ,ᵢ} (or variance of a) as a scalar "how selective is the gate?" metric.
- **Head vs hidden**: compare tail mass and max–bulk gap for output layer vs hidden layers.

Observed signature (from the seed=2 reruns) that supports (S1)–(S2):

- Output layer has **larger high-utility shoulder mass** (`hist_52_56_pct`) and **larger raw utility maxima** than hidden layers across datasets.
- The strength of “boundary excursions” depends on the dataset/stream: it is strong for label-permuted CIFAR-10/EMNIST and weaker elsewhere.

---

## 5) Concrete, falsifiable predictions (helpful for theory + future experiments)

- **P1 (normalization choice)**: replacing global-max normalization with a high quantile (e.g., u₀.₉₉) will reduce tail sensitivity and broaden the gate distribution; clamping should become less harmful.
- **P2 (layer reliability)**: the correlation between utility and "true importance" (e.g., loss increase under coordinate noise) should be higher in the output layer than in early layers.
- **P3 (top-k equivalence)**: a simplified optimizer that explicitly freezes the top-k utilities (per step) and uses constant scaling for the rest should approximate UPGD-W behavior when k matches the observed tail mass.
- **P4 (curvature-as-memorization proxy in continual learning; Ravikumar/Garg line)**: define per-example input loss curvature Curv(x; θ) (trace-based proxy). Then, for a fixed buffer of past-task examples, Curv should increase as those tasks are forgotten (often tracking or preceding accuracy drops). Methods that retain better (e.g., Output-only) should show smaller curvature growth on past-task buffers than baselines.
- **P5 (tail ↔ memorization bridge)**: if the “protected tail set” Iₜ(τ) behaves like a consolidation set, then improvements from Output-only should be disproportionately visible on *high-curvature* examples (the “hard/atypical” subset). Concretely, stratify examples by curvature-at-learn-time; retention gains should be largest in the top-curvature quantile.

---

## 6) Next steps (what I need from you, and what we’ll produce)

### 6.1 Immediate “close-the-loop” additions (based on existing logs)

The fastest next step is to directly visualize the *layerwise* extremes and tails that the theory claims matter.

- **Done (seed=2 reruns completed)**: the JSON logs now include the new per-layer max time series (`layer_utility_max_per_step`), so we can directly test “head max dominance” and “boundary excursions”.
- Generate a compact **time-series figure** from the per-layer tail histograms (e.g., \([0.52,0.56)\) mass per layer over training).
- Generate a companion **per-layer max-over-time figure** (raw utility max) to test the “head max is larger + spikes at task boundaries” signature.
- Add a **curvature-as-memorization proxy** analysis (Ravikumar/Garg line): compute input loss curvature on a fixed buffer of past-task examples and track it across tasks.

### 6.2 After the reruns finish (exact checklist)

**A) Confirm the new logs contain per-layer max time series.**

For each finished rerun JSON (seed 2), verify it contains:

- `layer_utility_histogram_per_step` (already existed): per-layer tail bins over time
- `layer_utility_max_per_step` (new): per-layer `raw_utility_max` and `utility_max` over time

**Where to look.** The JSONs live under:

- `logs/<task>/<learner>/fully_connected_relu_with_hooks/<hparam_string>/2.json`

with `<task> ∈ {label_permuted_emnist_stats, label_permuted_cifar10_stats, label_permuted_mini_imagenet_stats, input_permuted_mnist_stats}` and `<learner> ∈ {upgd_fo_global_outputonly, upgd_fo_global_hiddenandoutput}` for the 8 reruns.

**B) Produce the plots (paper-ready PNGs).**

Submit one plot job per dataset:

- `sbatch upgd/slurm_plots/plot_layer_tail_over_time.sh emnist`
- `sbatch upgd/slurm_plots/plot_layer_tail_over_time.sh cifar10`
- `sbatch upgd/slurm_plots/plot_layer_tail_over_time.sh mini_imagenet`
- `sbatch upgd/slurm_plots/plot_layer_tail_over_time.sh input_mnist`

Expected outputs:

- `upgd_plots/figures/<dataset>/layer_tail_over_time(.png|_log.png)`
- `upgd_plots/figures/<dataset>/layer_raw_umax_over_time(.png|_log.png)` (only if the reruns produced `layer_utility_max_per_step`)

**C) What to check in the figures (theory-aligned signatures).**

**Observed from the reruns (seed=2, last 20% of training; using `hist_52_56_pct` and `raw_utility_max`).**

Notes:
- `hist_52_56_pct` is a convenient “high-utility shoulder” proxy around 0.54 (not a strict near-max tail). It is still useful because it measures how much probability mass gets pushed above the σ(0)=0.5 baseline region.
- A more direct “near-max” tail would be threshold-based (e.g., û ≥ 0.9·u_max); current logs don’t store that exact statistic yet.

**Head tail dominance (output layer concentrates high-utility shoulder mass).**
- CIFAR-10: output/hidden tail ratio ≈ 3.8–5.0; tail mean last20% ≈ 1.06–1.08% in `linear_3`.
- EMNIST: output/hidden tail ratio ≈ 2.3–5.4; tail mean last20% ≈ 0.24–0.39% in `linear_3`.
- Mini-ImageNet: output/hidden tail ratio ≈ 1.3–2.1 (weaker but still >1 on average).
- Input-MNIST: output/hidden tail ratio ≈ 76–98; tail mean last20% ≈ 17.7–18.6% in `linear_3` while hidden layers stay <0.5%.

**Head max dominance (output layer has larger raw utility maxima).**
- Across all four datasets, output/hidden `raw_utility_max` ratios are >1, typically:
  - CIFAR-10: ≈ 1.4–2.4
  - EMNIST: ≈ 2.5–5.5
  - Mini-ImageNet: ≈ 2.2–7.3
  - Input-MNIST: ≈ 2.6–3.5

**Task-boundary excursions (dataset-dependent).**
- On label-permuted CIFAR-10 and EMNIST, `raw_utility_max` shows stronger boundary-local modulation in `linear_3` (e.g., median boundary spike ratio ≈ 1.21 for CIFAR-10 output-only head; ≈ 1.53 for EMNIST output-only head).
- On Mini-ImageNet the boundary modulation is weaker / can appear as dips (<1 in the simple spike ratio), and on Input-MNIST it is ~flat (~1).

**Where to see it.** The plots are already generated:
- `upgd_plots/figures/cifar10/layer_tail_over_time*.png`, `.../layer_raw_umax_over_time*.png`
- `upgd_plots/figures/emnist/layer_tail_over_time*.png`, `.../layer_raw_umax_over_time*.png`
- `upgd_plots/figures/mini_imagenet/layer_tail_over_time*.png`, `.../layer_raw_umax_over_time*.png`
- `upgd_plots/figures/input_mnist/layer_tail_over_time*.png`, `.../layer_raw_umax_over_time*.png`

**C2) Curvature-as-memorization proxy checks (continual learning).**

These are the simplest experiments that connect “tails” to a memorization/fragility proxy without doing privacy/MIA.

**Proxy vocabulary alignment (Ravikumar/Garg).**
To match the definitions in:
- `Ravikumar et al., 2024 - Towards memorization estimation, Fast, formal and free.md` and
- `Ravikumar et al., 2024 - _Unveiling privacy, memorization, and input curvature links_.md`,
we will treat three quantities as “memorization proxies” in the continual-learning (CL) setting:

- **CSL (Cumulative Sample Loss)**: for an example z, CSL(z) = ∑ₜ ℓ(θₜ, z). In CL, this becomes **cumulative loss over the stream** (optionally restricted to “after the example is first stored/learned”).
- **CSG (Cumulative Sample Gradient proxy)**: Ravikumar’s “learning condition / learning time” is defined via the average squared input-gradient norm (‖∇ₓ ℓ(θₜ, z)‖²). In our CL usage, treat CSG as the **cumulative (or time-averaged) sample gradient magnitude**, preferably in input space:
  - CSGₓ(z) = ∑ₜ ‖∇ₓ ℓ(θₜ, z)‖₂²  (closest to their Eq. 6 / Eq. 7 learning condition style)
  - (optional, cheaper alternative) CSG_θ(z) = ∑ₜ ‖∇_θ ℓ(θₜ, z)‖₂² for a chosen block (head-only vs hidden-only)
- **Curv (Input loss curvature)**: Curv(z; θ) ≈ tr(∇ₓ² ℓ(θ, z)) via Hutchinson / finite differences (as in Garg/Ravikumar). This is the most “sharpness-like” and connects to stability-based memorization theory.

Why we want all three:
- CSL is “free” during evaluation passes and captures noisy learn/unlearn dynamics robustly (Ravikumar’s motivation).
- CSG is a CL-friendly proxy for “learning difficulty / instability” (large gradients persist when the model is struggling).
- Curv is the closest proxy to stability-based memorization bounds and “fragile reliance”.

- **Curvature-vs-forgetting on buffers (core test)**:
  - At each task boundary k, save a probe buffer Bₖ of size M (recommend M=256) containing (x, yₖ) pairs *as labeled in task k*.
  - After each subsequent task boundary (or every N tasks), evaluate on every stored buffer Bⱼ:
    - accuracy(Bⱼ; θₜ), loss(Bⱼ; θₜ)
    - Curv stats on Bⱼ; θₜ: mean, median, 90p (optionally max)
  - Prediction: as task j is forgotten, Curv(Bⱼ; θₜ) increases (often alongside or before accuracy drop). Output-only should show smaller Curv growth than Hidden+Output and baselines.

  **Compute-light default**: compute curvature on a random subset of each buffer (e.g., 64 points) and use niter=3–5 with h=1e−3 in the finite-diff estimator.

- **CSL/CSG-vs-forgetting on buffers (same protocol, cheaper than Curv)**:
  - At each boundary evaluation of Bⱼ, also record per-example:
    - ℓ(θₜ, z) (for CSL accumulation)
    - ‖∇ₓ ℓ(θₜ, z)‖₂² (for CSGₓ accumulation; optionally head/hidden CSG_θ blocks)
  - Define per-example cumulative proxies over time:
    - CSL_stream(z) = ∑_{t ∈ eval_times} ℓ(θₜ, z)
    - CSGₓ,stream(z) = ∑_{t ∈ eval_times} ‖∇ₓ ℓ(θₜ, z)‖₂²
  - Prediction (CL version of Ravikumar intuition):
    - examples with higher CSL/CSG are the ones that are forgotten more often / earlier
    - Output-only reduces CSL/CSG growth on old buffers relative to Hidden+Output (better retention / less “relearning”)

- **Curvature-stratified retention (shows “memorized exceptions”)**:
  - During task k, compute Curv(x; θ) for examples you will store (or for mini-batches and then sample points).
  - Split Bₖ into quantiles by Curv-at-learn-time (low/med/high; e.g., bottom 50%, 50–90%, top 10%).
  - Track retention for each stratum over time: accuracy and Curv growth.
  - Prediction: high-curvature stratum is forgotten more under baselines; Output-only gives the largest retention gains on that stratum (consistent with “tail protects fragile content”).

- **CSL/CSG-stratified retention (often more stable than Curv under low compute)**:
  - At learn-time for task k, compute CSLₖ(z) over a short window (few evaluations) and CSGₓ,ₖ(z) at boundary.
  - Stratify Bₖ by high/low CSL and/or high/low CSG and track forgetting.
  - Prediction: high-CSL/high-CSG strata are the “memorization-heavy” strata; Output-only gains are largest there.

- **Tail/curvature co-movement (bridge to the tail theory)**:
  - At task boundaries, record:
    - Curv spikes: ΔCurv = Curv(t_boundary+δ) − Curv(t_boundary−δ) on the current-task batch or on Bₖ
    - head tail mass (e.g., hist_52_56_pct in linear_3) and head raw_utility_max
  - Prediction: in target shift streams, these co-spike; the strongest co-movement is in the head.

  CL-aligned extension (Ravikumar proxies):
  - also record ΔCSL and ΔCSG at boundaries on (i) current-task batch and (ii) a fixed past-task buffer
  - expectation: head tail/selectivity measures co-move most strongly with CSL/CSG dynamics on past buffers when forgetting happens

**C3) Better ways to understand the role of memorization (diagnostics aligned with the theory).**

You’re right to prioritize *understanding* over “causal ablations.” Here are analysis-first experiments that tell you (i) whether forgetting is driven by “memorization-heavy” content, (ii) where that content lives (head vs hidden), and (iii) whether UPGD’s head-localized tail/max signatures are the mechanism that stabilizes it.

Guiding idea: treat “memorization” as *fragile, example-specific reliance* that shows up as high sensitivity / sharp loss landscape around those examples, and as prediction instability across tasks.

**C3.1 Per-example forgetting events + stability (the clean CL memorization readout).**
Goal: quantify memorization as “learned then forgotten” on past-task buffers, and see if it concentrates in a subset of examples.

Protocol:
- On each buffer Bₖ, record per-example correctness over time (at boundaries): 1{ŷ(x)=y}.
- Define *forgetting events* for an example as flips from correct→incorrect after being correct at least once.
- Summaries to plot:
  - distribution of #forgetting-events per example (heavy tail = memorization-like fragility)
  - fraction of examples never forgotten vs repeatedly forgotten

Theory link:
- If the head-localized tail is a consolidation mechanism, Output-only should reduce the heavy tail of forgetting events (especially for “hard/atypical” examples).

**C3.2 Curvature as a memorization proxy (Ravikumar/Garg) but with CL-specific alignment.**
Instead of only tracking mean Curv, treat Curv as an *example tag*:
- At learn-time (task k), compute Curv(x; θ) on candidates for Bₖ and store it as Curvₖ(x).
- Later, relate Curvₖ(x) to:
  - probability of being forgotten
  - time-to-forget (age until first forgetting)
  - loss increase under task shift

Theory link:
- This tests the claim “UPGD mainly stabilizes a small head set that matters disproportionately for fragile/memorization-heavy examples.”

**C3.2b CSL/CSG as memorization proxies (Ravikumar “Fast, formal and free” framing, CL adaptation).**
In CL, CSL/CSG are often the most actionable because they can be computed at every boundary with little overhead.

Recommended definitions for our setting:
- CSL_after_store(z ∈ Bₖ) = ∑_{t ≥ store_time(k)} ℓ(θₜ, z)
- CSGₓ,after_store(z ∈ Bₖ) = ∑_{t ≥ store_time(k)} ‖∇ₓ ℓ(θₜ, z)‖₂²

Key plots:
- CSL/CSG distributions per task-age (do they become heavy-tailed as tasks age?)
- forgetting-events vs CSL/CSG (do high-CSL/high-CSG examples drive forgetting?)
- correlation with head tail/max metrics (does more head selectivity mean lower CSL/CSG growth?)

Theory link to our UPGD story:
- Proposition B says “effective top‑k protection” is controlled by a tiny set; CSL/CSG give an example-level readout of whether that protection actually reduces repeated relearning/instability on old examples.

**C3.3 Head-vs-hidden localization of memorization (where does the “memory” live?).**
Goal: test the head-localization hypothesis *directly* at the level of representation vs head, without changing training.

Two complementary probes:
- **Representation drift probe**:
  - At each boundary, compute feature statistics on Bⱼ: mean z, covariance of z, and class-conditional means μ_c.
  - Track how much those drift across tasks (e.g., ||μ_c(t)−μ_c(t−1)||).
  - If drift is small but accuracy still drops, that implicates the head mapping (W) rather than representation (V).
- **Linear-probe “head recoverability”**:
  - Freeze V at time t and refit a fresh linear head W′ on each past buffer Bⱼ (small ridge regression / logistic regression).
  - Compare recovered accuracy to the model’s current head accuracy.
  - Large gap ⇒ the representation still contains the information, but the head mapping forgot it.

Theory link:
- Proposition A argues target-shift primarily stresses the head mapping; this quantifies that mismatch directly.

**C3.4 “Logit geometry” view of memorization (margin + confidence collapse).**
Goal: detect memorization-like behavior via how margins/confidence evolve on old examples.

Protocol (on buffers at boundaries):
- Track per-example margin: (logit_true − max logit_other)
- Track entropy of predicted distribution
- Track calibration gap on old buffers (ECE-like, optional)

Theory link:
- If head-tail protection stabilizes decisive mappings, Output-only should maintain margins on old examples (especially high-curv ones), even when mean accuracy changes slowly.

**C3.5 Connect memorization diagnostics to the tail/max stats you already logged.**
This is the key “bridge plot” that closes the loop without ablations:
- For each boundary t and method:
  - x-axis: head tail/shoulder mass (e.g., `linear_3` `hist_52_56_pct`) and head `raw_utility_max`
  - y-axis: memorization diagnostic on past buffers (e.g., #forgetting-events rate, mean margin drop, Curv growth)

Prediction:
- Methods/times with more head selectivity (higher tail/shoulder mass and/or higher head raw_u_max relative to hidden) show **lower forgetting-event rate** and **lower Curv growth** on past buffers.

**Implementation notes (so this is doable in one pass).**

- **Where to compute Curv**: reuse the existing finite-diff estimator `compute_input_curvature_finite_diff` (returns per-sample values). You do not need ZO/black-box estimation for this purpose.
- **When to compute Curv**: do it only at boundaries (every U steps) to keep cost low; you already know U from the task change frequency.
- **What to save**:
  - buffers Bₖ (store tensors in-memory during training; optionally dump to disk in the final JSON for reproducibility)
  - per-boundary summary arrays: for each buffer id j and boundary t, store mean/90p Curv and accuracy.
- **Minimal outputs**:
  - Plot 1: Curv mean/90p vs task age (how old the buffer is) for each method
  - Plot 2: accuracy vs Curv for buffers (scatter), colored by method
  - Plot 3: curvature-stratified forgetting curves (3 lines per method: low/med/high-curv strata)
  - Plot 4: CSL/CSG mean/90p vs task age (same style as Curv, cheaper to compute)
  - Plot 5 (diagnostic bridge): forgetting-event rate / margin drop / ΔCSL / ΔCSG vs head tail/shoulder mass and head raw_u_max

**D) Paper integration (minimal).**

Add one paragraph + a figure reference:

- “Output layer concentrates both the high-utility tail mass and the largest utility maxima, supporting a head-centric non-stationarity picture under target shift.”
- “Input loss curvature (Curv) on past-task buffers increases as tasks are forgotten; Output-only reduces Curv growth, consistent with selective consolidation of a small parameter subset.”

### 6.3 Point me to the raw utility logs

Please locate the directory (or files) that contain *raw utilities* (or raw gates) for at least one run/dataset, ideally saved per step or per task boundary. Common formats:

- `.npz/.npy` (NumPy arrays)
- `.pt/.pth` (PyTorch tensors)
- `.pkl` (pickles)
- `.json/.csv` (tabular summaries)

If you’re not sure where they are, search for filenames containing substrings like:

- `utility`, `u_max`, `gate`, `s_t`, `a_t`, `hist`, `bins`, `clamp`

Once you tell me the path(s), I’ll write a small extractor script and produce a compact table of tail metrics.

### 6.4 What I will extract (paper-ready numbers)

For each dataset × method × layer (hidden vs output), at (i) end-of-training and (ii) around task switches:

- **Tail mass (two options)**:
  - **Available now from logs**: `hist_52_56_pct` (and other fixed bins) per layer over time.
  - **More theory-direct (requires a small logging addition)**: p_ℓ,t(τ) for τ ∈ {0.9, 0.99} defined by û ≥ τ·u_max,t.
- **Max–bulk gap**: u_max,t/q₀.₅,t and u_max,t/q₀.₉,t (layerwise + global).
- **Gate selectivity**: Var(aₜ,·) (or entropy of aₜ,·), layerwise.
- **Stability across tasks**: how these quantities change immediately after a label permutation.

Deliverable: a table like  
`Dataset | Method | Layer | p(0.99) | u_max/q0.5 | Var(a) | Acc`  
plus 1–2 supporting plots (optional).

### 6.5 What I will write next (theory text you can paste into the paper)

I’ll convert Sections 2–3 into a short, formal subsection containing:

- **Proposition A (output-only advantage)** in a head/representation model under target shift, with assumptions stated clearly (cross-entropy, piecewise-stationary tasks, head-dominated drift).
- **Proposition B (tail/top-k equivalence)** showing global-max normalization yields effective “protect an extreme set” behavior; clamping reduces max–bulk gap and collapses selectivity.
- A paragraph explicitly tying the propositions to your extracted metrics (tail mass, max–bulk gap, Var(a)).

### 6.6 Minimal checks (to validate the story quickly)

Before polishing proofs, we’ll do two quick validations from the logs:

- **Check 1 (head vs hidden)**: output layer has noticeably larger tail mass and/or larger max–bulk gap than hidden layers, especially for Output-only.
  - **Confirmed in reruns** (ratios above).
- **Check 2 (clamping)**: clamped runs show sharply lower Var(a) / tail mass and correspondingly lower accuracy.
- **Check 3 (max-over-time)**: output layer’s `raw_utility_max` is larger than hidden layers; boundary excursions are strongest on label-permuted CIFAR-10/EMNIST and weaker elsewhere.
- **Check 4 (curvature-as-memorization proxy)**: mean/90p Curv on past-task buffers increases with forgetting; Output-only reduces Curv growth (especially for the high-curvature quantile).
- **Check 5 (memorization link, non-causal)**: head tail/shoulder mass and head raw_u_max track “memorization diagnostics” (forgetting-event rate, margin collapse, Curv growth) on past-task buffers, with the strongest association in the head.
- **Check 6 (CSL/CSG alignment)**: high-CSL/high-CSG examples are the ones with frequent forgetting events; Output-only reduces CSL/CSG accumulation on past-task buffers and strengthens the coupling between head selectivity (tail/max) and reduced forgetting.
