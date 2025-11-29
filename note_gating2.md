### Interpreting your requirement (clear, testable)
- Define update multiplier g(u, λ) in [0,1]; smaller g = more protection; larger g = less protection.
- Your four constraints become:
  - Higher λ ⇒ more protection: ∂g/∂λ < 0
  - Higher u ⇒ more protection: ∂g/∂u < 0
  - High u + low λ ⇒ less protection than high u + high λ
  - Low u + high λ ⇒ more protection than low u + low λ

### Problems with current design
- λ via sigmoid around τ compresses near 1.0 → weak dynamic range.
- g = 1 - u·λ (even clamped to [0,1]) violates “low u + high λ ⇒ more protection” because u·λ stays small when u is small.

### Meets-all-constraints options
- Recommended λ mapping (centered, tunable, bounded):
  - λ = clamp(1 + (curv - τ)/λ_scale, 0, λ_max)
- Gating functions that satisfy all four constraints:
  - Option C (additive protection, simple and independent):
    - g = 1 / (1 + a·λ + b·u), with a>0, b>0
    - Properties: strictly decreasing in u and λ; no negatives; λ can dominate protection even when u is small; low λ releases protection even when u is high.
  - Option D (weighted multiplicative, λ impact scales with u but never vanishes):
    - g = 1 / (1 + λ·(c + d·u)), with c>0, d≥0
    - Properties: strictly decreasing in λ; decreasing in u when λ>0; ensures λ still matters at low u via c; makes λ’s effect stronger as u increases via d.

### Quick sanity check (Option C with a=b=1; u∈{0.1, 1.0}, λ∈{0.2, 2.0})
- High u, high λ: g = 1/(1+2+1)=0.25 → more protection
- Low u, low λ: g = 1/(1+0.2+0.1)=0.77 → less protection
- High u, low λ: g = 1/(1+0.2+1)=0.45 → less protection than high u, high λ
- Low u, high λ: g = 1/(1+2+0.1)=0.32 → more protection than low u, low λ
All four requirements hold.

### Calibration guidance
- Balance a and b:
  - Increase a to ensure “low u + high λ ⇒ more protection” is strong enough.
  - Increase b to ensure “high u + low λ ⇒ less protection” is noticeable.
- Keep g∈[ε,1] by optional lower clamp (e.g., ε=0.05) to avoid total freezing.
- If you want comparability to standard UPGD at λ≈1, adjust learning rate globally so E[g(u,1)] matches E[1-u], or tune (a,b) by grid to match average step size.

### Practical defaults
- λ: τ at desired percentile (e.g., 80–90th), λ_scale sets sensitivity, λ_max=2.0.
- Start with Option C:
  - a = 1.0, b = 1.0; adjust a up if low-u/high-λ needs stronger protection; adjust b down if high-u/low-λ should release more.
- If you prefer u to amplify λ’s effect: Option D with c=0.5, d=0.5.

### Minimal implementation sketch
```python
# curvature -> lambda (centered at τ, bounded)
lambda_value = torch.clamp(1.0 + (curv - tau) / lambda_scale, min=0.0, max=lambda_max)

# Option C
g = 1.0 / (1.0 + a * lambda_value + b * u)

# Option D
g = 1.0 / (1.0 + lambda_value * (c + d * u))
```

### What to verify (fast checks)
- Monotonicity: finite-difference ∂g/∂u<0 and ∂g/∂λ<0 over your operating range.
- Quadrant ordering: {high u, high λ} has smallest g; {low u, low λ} largest; the mixed pairs rank as desired.
- Distribution sanity: g has usable spread (not collapsed), no negatives, bounded.

- In short:
  - Switch λ to a centered linear mapping with cap.
  - Use g = 1/(1 + a·λ + b·u) (Option C) or g = 1/(1 + λ·(c + d·u)) (Option D).
  - Both satisfy your four requirements by construction; Option C is simplest and most interpretable.