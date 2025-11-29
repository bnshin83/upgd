#!/usr/bin/env python3
"""
Analyze curvature distribution from WandB export
Validates hypotheses from note_gating.md
"""

import numpy as np
import csv

# Read CSV
csv_path = '/scratch/gautschi/shin283/upgd/wandb_export_2025-10-02T09_54_55.718-04_00.csv'

steps = []
curvatures = []

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    curvature_col = header[1]

    for row in reader:
        if len(row) >= 2 and row[1]:  # Has curvature data
            try:
                steps.append(int(row[0]))
                curvatures.append(float(row[1]))
            except (ValueError, IndexError):
                continue

curvatures = np.array(curvatures)
steps = np.array(steps)

print("=" * 80)
print("CURVATURE DISTRIBUTION ANALYSIS")
print("=" * 80)
print(f"\nDataset: {csv_path}")
print(f"Column: {curvature_col}")
print(f"Total samples: {len(curvatures):,}")

# Basic statistics
print("\n" + "-" * 80)
print("BASIC STATISTICS")
print("-" * 80)
print(f"Mean:       {np.mean(curvatures):.6e}")
print(f"Median:     {np.median(curvatures):.6e}")
print(f"Std Dev:    {np.std(curvatures):.6e}")
print(f"Min:        {np.min(curvatures):.6e}")
print(f"Max:        {np.max(curvatures):.6e}")
print(f"Range:      {np.max(curvatures) - np.min(curvatures):.6e}")

# Percentiles
print("\n" + "-" * 80)
print("PERCENTILE ANALYSIS")
print("-" * 80)
percentiles = [10, 25, 50, 75, 80, 85, 90, 95, 99, 99.5, 99.9]
for p in percentiles:
    val = np.percentile(curvatures, p)
    print(f"P{p:5.1f}: {val:.6e}")

# Threshold analysis
print("\n" + "-" * 80)
print("THRESHOLD ANALYSIS (τ = 0.01)")
print("-" * 80)
tau = 0.01
above_tau = curvatures > tau
n_above = np.sum(above_tau)
pct_above = 100 * n_above / len(curvatures)

print(f"Threshold τ:              {tau}")
print(f"Samples above τ:          {n_above:,} ({pct_above:.2f}%)")
print(f"Samples below τ:          {len(curvatures) - n_above:,} ({100-pct_above:.2f}%)")
print(f"Mean (above τ):           {np.mean(curvatures[above_tau]):.6e}")
print(f"Mean (below τ):           {np.mean(curvatures[~above_tau]):.6e}")
print(f"Max (above τ):            {np.max(curvatures[above_tau]):.6e}")

# Percentile position of tau
tau_percentile = 100 * np.sum(curvatures < tau) / len(curvatures)
print(f"\nτ = {tau} is at the {tau_percentile:.1f}th percentile")

# Lambda distribution with different tau values
print("\n" + "-" * 80)
print("LAMBDA ACTIVATION ANALYSIS")
print("-" * 80)

def compute_lambda(curvature, tau, lambda_max, lambda_scale=0.1):
    """Compute λ(x) = λ_max · sigmoid((Tr(H_x²) - τ) / λ_scale)"""
    return lambda_max / (1 + np.exp(-(curvature - tau) / lambda_scale))

lambda_max = 2.0
lambda_scale = 0.1

lambdas = compute_lambda(curvatures, tau, lambda_max, lambda_scale)

print(f"Configuration: λ_max={lambda_max}, τ={tau}, λ_scale={lambda_scale}")
print(f"\nLambda statistics:")
print(f"  Mean λ:     {np.mean(lambdas):.4f}")
print(f"  Median λ:   {np.median(lambdas):.4f}")
print(f"  Max λ:      {np.max(lambdas):.4f}")
print(f"  Min λ:      {np.min(lambdas):.4f}")

# Lambda thresholds
print(f"\nLambda activation rates:")
for lambda_thresh in [0.1, 0.5, 1.0, 1.5, 1.8]:
    n_active = np.sum(lambdas > lambda_thresh)
    pct_active = 100 * n_active / len(lambdas)
    print(f"  λ > {lambda_thresh:3.1f}: {n_active:6,} samples ({pct_active:5.2f}%)")

# High lambda samples
print(f"\nHigh protection samples (λ > 1.5):")
high_lambda = lambdas > 1.5
print(f"  Count: {np.sum(high_lambda):,} ({100*np.sum(high_lambda)/len(lambdas):.2f}%)")
print(f"  Mean curvature: {np.mean(curvatures[high_lambda]):.6e}")

# Protection gating analysis
print("\n" + "-" * 80)
print("PROTECTION GATING ANALYSIS")
print("-" * 80)

# Simulate different utility values
utilities = [0.2, 0.5, 0.8]
print(f"Gating term: (1 - u·λ(x))")
print(f"\nFor different utility levels:")

for u in utilities:
    gating = 1 - u * lambdas
    n_negative = np.sum(gating < 0)
    pct_negative = 100 * n_negative / len(gating)

    print(f"\n  Utility u = {u:.1f}:")
    print(f"    Mean gating:        {np.mean(gating):.4f}")
    print(f"    Median gating:      {np.median(gating):.4f}")
    print(f"    Min gating:         {np.min(gating):.4f}")
    print(f"    Negative gating:    {n_negative:,} ({pct_negative:.2f}%)")
    print(f"    Full plasticity:    {np.sum(gating > 0.95):,} ({100*np.sum(gating > 0.95)/len(gating):.2f}%)")

# Distribution shape analysis
print("\n" + "-" * 80)
print("DISTRIBUTION SHAPE")
print("-" * 80)
skewness = ((curvatures - np.mean(curvatures))**3).mean() / (np.std(curvatures)**3)
kurtosis = ((curvatures - np.mean(curvatures))**4).mean() / (np.std(curvatures)**4) - 3

print(f"Skewness:  {skewness:.2f} (>0 = right-skewed/long tail)")
print(f"Kurtosis:  {kurtosis:.2f} (>0 = heavy tails)")
print(f"\nDistribution type: ", end="")
if skewness > 2:
    print("HIGHLY RIGHT-SKEWED (long tail)")
else:
    print("Moderately right-skewed")

# Key finding validation
print("\n" + "=" * 80)
print("HYPOTHESIS VALIDATION")
print("=" * 80)

print(f"\n✓ Hypothesis 1: Long-tailed distribution")
print(f"  Mean/Median ratio: {np.mean(curvatures)/np.median(curvatures):.2f}x")
print(f"  Max/Mean ratio:    {np.max(curvatures)/np.mean(curvatures):.1f}x")
print(f"  Status: {'CONFIRMED' if np.max(curvatures)/np.mean(curvatures) > 100 else 'REJECTED'}")

print(f"\n✓ Hypothesis 2: τ=0.01 targets rare samples")
print(f"  Percentile of τ: {tau_percentile:.1f}th")
print(f"  Samples above τ: {pct_above:.2f}%")
print(f"  Status: {'CONFIRMED' if 75 <= tau_percentile <= 95 else 'NEEDS REVIEW'}")

print(f"\n✓ Hypothesis 3: Most samples have λ ≈ 0 (full plasticity)")
low_lambda = np.sum(lambdas < 0.1)
pct_low_lambda = 100 * low_lambda / len(lambdas)
print(f"  Samples with λ < 0.1: {low_lambda:,} ({pct_low_lambda:.2f}%)")
print(f"  Status: {'CONFIRMED' if pct_low_lambda > 70 else 'NEEDS REVIEW'}")

print(f"\n✓ Hypothesis 4: Negative gating occurs with λ_max=2.0")
# For high-utility weights (u=0.8)
u_high = 0.8
high_u_gating = 1 - u_high * lambdas
n_negative_high_u = np.sum(high_u_gating < 0)
pct_negative_high_u = 100 * n_negative_high_u / len(high_u_gating)
print(f"  Negative gating (u=0.8): {n_negative_high_u:,} ({pct_negative_high_u:.2f}%)")
print(f"  Status: {'CONFIRMED' if pct_negative_high_u > 1 else 'MINIMAL'}")

# Task boundary analysis (every 5000 steps for Input-Permuted MNIST)
print("\n" + "=" * 80)
print("TASK BOUNDARY ANALYSIS")
print("=" * 80)

task_boundaries = []
for boundary in range(5000, int(steps[-1]) + 1, 5000):
    # Find samples near boundary (±500 steps)
    near_boundary = (steps >= boundary - 500) & (steps <= boundary + 500)
    if np.sum(near_boundary) > 0:
        mean_curv_boundary = np.mean(curvatures[near_boundary])
        task_boundaries.append((boundary, mean_curv_boundary, np.sum(near_boundary)))

print(f"\nTask transitions (every 5000 steps):")
print(f"Overall mean curvature: {np.mean(curvatures):.6e}")
print(f"\nCurvature near task boundaries (±500 steps):")
for boundary, mean_curv, n_samples in task_boundaries[:10]:  # First 10
    ratio = mean_curv / np.mean(curvatures)
    print(f"  Step {boundary:6d}: {mean_curv:.6e} ({ratio:.2f}x overall, n={n_samples})")

if len(task_boundaries) > 0:
    boundary_curvs = [bc[1] for bc in task_boundaries]
    mean_boundary_curv = np.mean(boundary_curvs)
    ratio = mean_boundary_curv / np.mean(curvatures)
    print(f"\n  Average at boundaries: {mean_boundary_curv:.6e} ({ratio:.2f}x overall)")
    print(f"  Status: {'SPIKE DETECTED' if ratio > 1.5 else 'NO CLEAR SPIKE'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Save summary statistics to file
output_file = '/scratch/gautschi/shin283/upgd/curvature_distribution_analysis.txt'
with open(output_file, 'w') as f:
    f.write("CURVATURE DISTRIBUTION ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset: {csv_path}\n")
    f.write(f"Total samples: {len(curvatures):,}\n\n")
    f.write(f"Mean:       {np.mean(curvatures):.6e}\n")
    f.write(f"Median:     {np.median(curvatures):.6e}\n")
    f.write(f"Max:        {np.max(curvatures):.6e}\n\n")
    f.write(f"τ = 0.01 percentile: {tau_percentile:.1f}%\n")
    f.write(f"Samples above τ: {pct_above:.2f}%\n\n")
    f.write(f"λ_max = 2.0 configuration:\n")
    f.write(f"  Mean λ: {np.mean(lambdas):.4f}\n")
    f.write(f"  Samples with λ > 1.5: {100*np.sum(lambdas > 1.5)/len(lambdas):.2f}%\n")
    f.write(f"  Negative gating (u=0.8): {pct_negative_high_u:.2f}%\n")

print(f"\nSummary saved to: {output_file}")
