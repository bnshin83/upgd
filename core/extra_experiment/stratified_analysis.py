#!/usr/bin/env python3
"""
Stratified analysis: Partition samples by utility and curvature to find
where input-aware gating provides value.
"""

import numpy as np
import csv
import json
import sys

def load_curvature_data(csv_path):
    """Load curvature data from WandB export."""
    curvatures = []
    steps = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2 and row[1]:
                try:
                    steps.append(int(row[0]))
                    curvatures.append(float(row[1]))
                except ValueError:
                    continue

    return np.array(steps), np.array(curvatures)

def stratified_analysis(curvatures):
    """
    Partition curvatures into quantiles and analyze.

    This gives you targets for per-sample analysis in your training loop.
    """
    print("="*80)
    print("STRATIFIED CURVATURE ANALYSIS")
    print("="*80)
    print()

    # Define strata
    thresholds = {
        'P25': np.percentile(curvatures, 25),
        'P50': np.percentile(curvatures, 50),
        'P75': np.percentile(curvatures, 75),
        'P90': np.percentile(curvatures, 90),
        'P95': np.percentile(curvatures, 95),
        'P99': np.percentile(curvatures, 99),
    }

    print("Curvature Thresholds:")
    for name, val in thresholds.items():
        print(f"  {name}: {val:.6e}")
    print()

    # Create strata
    strata = {
        'Very Low':  curvatures < thresholds['P25'],
        'Low':       (curvatures >= thresholds['P25']) & (curvatures < thresholds['P50']),
        'Medium':    (curvatures >= thresholds['P50']) & (curvatures < thresholds['P75']),
        'High':      (curvatures >= thresholds['P75']) & (curvatures < thresholds['P90']),
        'Very High': (curvatures >= thresholds['P90']) & (curvatures < thresholds['P95']),
        'Extreme':   curvatures >= thresholds['P95'],
    }

    print("Strata Distribution:")
    print("-"*80)
    print(f"{'Stratum':<12} | {'Count':>8} | {'Percent':>8} | {'Mean Curv':>12} | {'Min Curv':>12} | {'Max Curv':>12}")
    print("-"*80)

    for name, mask in strata.items():
        count = np.sum(mask)
        pct = 100 * count / len(curvatures)
        mean_c = np.mean(curvatures[mask]) if count > 0 else 0
        min_c = np.min(curvatures[mask]) if count > 0 else 0
        max_c = np.max(curvatures[mask]) if count > 0 else 0

        print(f"{name:<12} | {count:8d} | {pct:7.2f}% | {mean_c:12.6e} | {min_c:12.6e} | {max_c:12.6e}")

    print()
    print("="*80)
    print("NEXT STEPS: Instrument your training loop")
    print("="*80)
    print()
    print("To prove that curvature improves utility scaling, you need to:")
    print()
    print("1. Track per-sample metrics during training:")
    print("   - Sample curvature: Tr(H_x²)")
    print("   - Active weights for this sample")
    print("   - Utility of active weights: u_i")
    print("   - Prediction accuracy on this sample")
    print("   - Weight updates: |Δw_i|")
    print()
    print("2. Stratify results by curvature (use thresholds above):")
    print("   For each stratum, compute:")
    print("   - Mean accuracy (UPGD vs Input-aware)")
    print("   - Mean utility of active weights")
    print("   - Mean update magnitude")
    print("   - Forgetting rate (if applicable)")
    print()
    print("3. Create 2D analysis (utility × curvature):")
    print("   Partition into quadrants:")
    print("   - Q1: Low utility (u<0.5), Low curvature (<P50)")
    print("   - Q2: Low utility (u<0.5), High curvature (>=P50)  ← KEY!")
    print("   - Q3: High utility (u>=0.5), Low curvature (<P50)")
    print("   - Q4: High utility (u>=0.5), High curvature (>=P50)")
    print()
    print("4. Statistical test:")
    print("   Compare Q2 performance:")
    print("   - If input-aware >> UPGD in Q2: Curvature adds unique value")
    print("   - If similar: Curvature is redundant")
    print()
    print("5. Check Q2 population:")
    print(f"   - If Q2 has <1% of samples: Too rare to matter")
    print(f"   - If Q2 has >5% of samples: Significant opportunity")
    print()

    return thresholds, strata

def main():
    if len(sys.argv) < 2:
        print("Usage: python stratified_analysis.py <curvature_csv_path>")
        print("\nExample:")
        print("  python stratified_analysis.py wandb_export_2025-10-02T09_54_55.718-04_00.csv")
        return

    csv_path = sys.argv[1]

    print(f"Loading curvature data from: {csv_path}")
    steps, curvatures = load_curvature_data(csv_path)
    print(f"Loaded {len(curvatures)} samples")
    print()

    thresholds, strata = stratified_analysis(curvatures)

    print("="*80)
    print("EXPECTED RESULTS IF HYPOTHESIS IS CORRECT")
    print("="*80)
    print()
    print("Hypothesis: Curvature improves utility scaling (beyond first-order)")
    print()
    print("Expected evidence:")
    print("  1. Q2 (Low u, High curv) has non-trivial population (>2%)")
    print("  2. Input-aware accuracy >> UPGD accuracy in Q2")
    print("  3. Q4 (High u, High curv) also benefits (better scaling)")
    print("  4. Correlation between utility and curvature is <0.7")
    print()
    print("If NOT observed:")
    print("  → Utility and curvature are highly correlated")
    print("  → First-order utility is sufficient")
    print("  → Input curvature is redundant")
    print()

if __name__ == "__main__":
    main()
