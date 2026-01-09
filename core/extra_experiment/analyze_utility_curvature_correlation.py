#!/usr/bin/env python3
"""
Analyze correlation between input curvature and weight utility.

This script helps determine if input curvature provides orthogonal
information to utility-based protection in UPGD.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def load_experiment_data(json_path):
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_correlation(utilities, curvatures):
    """
    Analyze correlation between utility and curvature.

    Args:
        utilities: Array of weight utilities at different timesteps
        curvatures: Array of input curvatures at different timesteps

    Returns:
        Dictionary with correlation metrics
    """
    # Pearson correlation
    pearson = np.corrcoef(utilities, curvatures)[0, 1]

    # Spearman rank correlation (for non-linear relationships)
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(utilities, curvatures)

    # Mutual information (for any dependency)
    from sklearn.metrics import mutual_info_score

    # Discretize for mutual information
    u_bins = np.digitize(utilities, bins=np.percentile(utilities, [25, 50, 75]))
    c_bins = np.digitize(curvatures, bins=np.percentile(curvatures, [25, 50, 75]))
    mi = mutual_info_score(u_bins, c_bins)

    return {
        'pearson': pearson,
        'spearman': spearman,
        'mutual_info': mi
    }

def quadrant_analysis(utilities, curvatures, accuracies=None):
    """
    Partition samples into 4 quadrants based on utility and curvature.

    Quadrants:
    1. Low utility, Low curvature
    2. Low utility, High curvature  ← Key quadrant!
    3. High utility, Low curvature
    4. High utility, High curvature
    """
    u_median = np.median(utilities)
    c_median = np.median(curvatures)

    q1 = (utilities < u_median) & (curvatures < c_median)
    q2 = (utilities < u_median) & (curvatures >= c_median)  # KEY
    q3 = (utilities >= u_median) & (curvatures < c_median)
    q4 = (utilities >= u_median) & (curvatures >= c_median)

    results = {}
    for i, (quad, name) in enumerate([
        (q1, "Low u, Low curv"),
        (q2, "Low u, High curv"),  # KEY
        (q3, "High u, Low curv"),
        (q4, "High u, High curv")
    ], 1):
        results[f"Q{i}"] = {
            'name': name,
            'count': np.sum(quad),
            'pct': 100 * np.sum(quad) / len(utilities),
            'mean_utility': np.mean(utilities[quad]) if np.sum(quad) > 0 else 0,
            'mean_curvature': np.mean(curvatures[quad]) if np.sum(quad) > 0 else 0,
        }

        if accuracies is not None and np.sum(quad) > 0:
            results[f"Q{i}"]['mean_accuracy'] = np.mean(accuracies[quad])

    return results

def plot_utility_curvature_scatter(utilities, curvatures, save_path):
    """Create scatter plot of utility vs curvature."""
    plt.figure(figsize=(10, 8))

    # 2D histogram (heatmap)
    plt.subplot(2, 2, 1)
    plt.hist2d(utilities, curvatures, bins=50, cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel('Utility')
    plt.ylabel('Input Curvature')
    plt.title('2D Histogram: Utility vs Curvature')

    # Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(utilities, curvatures, alpha=0.3, s=1)
    plt.xlabel('Utility')
    plt.ylabel('Input Curvature')
    plt.title('Scatter: Utility vs Curvature')

    # Quadrant boundaries
    u_median = np.median(utilities)
    c_median = np.median(curvatures)
    plt.axvline(u_median, color='r', linestyle='--', alpha=0.5)
    plt.axhline(c_median, color='r', linestyle='--', alpha=0.5)

    # Marginal distributions
    plt.subplot(2, 2, 3)
    plt.hist(utilities, bins=50, alpha=0.7)
    plt.xlabel('Utility')
    plt.ylabel('Count')
    plt.title('Utility Distribution')

    plt.subplot(2, 2, 4)
    plt.hist(curvatures, bins=50, alpha=0.7)
    plt.xlabel('Input Curvature')
    plt.ylabel('Count')
    plt.title('Curvature Distribution')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

def main():
    """
    Main analysis script.

    Usage:
        python analyze_utility_curvature_correlation.py <path_to_wandb_log>
    """
    if len(sys.argv) < 2:
        print("Usage: python analyze_utility_curvature_correlation.py <log_path>")
        print("\nNote: This script requires logged utility and curvature values.")
        print("If your logs don't contain this data, you'll need to modify")
        print("the training loop to track per-weight utilities and curvatures.")
        return

    print("="*80)
    print("UTILITY-CURVATURE CORRELATION ANALYSIS")
    print("="*80)
    print()

    print("This analysis determines if input curvature provides information")
    print("beyond what utility already captures.")
    print()

    print("Key metrics:")
    print("  - Pearson correlation: Linear relationship (-1 to 1)")
    print("  - Spearman correlation: Monotonic relationship (-1 to 1)")
    print("  - Mutual information: Any dependency (0 to ∞)")
    print()

    print("Quadrant analysis:")
    print("  Q1: Low utility, Low curvature   - Easy samples, unimportant weights")
    print("  Q2: Low utility, High curvature  - KEY: Rare/hard samples on dormant weights")
    print("  Q3: High utility, Low curvature  - Easy samples, important weights")
    print("  Q4: High utility, High curvature - Hard samples, important weights")
    print()

    print("="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print()

    print("High correlation (|r| > 0.7):")
    print("  → Curvature is redundant with utility")
    print("  → Standard UPGD already captures the information")
    print("  → Input-aware optimization not beneficial")
    print()

    print("Low correlation (|r| < 0.3):")
    print("  → Curvature provides orthogonal information")
    print("  → Input-aware optimization may help")
    print("  → Check Q2 quadrant for evidence")
    print()

    print("Q2 quadrant (Low u, High curv) has many samples:")
    print("  → Rare/hard samples exist on low-utility weights")
    print("  → These are NOT protected by standard UPGD")
    print("  → Input-aware gating provides unique value")
    print()

    print("Q2 quadrant is empty or tiny:")
    print("  → High curvature always correlates with high utility")
    print("  → Confirms redundancy hypothesis")
    print("  → Input-aware optimization redundant")
    print()

    print("="*80)
    print()
    print("NOTE: This script is a template. You'll need to:")
    print("  1. Modify your training loop to log per-weight utilities")
    print("  2. Log input curvatures at the same timesteps")
    print("  3. Pass the log file path to this script")
    print()

if __name__ == "__main__":
    main()
