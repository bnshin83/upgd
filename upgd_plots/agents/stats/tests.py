"""
StatisticalTestAgent: Perform statistical hypothesis tests comparing methods.

Supports various statistical tests including t-tests, Wilcoxon signed-rank test,
and ANOVA, with multiple comparison correction methods.
"""

from typing import List, Optional, Dict, Any, Literal
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t as t_dist

from ..base import BaseAgent


class StatisticalTestAgent(BaseAgent):
    """
    Agent for performing statistical significance tests between methods.

    Supports:
    - Paired/unpaired t-tests
    - Wilcoxon signed-rank test (non-parametric)
    - ANOVA (one-way)
    - Multiple comparison corrections (Bonferroni, Holm, FDR)
    - Effect size calculations (Cohen's d, Cliff's delta)
    """

    def __init__(self, name: str = 'stats_tester', config: Optional[Dict[str, Any]] = None):
        """
        Initialize StatisticalTestAgent.

        Args:
            name: Agent name
            config: Configuration dictionary with 'stats' section
        """
        super().__init__(name, config)
        self.default_alpha = self.get_config('stats.default_alpha', 0.05)
        self.default_correction = self.get_config('stats.default_correction', 'holm')
        self.default_test = self.get_config('stats.default_test', 'wilcoxon')

    def execute(
        self,
        data: pd.DataFrame,
        baseline_method: str,
        comparison_methods: List[str],
        metric: str,
        test_type: Literal['ttest', 'ttest_paired', 'wilcoxon', 'anova'] = 'wilcoxon',
        correction: Optional[Literal['bonferroni', 'holm', 'fdr']] = 'holm',
        alpha: float = 0.05,
        effect_size_type: Literal['cohen_d', 'cliff_delta'] = 'cohen_d'
    ) -> pd.DataFrame:
        """
        Perform statistical tests comparing methods against a baseline.

        Args:
            data: DataFrame with columns: method, seed, metric, value
            baseline_method: Name of baseline method for comparison
            comparison_methods: List of methods to compare against baseline
            metric: Metric name to test
            test_type: Statistical test to use
            correction: Multiple comparison correction method (None for no correction)
            alpha: Significance level
            effect_size_type: Type of effect size to compute

        Returns:
            DataFrame with columns:
                - method_a: Baseline method
                - method_b: Comparison method
                - metric: Metric tested
                - test_type: Test used
                - statistic: Test statistic
                - p_value: Raw p-value
                - p_value_corrected: Corrected p-value (if correction applied)
                - significant: Whether result is significant at alpha level
                - effect_size: Effect size magnitude
                - effect_size_type: Type of effect size
        """
        self.logger.info(
            f"Running {test_type} tests: {baseline_method} vs {comparison_methods} "
            f"on metric='{metric}'"
        )

        # Filter data for the specific metric
        metric_data = data[data['metric'] == metric].copy()

        if metric_data.empty:
            self.logger.error(f"No data found for metric '{metric}'")
            return pd.DataFrame()

        # Get baseline data
        baseline_data = metric_data[metric_data['method'] == baseline_method]

        if baseline_data.empty:
            self.logger.error(f"No data found for baseline method '{baseline_method}'")
            return pd.DataFrame()

        results = []

        for comp_method in comparison_methods:
            comp_data = metric_data[metric_data['method'] == comp_method]

            if comp_data.empty:
                self.logger.warning(f"No data found for method '{comp_method}'. Skipping.")
                continue

            # Perform test
            test_result = self._perform_test(
                baseline_data=baseline_data['value'].values,
                comparison_data=comp_data['value'].values,
                test_type=test_type
            )

            # Compute effect size
            effect_size = self._compute_effect_size(
                baseline_data['value'].values,
                comp_data['value'].values,
                effect_size_type=effect_size_type
            )

            results.append({
                'method_a': baseline_method,
                'method_b': comp_method,
                'metric': metric,
                'test_type': test_type,
                'statistic': test_result['statistic'],
                'p_value': test_result['p_value'],
                'effect_size': effect_size,
                'effect_size_type': effect_size_type,
            })

        if not results:
            self.logger.error("No test results generated")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Apply multiple comparison correction
        if correction:
            results_df = self._apply_correction(results_df, correction=correction, alpha=alpha)
        else:
            results_df['p_value_corrected'] = results_df['p_value']

        # Mark significant results
        results_df['significant'] = results_df['p_value_corrected'] < alpha

        self.logger.info(
            f"Completed {len(results_df)} tests, "
            f"{results_df['significant'].sum()} significant at alpha={alpha}"
        )

        return results_df

    def _perform_test(
        self,
        baseline_data: np.ndarray,
        comparison_data: np.ndarray,
        test_type: str
    ) -> Dict[str, float]:
        """Perform a single statistical test."""
        if test_type == 'ttest':
            # Independent samples t-test (unpaired)
            statistic, p_value = stats.ttest_ind(baseline_data, comparison_data)

        elif test_type == 'ttest_paired':
            # Paired t-test (requires same length)
            min_len = min(len(baseline_data), len(comparison_data))
            statistic, p_value = stats.ttest_rel(
                baseline_data[:min_len],
                comparison_data[:min_len]
            )

        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric paired test)
            min_len = min(len(baseline_data), len(comparison_data))
            statistic, p_value = stats.wilcoxon(
                baseline_data[:min_len],
                comparison_data[:min_len],
                alternative='two-sided'
            )

        elif test_type == 'anova':
            # One-way ANOVA
            statistic, p_value = stats.f_oneway(baseline_data, comparison_data)

        else:
            raise ValueError(f"Unknown test type: {test_type}")

        return {'statistic': float(statistic), 'p_value': float(p_value)}

    def _compute_effect_size(
        self,
        baseline_data: np.ndarray,
        comparison_data: np.ndarray,
        effect_size_type: str
    ) -> float:
        """Compute effect size between two samples."""
        if effect_size_type == 'cohen_d':
            # Cohen's d: standardized mean difference
            mean_diff = np.mean(comparison_data) - np.mean(baseline_data)
            pooled_std = np.sqrt(
                (np.var(baseline_data, ddof=1) + np.var(comparison_data, ddof=1)) / 2
            )
            return mean_diff / pooled_std if pooled_std > 0 else 0.0

        elif effect_size_type == 'cliff_delta':
            # Cliff's delta: non-parametric effect size
            n1, n2 = len(baseline_data), len(comparison_data)
            greater = sum((comparison_data[i] > baseline_data[j])
                         for i in range(n2) for j in range(n1))
            less = sum((comparison_data[i] < baseline_data[j])
                      for i in range(n2) for j in range(n1))
            return (greater - less) / (n1 * n2) if (n1 * n2) > 0 else 0.0

        else:
            raise ValueError(f"Unknown effect size type: {effect_size_type}")

    def _apply_correction(
        self,
        results_df: pd.DataFrame,
        correction: str,
        alpha: float
    ) -> pd.DataFrame:
        """Apply multiple comparison correction to p-values."""
        p_values = results_df['p_value'].values
        n_tests = len(p_values)

        if correction == 'bonferroni':
            # Bonferroni correction: multiply by number of tests
            corrected = np.minimum(p_values * n_tests, 1.0)

        elif correction == 'holm':
            # Holm-Bonferroni correction (sequentially rejective)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected = np.empty(n_tests)

            for i, p in enumerate(sorted_p):
                corrected[sorted_indices[i]] = min(p * (n_tests - i), 1.0)

            # Ensure monotonicity
            for i in range(1, n_tests):
                if corrected[sorted_indices[i]] < corrected[sorted_indices[i-1]]:
                    corrected[sorted_indices[i]] = corrected[sorted_indices[i-1]]

        elif correction == 'fdr':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected = np.empty(n_tests)

            for i in range(n_tests):
                corrected[sorted_indices[i]] = min(sorted_p[i] * n_tests / (i + 1), 1.0)

            # Ensure monotonicity (from high to low)
            for i in range(n_tests - 2, -1, -1):
                if corrected[sorted_indices[i]] > corrected[sorted_indices[i+1]]:
                    corrected[sorted_indices[i]] = corrected[sorted_indices[i+1]]

        else:
            raise ValueError(f"Unknown correction method: {correction}")

        results_df['p_value_corrected'] = corrected
        return results_df

    def compute_confidence_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95
    ) -> tuple:
        """
        Compute confidence interval for mean.

        Args:
            data: Data array
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)  # Standard error of mean
        margin = se * t_dist.ppf((1 + confidence_level) / 2, n - 1)

        return (mean - margin, mean + margin)
