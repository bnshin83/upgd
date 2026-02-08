"""
ComparatorAgent: Compare methods across multiple metrics and dimensions.

Generates comprehensive comparison tables including:
- Summary statistics for each method
- Win/tie/loss matrices
- Pairwise performance comparisons
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from ..base import BaseAgent


class ComparatorAgent(BaseAgent):
    """
    Agent for comparing methods across multiple dimensions.

    Generates:
    - Summary tables with mean ± std for each method × metric
    - Win/tie/loss matrices showing pairwise comparisons
    - Performance profiles
    """

    def __init__(self, name: str = 'comparator', config: Optional[Dict[str, Any]] = None):
        """
        Initialize ComparatorAgent.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.metric_config = self.get_config('metrics', {})

    def execute(
        self,
        data: pd.DataFrame,
        methods: List[str],
        metrics: List[str],
        aggregation: str = 'final'  # 'final', 'mean', 'median'
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive comparison of methods across metrics.

        Args:
            data: DataFrame with columns: method, seed, metric, value
                 (or aggregated data with method, metric, value_mean)
            methods: List of methods to compare
            metrics: List of metrics to compare on
            aggregation: How to aggregate across time/tasks
                        'final': use final value per seed
                        'mean': average across all time points
                        'median': median across all time points

        Returns:
            Dictionary with:
            - 'summary': DataFrame with mean±std for each method×metric
            - 'win_matrix': Win/tie/loss counts for pairwise comparisons
            - 'best_method': Best method per metric

        Example:
            ```python
            comparison = comparator.execute(
                data=data,
                methods=['S&P', 'UPGD (Full)', 'UPGD (Output Only)'],
                metrics=['accuracy', 'plasticity']
            )
            print(comparison['summary'])
            ```
        """
        self.logger.info(f"Comparing {len(methods)} methods on {len(metrics)} metrics")

        # Generate summary table
        summary = self._generate_summary(data, methods, metrics, aggregation)

        # Generate win/tie/loss matrix
        win_matrix = self._generate_win_matrix(data, methods, metrics, aggregation)

        # Identify best method per metric
        best_methods = self._identify_best_methods(summary, metrics)

        result = {
            'summary': summary,
            'win_matrix': win_matrix,
            'best_method': best_methods
        }

        return result

    def _generate_summary(
        self,
        data: pd.DataFrame,
        methods: List[str],
        metrics: List[str],
        aggregation: str
    ) -> pd.DataFrame:
        """Generate summary table with mean±std for each method×metric."""
        summary_rows = []

        for method in methods:
            method_data = data[data['method'] == method]

            for metric in metrics:
                metric_data = method_data[method_data['metric'] == metric]

                if metric_data.empty:
                    continue

                # Determine value column
                val_col = 'value_mean' if 'value_mean' in metric_data.columns else 'value'

                # Aggregate across time/tasks if needed
                if aggregation == 'final' and 'seed' in metric_data.columns:
                    # Get final value per seed
                    if 'step' in metric_data.columns:
                        # Group by seed and take last step
                        final_values = metric_data.groupby('seed').apply(
                            lambda x: x.loc[x['step'].idxmax(), val_col]
                        ).values
                    elif 'task' in metric_data.columns:
                        # Group by seed and take last task
                        final_values = metric_data.groupby('seed').apply(
                            lambda x: x.loc[x['task'].idxmax(), val_col]
                        ).values
                    else:
                        # Already aggregated
                        final_values = metric_data[val_col].values
                elif aggregation == 'mean':
                    final_values = [metric_data[val_col].mean()]
                elif aggregation == 'median':
                    final_values = [metric_data[val_col].median()]
                else:
                    final_values = metric_data[val_col].values

                mean_val = np.mean(final_values)
                std_val = np.std(final_values, ddof=1) if len(final_values) > 1 else 0.0

                summary_rows.append({
                    'method': method,
                    'metric': metric,
                    'mean': mean_val,
                    'std': std_val,
                    'n': len(final_values)
                })

        summary = pd.DataFrame(summary_rows)

        # Format as mean ± std string
        if not summary.empty:
            summary['mean_std'] = summary.apply(
                lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}",
                axis=1
            )

        return summary

    def _generate_win_matrix(
        self,
        data: pd.DataFrame,
        methods: List[str],
        metrics: List[str],
        aggregation: str
    ) -> pd.DataFrame:
        """
        Generate win/tie/loss matrix for pairwise method comparisons.

        For each pair of methods, counts on how many metrics method A is better than B.
        """
        # First get summary to determine which method wins on each metric
        summary = self._generate_summary(data, methods, metrics, aggregation)

        # Initialize win matrix
        win_counts = pd.DataFrame(0, index=methods, columns=methods)

        for metric in metrics:
            metric_summary = summary[summary['metric'] == metric]

            if metric_summary.empty:
                continue

            # Determine if higher or lower is better
            higher_is_better = self.metric_config.get('higher_is_better', {}).get(metric, True)

            # Rank methods for this metric
            if higher_is_better:
                metric_summary = metric_summary.sort_values('mean', ascending=False)
            else:
                metric_summary = metric_summary.sort_values('mean', ascending=True)

            ranked_methods = metric_summary['method'].tolist()

            # Update win counts
            for i, method_a in enumerate(ranked_methods):
                for method_b in ranked_methods[i+1:]:
                    win_counts.loc[method_a, method_b] += 1

        return win_counts

    def _identify_best_methods(
        self,
        summary: pd.DataFrame,
        metrics: List[str]
    ) -> pd.DataFrame:
        """Identify the best method for each metric."""
        best_rows = []

        for metric in metrics:
            metric_summary = summary[summary['metric'] == metric]

            if metric_summary.empty:
                continue

            # Determine if higher or lower is better
            higher_is_better = self.metric_config.get('higher_is_better', {}).get(metric, True)

            if higher_is_better:
                best_row = metric_summary.loc[metric_summary['mean'].idxmax()]
            else:
                best_row = metric_summary.loc[metric_summary['mean'].idxmin()]

            best_rows.append({
                'metric': metric,
                'best_method': best_row['method'],
                'value': best_row['mean'],
                'std': best_row['std']
            })

        return pd.DataFrame(best_rows)

    def compare_two_methods(
        self,
        data: pd.DataFrame,
        method_a: str,
        method_b: str,
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Detailed comparison of two specific methods.

        Args:
            data: DataFrame with method data
            method_a: First method
            method_b: Second method
            metrics: Metrics to compare on

        Returns:
            DataFrame showing side-by-side comparison
        """
        comparison_rows = []

        for metric in metrics:
            # Get data for both methods
            data_a = data[(data['method'] == method_a) & (data['metric'] == metric)]
            data_b = data[(data['method'] == method_b) & (data['metric'] == metric)]

            if data_a.empty or data_b.empty:
                continue

            val_col = 'value_mean' if 'value_mean' in data_a.columns else 'value'

            mean_a = data_a[val_col].mean()
            mean_b = data_b[val_col].mean()

            std_a = data_a[val_col].std()
            std_b = data_b[val_col].std()

            # Compute difference
            diff = mean_b - mean_a
            pct_change = (diff / mean_a * 100) if mean_a != 0 else np.nan

            # Determine winner
            higher_is_better = self.metric_config.get('higher_is_better', {}).get(metric, True)
            if higher_is_better:
                winner = method_b if mean_b > mean_a else method_a
            else:
                winner = method_b if mean_b < mean_a else method_a

            comparison_rows.append({
                'metric': metric,
                f'{method_a}_mean': mean_a,
                f'{method_a}_std': std_a,
                f'{method_b}_mean': mean_b,
                f'{method_b}_std': std_b,
                'difference': diff,
                'pct_change': pct_change,
                'winner': winner
            })

        return pd.DataFrame(comparison_rows)
