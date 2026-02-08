"""
AggregatorAgent: Aggregate data across different dimensions.

Provides flexible aggregation of experiment data including:
- Per-task window aggregation (convert 1M steps to ~400 tasks)
- Cross-seed aggregation (mean, std, CI)
- Custom groupby aggregations
"""

from typing import Optional, Dict, Any, List, Union, Callable
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from ..base import BaseAgent


class AggregatorAgent(BaseAgent):
    """
    Agent for aggregating experiment data across different dimensions.

    Supports:
    - Per-task window aggregation (e.g., 2500 steps per task)
    - Cross-seed aggregation with statistics
    - Flexible groupby operations
    - Confidence interval computation
    """

    def __init__(self, name: str = 'aggregator', config: Optional[Dict[str, Any]] = None):
        """
        Initialize AggregatorAgent.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.datasets_config = self.get_config('data.datasets', {})
        self.confidence_level = self.get_config('stats.confidence_level', 0.95)

    def execute(
        self,
        data: pd.DataFrame,
        groupby: List[str],
        agg_functions: Dict[str, Union[str, List[str], Callable]],
        add_ci: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate data with specified functions.

        Args:
            data: Input DataFrame
            groupby: List of columns to group by
            agg_functions: Dictionary mapping column names to aggregation functions
                          e.g., {'value': ['mean', 'std', 'sem']}
            add_ci: Whether to add confidence interval columns

        Returns:
            Aggregated DataFrame

        Example:
            ```python
            aggregated = aggregator.execute(
                data=data,
                groupby=['method', 'step'],
                agg_functions={'value': ['mean', 'std', 'sem']},
                add_ci=True
            )
            ```
        """
        self.logger.info(f"Aggregating data by {groupby}")

        if data.empty:
            self.logger.warning("Empty input data")
            return pd.DataFrame()

        # Perform aggregation
        grouped = data.groupby(groupby).agg(agg_functions)

        # Flatten column names if multi-level
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

        # Reset index to make groupby columns regular columns
        result = grouped.reset_index()

        # Add confidence intervals if requested
        if add_ci and 'value_mean' in result.columns and 'value_sem' in result.columns:
            result = self._add_confidence_intervals(result)

        self.logger.info(f"Aggregated to {len(result)} rows")
        return result

    def aggregate_per_task(
        self,
        data: pd.DataFrame,
        dataset: Optional[str] = None,
        steps_per_task: Optional[int] = None,
        add_ci: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate per-step data into per-task windows.

        Converts high-resolution per-step data (e.g., 1M steps) into
        per-task aggregates (e.g., ~400 tasks with 2500 steps each).

        Args:
            data: DataFrame with 'step' column
            dataset: Dataset name (to get default steps_per_task from config)
            steps_per_task: Steps per task window. If None, tries to infer from dataset.
            add_ci: Whether to add confidence intervals

        Returns:
            DataFrame with 'task' column instead of 'step'

        Example:
            ```python
            # Convert 1M steps to ~400 tasks (2500 steps each)
            task_data = aggregator.aggregate_per_task(
                data=step_data,
                dataset='mini_imagenet'
            )
            ```
        """
        self.logger.info("Aggregating per-step data to per-task windows")

        if 'step' not in data.columns:
            raise ValueError("Data must have 'step' column for per-task aggregation")

        # Determine steps_per_task
        if steps_per_task is None:
            if dataset and dataset in self.datasets_config:
                steps_per_task = self.datasets_config[dataset].get('steps_per_task', 2500)
            else:
                steps_per_task = 2500  # Default for label-permuted datasets
                self.logger.warning(
                    f"steps_per_task not specified, using default: {steps_per_task}"
                )

        # Create task column
        data = data.copy()
        data['task'] = data['step'] // steps_per_task

        # Group by all columns except 'step' and 'value', plus the new 'task' column
        groupby_cols = [col for col in data.columns if col not in ['step', 'value']]
        if 'task' not in groupby_cols:
            groupby_cols.append('task')

        # Aggregate
        agg_functions = {'value': ['mean', 'std', 'sem', 'count']}
        result = self.execute(
            data=data,
            groupby=groupby_cols,
            agg_functions=agg_functions,
            add_ci=add_ci
        )

        # Rename count column
        if 'value_count' in result.columns:
            result = result.rename(columns={'value_count': 'n_steps'})

        self.logger.info(
            f"Converted {len(data)} steps to {len(result)} task windows "
            f"({steps_per_task} steps/task)"
        )

        return result

    def aggregate_across_seeds(
        self,
        data: pd.DataFrame,
        groupby: Optional[List[str]] = None,
        add_ci: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate across seeds (compute mean, std, CI across random seeds).

        Args:
            data: DataFrame with 'seed' column
            groupby: Additional columns to group by (e.g., ['method', 'step'])
                    If None, groups by all columns except 'seed' and 'value'
            add_ci: Whether to add confidence intervals

        Returns:
            DataFrame with statistics across seeds

        Example:
            ```python
            # Average across 3 seeds
            avg_data = aggregator.aggregate_across_seeds(
                data=multi_seed_data,
                groupby=['method', 'step', 'metric']
            )
            ```
        """
        self.logger.info("Aggregating across seeds")

        if 'seed' not in data.columns:
            raise ValueError("Data must have 'seed' column for cross-seed aggregation")

        # Determine groupby columns
        if groupby is None:
            groupby = [col for col in data.columns if col not in ['seed', 'value']]

        # Aggregate
        agg_functions = {'value': ['mean', 'std', 'sem', 'count']}
        result = self.execute(
            data=data,
            groupby=groupby,
            agg_functions=agg_functions,
            add_ci=add_ci
        )

        # Rename count column
        if 'value_count' in result.columns:
            result = result.rename(columns={'value_count': 'n_seeds'})

        self.logger.info(f"Aggregated across seeds: {result['n_seeds'].iloc[0]} seeds")

        return result

    def _add_confidence_intervals(
        self,
        data: pd.DataFrame,
        confidence_level: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Add confidence interval columns to aggregated data.

        Requires 'value_mean', 'value_sem', and 'value_count' columns.

        Args:
            data: Aggregated DataFrame
            confidence_level: Confidence level (default from config)

        Returns:
            DataFrame with 'value_ci_lower' and 'value_ci_upper' columns
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        required_cols = ['value_mean', 'value_sem', 'value_count']
        if not all(col in data.columns for col in required_cols):
            self.logger.warning(
                f"Cannot compute CI: missing required columns {required_cols}"
            )
            return data

        data = data.copy()

        # Compute t-value for confidence interval
        alpha = 1 - confidence_level
        df = data['value_count'] - 1  # Degrees of freedom

        # Handle edge case where count is 1 (no variance)
        t_values = np.where(
            df > 0,
            scipy_stats.t.ppf(1 - alpha / 2, df),
            np.nan
        )

        # Compute margin of error
        margin = t_values * data['value_sem']

        # Add CI columns
        data['value_ci_lower'] = data['value_mean'] - margin
        data['value_ci_upper'] = data['value_mean'] + margin

        return data

    def compute_summary_statistics(
        self,
        data: pd.DataFrame,
        groupby: List[str],
        value_col: str = 'value'
    ) -> pd.DataFrame:
        """
        Compute comprehensive summary statistics.

        Args:
            data: Input DataFrame
            groupby: Columns to group by
            value_col: Column to compute statistics on

        Returns:
            DataFrame with extensive statistics (mean, std, min, max, quartiles, etc.)
        """
        self.logger.info(f"Computing summary statistics by {groupby}")

        agg_functions = {
            value_col: [
                'mean', 'std', 'sem', 'min', 'max',
                'count',
                ('q25', lambda x: x.quantile(0.25)),
                ('q50', lambda x: x.quantile(0.50)),
                ('q75', lambda x: x.quantile(0.75))
            ]
        }

        result = data.groupby(groupby).agg(agg_functions)

        # Flatten columns
        result.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                         for col in result.columns.values]

        return result.reset_index()
