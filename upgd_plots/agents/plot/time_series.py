"""
TimeSeriesPlotAgent: Plot metrics over time with confidence bands.

Creates publication-quality time series plots showing training curves
with mean and confidence intervals across multiple seeds.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .base_plotter import PlotGeneratorAgent


class TimeSeriesPlotAgent(PlotGeneratorAgent):
    """
    Agent for plotting time series data (training curves).

    Features:
    - Mean trajectory with confidence bands
    - Multiple methods on same axes
    - Per-task or per-step x-axis
    - Customizable styling per method
    """

    def __init__(self, name: str = 'time_series_plotter', config: Optional[Dict[str, Any]] = None):
        """
        Initialize TimeSeriesPlotAgent.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)

    def execute(
        self,
        data: pd.DataFrame,
        methods: List[str],
        metric: str,
        x_axis: str = 'step',
        confidence_level: float = 0.95,
        show_bands: bool = True,
        subsample: Optional[int] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        filename: Optional[str] = None,
        subdir: Optional[str] = None
    ) -> Tuple[Figure, Path]:
        """
        Plot time series for a metric across methods.

        Args:
            data: DataFrame with columns: method, seed, step, metric, value
            methods: List of method names to plot
            metric: Metric name to plot
            x_axis: X-axis variable ('step' or 'task')
            confidence_level: Confidence level for bands (default: 0.95)
            show_bands: Whether to show confidence bands
            subsample: Optional subsampling factor (plot every Nth point)
            title: Plot title (optional)
            xlabel: X-axis label (optional)
            ylabel: Y-axis label (optional)
            filename: Output filename base (optional, auto-generated if None)
            subdir: Output subdirectory (optional)

        Returns:
            Tuple of (figure, saved_path)
        """
        self.logger.info(f"Plotting time series for metric='{metric}', methods={methods}")

        # Filter data for the specific metric
        plot_data = data[data['metric'] == metric].copy()

        if plot_data.empty:
            raise ValueError(f"No data found for metric '{metric}'")

        # Create figure
        fig, ax = self.setup_figure(figure_type='time_series')

        # Plot each method
        for method in methods:
            method_data = plot_data[plot_data['method'] == method]

            if method_data.empty:
                self.logger.warning(f"No data for method '{method}'. Skipping.")
                continue

            # Compute statistics across seeds
            stats = self._compute_statistics(
                method_data,
                x_axis=x_axis,
                confidence_level=confidence_level
            )

            if stats.empty:
                continue

            # Subsample if requested
            if subsample and subsample > 1:
                stats = stats.iloc[::subsample]

            # Get method style
            style = self.get_method_style(method)

            # Plot mean
            ax.plot(
                stats[x_axis],
                stats['mean'],
                label=method,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2,
                alpha=0.9
            )

            # Plot confidence bands
            if show_bands and 'ci_lower' in stats.columns and 'ci_upper' in stats.columns:
                ax.fill_between(
                    stats[x_axis],
                    stats['ci_lower'],
                    stats['ci_upper'],
                    color=style['color'],
                    alpha=0.2,
                    linewidth=0
                )

        # Set labels and title
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=14)
        else:
            ax.set_xlabel(x_axis.capitalize(), fontsize=14)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=14)
        else:
            # Get display name from config
            metric_names = self.get_config('metrics.display_names', {})
            ylabel = metric_names.get(metric, metric.replace('_', ' ').title())
            ax.set_ylabel(ylabel, fontsize=14)

        if title:
            ax.set_title(title, fontsize=16, fontweight='bold')

        # Add legend
        ax.legend(
            loc='best',
            fontsize=11,
            framealpha=0.9,
            edgecolor='gray'
        )

        # Tight layout
        fig.tight_layout()

        # Generate filename if not provided
        if filename is None:
            filename = f"{metric}_time_series"

        # Export figure
        saved_paths = self.export(fig, filename, subdir=subdir)

        return fig, saved_paths[0] if saved_paths else None

    def _compute_statistics(
        self,
        data: pd.DataFrame,
        x_axis: str,
        confidence_level: float
    ) -> pd.DataFrame:
        """
        Compute mean and confidence intervals across seeds.

        Args:
            data: DataFrame with method data
            x_axis: X-axis variable ('step' or 'task')
            confidence_level: Confidence level

        Returns:
            DataFrame with columns: x_axis, mean, std, ci_lower, ci_upper, n_seeds
        """
        # Group by x_axis and compute statistics
        grouped = data.groupby(x_axis)['value'].agg(['mean', 'std', 'count', 'sem'])

        # Compute confidence intervals
        from scipy.stats import t as t_dist

        # t-value for confidence level
        alpha = 1 - confidence_level
        t_value = t_dist.ppf(1 - alpha / 2, grouped['count'] - 1)

        # Confidence intervals
        margin = t_value * grouped['sem']
        grouped['ci_lower'] = grouped['mean'] - margin
        grouped['ci_upper'] = grouped['mean'] + margin
        grouped['n_seeds'] = grouped['count']

        # Reset index to make x_axis a column
        result = grouped.reset_index()
        result = result.rename(columns={'count': 'n_seeds'})

        return result
