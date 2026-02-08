"""
ComparisonBarPlotAgent: Create bar charts comparing methods with significance markers.

Generates publication-quality bar charts showing method performance across
metrics with error bars and statistical significance annotations.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from .base_plotter import PlotGeneratorAgent


class ComparisonBarPlotAgent(PlotGeneratorAgent):
    """
    Agent for creating bar chart comparisons with significance markers.

    Features:
    - Grouped or side-by-side bars
    - Error bars (CI or std)
    - Significance markers (*, **, ***)
    - Horizontal or vertical orientation
    """

    def __init__(self, name: str = 'bar_comparison_plotter', config: Optional[Dict[str, Any]] = None):
        """
        Initialize ComparisonBarPlotAgent.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)

    def execute(
        self,
        summary_data: pd.DataFrame,
        methods: List[str],
        metrics: List[str],
        significance_results: Optional[pd.DataFrame] = None,
        orientation: str = 'vertical',
        error_type: str = 'std',  # 'std', 'sem', 'ci'
        title: Optional[str] = None,
        filename: Optional[str] = None,
        subdir: Optional[str] = None
    ) -> Tuple[Figure, Path]:
        """
        Create bar chart comparison with optional significance markers.

        Args:
            summary_data: DataFrame with columns: method, metric, mean, std (and optionally ci_lower, ci_upper)
            methods: List of methods to plot
            metrics: List of metrics to plot
            significance_results: Optional DataFrame from StatisticalTestAgent with significance flags
            orientation: 'vertical' or 'horizontal'
            error_type: Type of error bars ('std', 'sem', 'ci')
            title: Plot title
            filename: Output filename
            subdir: Output subdirectory

        Returns:
            Tuple of (figure, saved_path)

        Example:
            ```python
            bar_plotter = ComparisonBarPlotAgent(config=default_config)
            fig, path = bar_plotter.execute(
                summary_data=comparison['summary'],
                methods=['S&P', 'UPGD (Full)', 'UPGD (Output Only)'],
                metrics=['accuracy'],
                significance_results=test_results
            )
            ```
        """
        self.logger.info(f"Creating bar comparison for {len(methods)} methods on {len(metrics)} metrics")

        # Create figure
        fig, ax = self.setup_figure(figure_type='comparison')

        # Determine bar positions
        n_metrics = len(metrics)
        n_methods = len(methods)
        bar_width = 0.8 / n_methods
        x_positions = np.arange(n_metrics)

        # Plot bars for each method
        for i, method in enumerate(methods):
            method_data = summary_data[summary_data['method'] == method]

            # Extract values for each metric
            means = []
            errors = []

            for metric in metrics:
                metric_row = method_data[method_data['metric'] == metric]

                if metric_row.empty:
                    means.append(0)
                    errors.append(0)
                    continue

                mean_val = metric_row['mean'].iloc[0]
                means.append(mean_val)

                # Get error based on type
                if error_type == 'std' and 'std' in metric_row.columns:
                    error = metric_row['std'].iloc[0]
                elif error_type == 'sem' and 'sem' in metric_row.columns:
                    error = metric_row['sem'].iloc[0]
                elif error_type == 'ci' and 'ci_lower' in metric_row.columns:
                    # For CI, error is distance from mean to bounds
                    error_lower = mean_val - metric_row['ci_lower'].iloc[0]
                    error_upper = metric_row['ci_upper'].iloc[0] - mean_val
                    error = [error_lower, error_upper]
                else:
                    error = 0

                errors.append(error)

            # Get method style
            style = self.get_method_style(method)

            # Calculate bar positions for this method
            if orientation == 'vertical':
                bar_positions = x_positions + (i - n_methods/2 + 0.5) * bar_width
                ax.bar(
                    bar_positions,
                    means,
                    bar_width,
                    label=method,
                    color=style['color'],
                    alpha=0.8,
                    yerr=errors if error_type != 'ci' else None,
                    error_kw={'linewidth': 1.5, 'capsize': 4}
                )

                # Add significance markers if provided
                if significance_results is not None:
                    self._add_significance_markers_vertical(
                        ax, method, metrics, means, bar_positions,
                        significance_results, methods[0]  # Assume first method is baseline
                    )
            else:
                bar_positions = x_positions + (i - n_methods/2 + 0.5) * bar_width
                ax.barh(
                    bar_positions,
                    means,
                    bar_width,
                    label=method,
                    color=style['color'],
                    alpha=0.8,
                    xerr=errors if error_type != 'ci' else None,
                    error_kw={'linewidth': 1.5, 'capsize': 4}
                )

        # Set labels and formatting
        metric_display_names = self.get_config('metrics.display_names', {})
        metric_labels = [metric_display_names.get(m, m.replace('_', ' ').title())
                        for m in metrics]

        if orientation == 'vertical':
            ax.set_xticks(x_positions)
            ax.set_xticklabels(metric_labels, rotation=45, ha='right')
            ax.set_ylabel('Value', fontsize=14)
        else:
            ax.set_yticks(x_positions)
            ax.set_yticklabels(metric_labels)
            ax.set_xlabel('Value', fontsize=14)

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
            filename = f"method_comparison_{'_'.join(metrics[:2])}"

        # Export figure
        saved_paths = self.export(fig, filename, subdir=subdir)

        return fig, saved_paths[0] if saved_paths else None

    def _add_significance_markers_vertical(
        self,
        ax,
        method: str,
        metrics: List[str],
        means: List[float],
        bar_positions: np.ndarray,
        significance_results: pd.DataFrame,
        baseline_method: str
    ):
        """Add significance markers (* ** ***) above bars."""
        # Only add markers for non-baseline methods
        if method == baseline_method:
            return

        for i, metric in enumerate(metrics):
            # Find significance result for this method vs baseline on this metric
            sig_row = significance_results[
                (significance_results['method_a'] == baseline_method) &
                (significance_results['method_b'] == method) &
                (significance_results['metric'] == metric)
            ]

            if sig_row.empty:
                continue

            if not sig_row['significant'].iloc[0]:
                continue

            # Determine marker based on p-value
            p_val = sig_row['p_value_corrected'].iloc[0]
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            elif p_val < 0.05:
                marker = '*'
            else:
                continue

            # Add marker above bar
            y_pos = means[i] if means[i] > 0 else 0
            ax.text(
                bar_positions[i],
                y_pos * 1.05,
                marker,
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
