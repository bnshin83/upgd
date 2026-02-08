"""
UtilityHistogramPlotAgent: Visualize utility distributions.

Creates publication-quality plots showing utility histogram distributions
across methods with scatter or bar plot styles.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure

from .base_plotter import PlotGeneratorAgent


class UtilityHistogramPlotAgent(PlotGeneratorAgent):
    """
    Agent for plotting utility histogram distributions.

    Features:
    - Scatter or bar plot styles
    - Global and per-layer views
    - Log scale support
    - Reference lines at key thresholds (0.52)
    """

    def __init__(self, name: str = 'utility_histogram_plotter', config: Optional[Dict[str, Any]] = None):
        """
        Initialize UtilityHistogramPlotAgent.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)

        # Utility bin labels
        self.utility_bins = [
            '[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.44)', '[0.44, 0.48)',
            '[0.48, 0.52)', '[0.52, 0.56)', '[0.56, 0.6)', '[0.6, 0.8)',
            '[0.8, 1.0]'
        ]

    def execute(
        self,
        histogram_data: Dict[str, Any],
        methods: List[str],
        plot_type: str = 'scatter',  # 'scatter' or 'bar'
        log_scale: bool = False,
        per_layer: bool = False,
        layer: Optional[str] = None,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        subdir: Optional[str] = None
    ) -> Tuple[Figure, Path]:
        """
        Plot utility histogram comparison.

        Args:
            histogram_data: Dictionary from UtilityHistogramLoaderAgent
            methods: List of methods to plot
            plot_type: 'scatter' or 'bar'
            log_scale: Use log scale for y-axis
            per_layer: If True, plot per-layer data; if False, plot global
            layer: Specific layer to plot (only if per_layer=True)
            title: Plot title
            filename: Output filename
            subdir: Output subdirectory

        Returns:
            Tuple of (figure, saved_path)

        Example:
            ```python
            plotter = UtilityHistogramPlotAgent(config=default_config)
            fig, path = plotter.execute(
                histogram_data=hist_data,
                methods=['UPGD (Full)', 'UPGD (Output Only)'],
                plot_type='scatter',
                log_scale=False
            )
            ```
        """
        self.logger.info(
            f"Plotting utility histograms for {len(methods)} methods "
            f"(type={plot_type}, log={log_scale}, per_layer={per_layer})"
        )

        # Create figure
        fig, ax = self.setup_figure(figure_type='histogram')

        # X-axis positions for bins
        x_positions = np.arange(len(self.utility_bins))

        if plot_type == 'scatter':
            self._plot_scatter(
                ax, histogram_data, methods, x_positions,
                per_layer=per_layer, layer=layer
            )
        else:  # bar
            self._plot_bar(
                ax, histogram_data, methods, x_positions,
                per_layer=per_layer, layer=layer
            )

        # Set labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(self.utility_bins, rotation=45, ha='right')
        ax.set_xlabel('Utility Range', fontsize=14)
        ax.set_ylabel('Percentage of Parameters (%)', fontsize=14)

        # Add reference line at bin 4 (0.48-0.52) boundary
        ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='0.52 threshold')

        # Log scale if requested
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Percentage of Parameters (%) [log scale]', fontsize=14)

        # Title
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold')
        elif per_layer and layer:
            ax.set_title(f'Utility Distribution - {layer}', fontsize=16, fontweight='bold')
        else:
            ax.set_title('Global Utility Distribution', fontsize=16, fontweight='bold')

        # Legend
        ax.legend(
            loc='upper right',
            fontsize=11,
            framealpha=0.9,
            edgecolor='gray',
            ncol=1 if len(methods) <= 4 else 2
        )

        # Tight layout
        fig.tight_layout()

        # Generate filename
        if filename is None:
            layer_suffix = f"_{layer}" if per_layer and layer else ("_per_layer" if per_layer else "_global")
            log_suffix = "_log" if log_scale else ""
            filename = f"utility_histogram{layer_suffix}{log_suffix}"

        # Export
        saved_paths = self.export(fig, filename, subdir=subdir)
        return fig, saved_paths[0] if saved_paths else None

    def _plot_scatter(
        self,
        ax,
        histogram_data: Dict,
        methods: List[str],
        x_positions: np.ndarray,
        per_layer: bool,
        layer: Optional[str]
    ):
        """Plot utility histograms as scatter plot."""
        for method in methods:
            if method not in histogram_data:
                self.logger.warning(f"Method '{method}' not in histogram data")
                continue

            method_data = histogram_data[method]

            # Get histogram values
            if per_layer:
                if 'utility' not in method_data or 'layers' not in method_data['utility']:
                    self.logger.warning(f"No per-layer data for {method}")
                    continue

                if layer:
                    if layer not in method_data['utility']['layers']:
                        self.logger.warning(f"Layer '{layer}' not found for {method}")
                        continue
                    hist_dict = method_data['utility']['layers'][layer]
                else:
                    # Average across all layers
                    layers_data = method_data['utility']['layers']
                    hist_dict = self._average_layers(layers_data)
            else:
                if 'utility' not in method_data or 'global' not in method_data['utility']:
                    self.logger.warning(f"No global utility data for {method}")
                    continue
                hist_dict = method_data['utility']['global']

            # Extract values in bin order
            values = [hist_dict.get(bin_label, 0.0) for bin_label in self.utility_bins]

            # Get style
            style = self.get_method_style(method)

            # Plot
            ax.scatter(
                x_positions,
                values,
                label=method,
                color=style['color'],
                marker=style['marker'],
                s=100,
                alpha=0.8,
                edgecolors='black',
                linewidths=1
            )

            # Connect with lines
            ax.plot(
                x_positions,
                values,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=1.5,
                alpha=0.5
            )

    def _plot_bar(
        self,
        ax,
        histogram_data: Dict,
        methods: List[str],
        x_positions: np.ndarray,
        per_layer: bool,
        layer: Optional[str]
    ):
        """Plot utility histograms as bar chart."""
        n_methods = len(methods)
        bar_width = 0.8 / n_methods

        for i, method in enumerate(methods):
            if method not in histogram_data:
                continue

            method_data = histogram_data[method]

            # Get histogram values (same logic as scatter)
            if per_layer:
                if 'utility' not in method_data or 'layers' not in method_data['utility']:
                    continue
                if layer:
                    if layer not in method_data['utility']['layers']:
                        continue
                    hist_dict = method_data['utility']['layers'][layer]
                else:
                    layers_data = method_data['utility']['layers']
                    hist_dict = self._average_layers(layers_data)
            else:
                if 'utility' not in method_data or 'global' not in method_data['utility']:
                    continue
                hist_dict = method_data['utility']['global']

            values = [hist_dict.get(bin_label, 0.0) for bin_label in self.utility_bins]
            style = self.get_method_style(method)

            # Calculate bar positions
            bar_positions = x_positions + (i - n_methods/2 + 0.5) * bar_width

            ax.bar(
                bar_positions,
                values,
                bar_width,
                label=method,
                color=style['color'],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )

    def _average_layers(self, layers_data: Dict[str, Dict]) -> Dict[str, float]:
        """Average histogram values across all layers."""
        # Get all layer names
        layer_names = list(layers_data.keys())

        if not layer_names:
            return {}

        # Initialize averaged histogram
        averaged = {}

        for bin_label in self.utility_bins:
            values = [layers_data[layer].get(bin_label, 0.0) for layer in layer_names]
            averaged[bin_label] = np.mean(values)

        return averaged

    def plot_per_layer_comparison(
        self,
        histogram_data: Dict[str, Any],
        method: str,
        log_scale: bool = False,
        filename: Optional[str] = None,
        subdir: Optional[str] = None
    ) -> Tuple[Figure, Path]:
        """
        Plot all layers for a single method in subplots.

        Args:
            histogram_data: Dictionary from UtilityHistogramLoaderAgent
            method: Method name
            log_scale: Use log scale
            filename: Output filename
            subdir: Output subdirectory

        Returns:
            Tuple of (figure, saved_path)
        """
        self.logger.info(f"Plotting per-layer comparison for {method}")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=self.dpi)
        layers = ['linear_1', 'linear_2', 'linear_3']

        if method not in histogram_data:
            raise ValueError(f"Method '{method}' not found in histogram data")

        method_data = histogram_data[method]
        style = self.get_method_style(method)

        for idx, layer in enumerate(layers):
            ax = axes[idx]
            x_positions = np.arange(len(self.utility_bins))

            if layer in method_data.get('utility', {}).get('layers', {}):
                hist_dict = method_data['utility']['layers'][layer]
                values = [hist_dict.get(bin_label, 0.0) for bin_label in self.utility_bins]

                ax.scatter(x_positions, values, color=style['color'], marker=style['marker'],
                          s=100, alpha=0.8, edgecolors='black', linewidths=1)
                ax.plot(x_positions, values, color=style['color'], linestyle=style['linestyle'],
                       linewidth=1.5, alpha=0.5)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(self.utility_bins, rotation=45, ha='right')
            ax.set_xlabel('Utility Range', fontsize=12)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title(layer, fontsize=14, fontweight='bold')
            ax.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.grid(True, alpha=0.3)

            if log_scale:
                ax.set_yscale('log')

        fig.suptitle(f'{method} - Per-Layer Utility Distribution', fontsize=16, fontweight='bold')
        fig.tight_layout()

        if filename is None:
            filename = f"utility_per_layer_{method.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
            if log_scale:
                filename += "_log"

        saved_paths = self.export(fig, filename, subdir=subdir)
        return fig, saved_paths[0] if saved_paths else None
