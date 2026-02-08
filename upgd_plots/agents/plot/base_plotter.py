"""
PlotGeneratorAgent: Base class for all plotting agents.

Provides shared functionality for publication-quality plotting including
consistent styling, color schemes, and export capabilities.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..base import BaseAgent


class PlotGeneratorAgent(BaseAgent):
    """
    Base class for all plotting agents.

    Provides:
    - Publication-quality matplotlib defaults
    - Consistent color schemes and styling
    - Export to multiple formats (PNG, PDF)
    - Figure management
    """

    def __init__(self, name: str = 'plotter', config: Optional[Dict[str, Any]] = None):
        """
        Initialize PlotGeneratorAgent.

        Args:
            name: Agent name
            config: Configuration dictionary with 'plotting' section
        """
        super().__init__(name, config)
        self._setup_matplotlib_style()

        # Get plotting configuration
        self.output_dir = Path(self.get_config('plotting.output_dir', 'upgd_plots/figures'))
        self.dpi = self.get_config('plotting.dpi', 150)
        self.formats = self.get_config('plotting.formats', ['png', 'pdf'])
        self.figure_sizes = self.get_config('plotting.figure_sizes', {})
        self.colors = self.get_config('plotting.colors', {})
        self.linestyles = self.get_config('plotting.linestyles', {})
        self.markers = self.get_config('plotting.markers', {})

    def _setup_matplotlib_style(self):
        """Setup matplotlib style for publication-quality plots."""
        style = self.get_config('plotting.style', 'seaborn-v0_8-whitegrid')

        try:
            plt.style.use(style)
        except:
            self.logger.warning(f"Style '{style}' not available, using default")

        # Set publication-quality parameters
        matplotlib.rcParams['axes.spines.right'] = False
        matplotlib.rcParams['axes.spines.top'] = False
        matplotlib.rcParams['pdf.fonttype'] = self.get_config('plotting.pdf_fonttype', 42)
        matplotlib.rcParams['ps.fonttype'] = self.get_config('plotting.ps_fonttype', 42)
        matplotlib.rcParams.update({
            'font.size': self.get_config('plotting.font_size', 12)
        })

    def setup_figure(
        self,
        figure_type: str = 'time_series',
        figsize: Optional[Tuple[float, float]] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create figure with publication-quality defaults.

        Args:
            figure_type: Type of figure ('time_series', 'comparison', 'histogram', 'heatmap')
            figsize: Custom figure size (width, height) in inches. If None, uses default for type.

        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = self.figure_sizes.get(figure_type, (12, 7))

        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        self.apply_style(ax)

        return fig, ax

    def apply_style(self, ax: Axes) -> None:
        """
        Apply consistent styling to axes.

        Args:
            ax: Matplotlib axes object
        """
        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    def get_method_style(self, method: str) -> Dict[str, Any]:
        """
        Get plotting style for a method.

        Args:
            method: Method name

        Returns:
            Dictionary with 'color', 'linestyle', 'marker' keys
        """
        return {
            'color': self.colors.get(method, '#000000'),
            'linestyle': self.linestyles.get(method, '-'),
            'marker': self.markers.get(method, 'o'),
        }

    def export(
        self,
        fig: Figure,
        filename: str,
        subdir: Optional[str] = None,
        formats: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Export figure to multiple formats.

        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
            subdir: Optional subdirectory within output_dir
            formats: List of formats to export. If None, uses config defaults.

        Returns:
            List of paths where figure was saved
        """
        if formats is None:
            formats = self.formats

        # Determine output directory
        if subdir:
            output_dir = self.output_dir / subdir
        else:
            output_dir = self.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for fmt in formats:
            filepath = output_dir / f"{filename}.{fmt}"

            try:
                fig.savefig(
                    filepath,
                    format=fmt,
                    dpi=self.dpi,
                    bbox_inches='tight',
                    pad_inches=0.1
                )
                saved_paths.append(filepath)
                self.logger.info(f"Saved plot to {filepath}")
            except Exception as e:
                self.logger.error(f"Error saving {filepath}: {e}")

        return saved_paths

    def close_figure(self, fig: Figure) -> None:
        """
        Close a figure to free memory.

        Args:
            fig: Matplotlib figure
        """
        plt.close(fig)

    def execute(self, **kwargs) -> Any:
        """
        Base execute method - must be overridden by subclasses.

        Raises:
            NotImplementedError: This is an abstract base class
        """
        raise NotImplementedError(
            "PlotGeneratorAgent is an abstract base class. "
            "Use a specific plotting agent like TimeSeriesPlotAgent."
        )
