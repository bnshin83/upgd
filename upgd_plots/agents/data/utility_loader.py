"""
UtilityHistogramLoaderAgent: Load utility histogram data for UPGD experiments.

Loads utility distribution data including:
- 9-bin scaled utility histograms ([0, 1] range)
- 5-bin raw utility histograms (centered at 0)
- Per-layer histograms (linear_1, linear_2, linear_3)
- Global histograms (aggregated across all layers)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..base import BaseAgent


class UtilityHistogramLoaderAgent(BaseAgent):
    """
    Agent for loading utility histogram data.

    Loads pre-extracted utility histograms from JSON files in the
    data/utility_histograms/ directory.
    """

    def __init__(self, name: str = 'utility_loader', config: Optional[Dict[str, Any]] = None):
        """
        Initialize UtilityHistogramLoaderAgent.

        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.base_dir = Path(self.get_config('data.base_dir', '/scratch/gautschi/shin283/upgd'))
        self.plots_dir = self.base_dir / 'upgd_plots'
        self.data_dir = self.plots_dir / 'data' / 'utility_histograms'

    def execute(
        self,
        dataset: str,
        methods: Optional[List[str]] = None,
        histogram_type: str = 'both'  # 'utility', 'raw_utility', 'both'
    ) -> Dict[str, Any]:
        """
        Load utility histogram data for a dataset.

        Args:
            dataset: Dataset name ('mini_imagenet', 'input_mnist', 'emnist', 'cifar10')
            methods: List of methods to load. If None, loads all available methods.
            histogram_type: Type of histogram to load ('utility', 'raw_utility', 'both')

        Returns:
            Dictionary with structure:
            {
                'method_name': {
                    'utility': {
                        'global': {'[0, 0.2)': 0.0, '[0.2, 0.4)': 0.05, ...},
                        'layers': {
                            'linear_1': {'[0, 0.2)': 0.0, ...},
                            'linear_2': {...},
                            'linear_3': {...}
                        }
                    },
                    'raw_utility': {
                        'global': {'< -0.001': 0.0, ...}
                    },
                    'total_params': 12345
                }
            }

        Example:
            ```python
            loader = UtilityHistogramLoaderAgent(config=default_config)
            hist_data = loader.execute(
                dataset='mini_imagenet',
                methods=['UPGD (Full)', 'UPGD (Output Only)']
            )
            ```
        """
        self.logger.info(f"Loading utility histograms for dataset='{dataset}'")

        # Construct file path
        data_file = self.data_dir / f'{dataset}_utility_histograms.json'

        if not data_file.exists():
            raise FileNotFoundError(
                f"Utility histogram file not found: {data_file}\n"
                f"Run extract_utility_histograms_local.py first to generate this file."
            )

        # Load JSON data
        with open(data_file, 'r') as f:
            all_data = json.load(f)

        self.logger.info(f"Loaded data for {len(all_data)} methods from {data_file}")

        # Filter by methods if specified
        if methods is not None:
            filtered_data = {}
            for method in methods:
                if method in all_data:
                    filtered_data[method] = all_data[method]
                else:
                    self.logger.warning(f"Method '{method}' not found in histogram data")
            all_data = filtered_data

        # Filter by histogram type if specified
        if histogram_type != 'both':
            for method in all_data:
                if histogram_type == 'utility':
                    all_data[method].pop('raw_utility', None)
                elif histogram_type == 'raw_utility':
                    all_data[method].pop('utility', None)

        self.logger.info(f"Returning histogram data for {len(all_data)} methods")
        return all_data

    def get_available_methods(self, dataset: str) -> List[str]:
        """
        Get list of methods with available utility histogram data.

        Args:
            dataset: Dataset name

        Returns:
            List of method names
        """
        data_file = self.data_dir / f'{dataset}_utility_histograms.json'

        if not data_file.exists():
            self.logger.warning(f"Histogram file not found: {data_file}")
            return []

        with open(data_file, 'r') as f:
            all_data = json.load(f)

        return list(all_data.keys())

    def get_utility_bins(self) -> Dict[str, List[str]]:
        """
        Get the bin labels for utility histograms.

        Returns:
            Dictionary with 'scaled' and 'raw' bin labels
        """
        return {
            'scaled': [
                '[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.44)', '[0.44, 0.48)',
                '[0.48, 0.52)', '[0.52, 0.56)', '[0.56, 0.6)', '[0.6, 0.8)',
                '[0.8, 1.0]'
            ],
            'raw': [
                '< -0.001', '[-0.001, -0.0002)', '[-0.0002, 0.0002]',
                '(0.0002, 0.001]', '> 0.001'
            ]
        }

    def get_layers(self) -> List[str]:
        """
        Get the layer names used in per-layer histograms.

        Returns:
            List of layer names
        """
        return ['linear_1', 'linear_2', 'linear_3']
