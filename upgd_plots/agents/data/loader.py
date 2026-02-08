"""
DataLoaderAgent: Load and preprocess experiment data from JSON logs.

This agent handles loading experiment data from the UPGD log files and
converting it into standardized pandas DataFrames for analysis.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from ..base import BaseAgent


class DataLoaderAgent(BaseAgent):
    """
    Agent for loading experiment data from JSON log files.

    Loads data for specified datasets, methods, and seeds, and converts
    it into a standardized pandas DataFrame format for downstream analysis.
    """

    def __init__(self, name: str = 'data_loader', config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataLoaderAgent.

        Args:
            name: Agent name
            config: Configuration dictionary with 'data' section
        """
        super().__init__(name, config)
        self.base_dir = Path(self.get_config('data.base_dir', '/scratch/gautschi/shin283/upgd'))
        self.logs_dir = Path(self.get_config('data.logs_dir', self.base_dir / 'logs'))
        self.datasets_config = self.get_config('data.datasets', {})

    def execute(
        self,
        dataset: str,
        methods: List[str],
        seeds: Optional[List[int]] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load experiment data for specified dataset and methods.

        Args:
            dataset: Dataset name ('mini_imagenet', 'input_mnist', 'emnist', 'cifar10')
            methods: List of method names to load (e.g., ['S&P', 'UPGD (Full)'])
            seeds: List of seed numbers to load. If None, loads all available seeds.
            metrics: List of metrics to extract. If None, loads all available metrics.

        Returns:
            DataFrame with columns: dataset, method, seed, step, metric, value

        Raises:
            ValueError: If dataset not found in configuration
        """
        self.logger.info(f"Loading data for dataset='{dataset}', methods={methods}")

        if dataset not in self.datasets_config:
            raise ValueError(
                f"Dataset '{dataset}' not found in configuration. "
                f"Available: {list(self.datasets_config.keys())}"
            )

        dataset_config = self.datasets_config[dataset]
        logs_subdir = dataset_config['logs_subdir']
        dataset_logs_dir = self.logs_dir / logs_subdir

        if not dataset_logs_dir.exists():
            raise FileNotFoundError(f"Logs directory not found: {dataset_logs_dir}")

        all_data = []

        for method in methods:
            if method not in dataset_config['experiments']:
                self.logger.warning(
                    f"Method '{method}' not found in {dataset} configuration. Skipping."
                )
                continue

            method_config = dataset_config['experiments'][method]
            method_path = dataset_logs_dir / method_config['path']

            if not method_path.exists():
                self.logger.warning(f"Method path not found: {method_path}. Skipping.")
                continue

            # Load data for this method
            method_data = self._load_method_data(
                dataset=dataset,
                method=method,
                method_path=method_path,
                seeds=seeds,
                metrics=metrics
            )

            if not method_data.empty:
                all_data.append(method_data)
            else:
                self.logger.warning(f"No data loaded for {method}")

        if not all_data:
            self.logger.error("No data loaded for any method")
            return pd.DataFrame()

        # Concatenate all data
        result = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Loaded {len(result)} rows of data")

        return result

    def _load_method_data(
        self,
        dataset: str,
        method: str,
        method_path: Path,
        seeds: Optional[List[int]],
        metrics: Optional[List[str]]
    ) -> pd.DataFrame:
        """Load data for a single method across multiple seeds."""
        json_files = list(method_path.glob('*.json'))

        if not json_files:
            self.logger.warning(f"No JSON files found in {method_path}")
            return pd.DataFrame()

        # Filter by seeds if specified
        if seeds is not None:
            json_files = [f for f in json_files if f.stem.isdigit() and int(f.stem) in seeds]

        if not json_files:
            self.logger.warning(f"No matching seed files found for seeds={seeds}")
            return pd.DataFrame()

        all_rows = []

        for json_file in json_files:
            seed = int(json_file.stem) if json_file.stem.isdigit() else 0

            # Load JSON data
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading {json_file}: {e}")
                continue

            # Extract metrics
            rows = self._extract_metrics_from_json(
                data=data,
                dataset=dataset,
                method=method,
                seed=seed,
                metrics=metrics
            )

            all_rows.extend(rows)

        return pd.DataFrame(all_rows)

    def _extract_metrics_from_json(
        self,
        data: Dict,
        dataset: str,
        method: str,
        seed: int,
        metrics: Optional[List[str]]
    ) -> List[Dict]:
        """Extract metrics from a single JSON file."""
        rows = []

        # Map of JSON keys to metric names
        metric_keys = {
            'accuracy': 'accuracy_per_step',
            'loss': 'losses_per_step',
            'plasticity': 'plasticity_per_step',
            'n_dead_units': 'n_dead_units_per_step',
            'weight_rank': 'weight_rank_per_step',
            'weight_l2': 'weight_l2_per_step',
            'weight_l1': 'weight_l1_per_step',
            'grad_l2': 'grad_l2_per_step',
            'grad_l1': 'grad_l1_per_step',
            'grad_l0': 'grad_l0_per_step',
        }

        # If metrics not specified, try to load all available
        if metrics is None:
            metrics = list(metric_keys.keys())

        for metric in metrics:
            if metric not in metric_keys:
                self.logger.warning(f"Unknown metric: {metric}")
                continue

            json_key = metric_keys[metric]

            if json_key not in data:
                continue

            metric_data = data[json_key]

            # Handle both list and numpy array formats
            if isinstance(metric_data, (list, np.ndarray)):
                metric_values = np.array(metric_data)
            else:
                self.logger.warning(f"Unexpected data type for {json_key}: {type(metric_data)}")
                continue

            # Create rows for each step
            for step, value in enumerate(metric_values):
                rows.append({
                    'dataset': dataset,
                    'method': method,
                    'seed': seed,
                    'step': step,
                    'metric': metric,
                    'value': float(value) if not np.isnan(value) else None,
                })

        return rows

    def get_available_methods(self, dataset: str) -> List[str]:
        """
        Get list of available methods for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            List of method names configured for this dataset
        """
        if dataset not in self.datasets_config:
            return []

        return list(self.datasets_config[dataset]['experiments'].keys())

    def get_available_seeds(self, dataset: str, method: str) -> List[int]:
        """
        Get list of available seeds for a dataset/method combination.

        Args:
            dataset: Dataset name
            method: Method name

        Returns:
            List of seed numbers that have JSON files
        """
        if dataset not in self.datasets_config:
            return []

        dataset_config = self.datasets_config[dataset]

        if method not in dataset_config['experiments']:
            return []

        method_config = dataset_config['experiments'][method]
        logs_subdir = dataset_config['logs_subdir']
        method_path = self.logs_dir / logs_subdir / method_config['path']

        if not method_path.exists():
            return []

        json_files = list(method_path.glob('*.json'))
        seeds = [int(f.stem) for f in json_files if f.stem.isdigit()]

        return sorted(seeds)
