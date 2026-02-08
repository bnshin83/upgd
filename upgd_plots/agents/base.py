"""
Base agent class providing common functionality for all UPGD analysis agents.

All agents inherit from BaseAgent and implement the execute() method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import logging
import pickle


class BaseAgent(ABC):
    """
    Base class for all analysis agents.

    Provides:
    - Configuration management
    - Logging infrastructure
    - Result caching
    - State persistence

    All subclasses must implement the execute() method.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.

        Args:
            name: Unique name for this agent instance
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = self._setup_logger()
        self.cache = {}
        self.state = {}

        self.logger.info(f"Initialized {self.__class__.__name__}: {self.name}")

    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logger."""
        logger = logging.getLogger(f"upgd.agents.{self.name}")

        # Only add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Main execution method - must be implemented by subclasses.

        Args:
            **kwargs: Agent-specific parameters

        Returns:
            Agent-specific results
        """
        pass

    def reset(self) -> None:
        """Reset agent state and cache."""
        self.cache.clear()
        self.state.clear()
        self.logger.info(f"Reset {self.name}")

    def save_state(self, filepath: Path) -> None:
        """
        Persist agent state to disk.

        Args:
            filepath: Path to save state
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'config': self.config,
                'state': self.state,
                'cache': self.cache,
            }, f)

        self.logger.info(f"Saved state to {filepath}")

    def load_state(self, filepath: Path) -> None:
        """
        Load agent state from disk.

        Args:
            filepath: Path to load state from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.name = data['name']
        self.config = data['config']
        self.state = data['state']
        self.cache = data['cache']

        self.logger.info(f"Loaded state from {filepath}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback to default.

        Args:
            key: Configuration key (supports nested keys with dots, e.g., 'data.base_dir')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
