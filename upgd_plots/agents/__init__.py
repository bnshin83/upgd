"""
UPGD Agent-Based Analysis and Plotting Framework

This package provides modular, composable agents for analyzing and visualizing
UPGD continual learning experiments.

Agent Types:
- Data Management: DataLoaderAgent, UtilityHistogramLoaderAgent, AggregatorAgent
- Statistical Analysis: StatisticalTestAgent, MethodRankingAgent, ComparatorAgent
- Plot Generation: TimeSeriesPlotAgent, ComparisonBarPlotAgent, UtilityHistogramPlotAgent
- Export: ExportAgent, ReportGeneratorAgent
"""

from .base import BaseAgent

__version__ = '0.1.0'
__all__ = ['BaseAgent']
