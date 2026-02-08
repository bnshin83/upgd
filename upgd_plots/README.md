# UPGD Agent-Based Analysis Framework

A modular, agent-based Python framework for analyzing and visualizing UPGD continual learning experiments.

## Features

- **Modular Architecture**: Composable agents for different analysis tasks
- **Interactive Classes**: Importable agents for use in notebooks and scripts
- **Statistical Analysis**: Hypothesis testing, effect sizes, multiple comparison corrections
- **Publication-Quality Plots**: Consistent styling, PNG/PDF export
- **Multi-Dataset Support**: Mini-ImageNet, Input-Permuted MNIST, EMNIST, CIFAR-10

## Installation

### Prerequisites

```bash
# Activate your UPGD environment
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate

# Required packages (already installed)
# - scipy>=1.7.0
# - pandas>=2.3.0
# - matplotlib>=3.5.0
# - numpy>=1.17.0
```

## Quick Start

### Basic Example

```python
from upgd_plots.agents.data import DataLoaderAgent
from upgd_plots.agents.stats import StatisticalTestAgent
from upgd_plots.agents.plot import TimeSeriesPlotAgent
from upgd_plots.config import default_config

# 1. Load data
loader = DataLoaderAgent(config=default_config)
data = loader.execute(
    dataset='mini_imagenet',
    methods=['S&P', 'UPGD (Full)', 'UPGD (Output Only)'],
    seeds=[1, 2, 3],
    metrics=['accuracy']
)

# 2. Perform statistical tests
tester = StatisticalTestAgent(config=default_config)
results = tester.execute(
    data=data,
    baseline_method='S&P',
    comparison_methods=['UPGD (Full)', 'UPGD (Output Only)'],
    metric='accuracy',
    test_type='wilcoxon',
    correction='holm'
)

print("Significant differences:")
print(results[results['significant']])

# 3. Plot time series
plotter = TimeSeriesPlotAgent(config=default_config)
fig, path = plotter.execute(
    data=data,
    methods=['S&P', 'UPGD (Full)', 'UPGD (Output Only)'],
    metric='accuracy',
    subsample=1000,
    subdir='mini_imagenet'
)
print(f"Saved plot to: {path}")
```

### Run the Example Script

```bash
cd /scratch/gautschi/shin283/upgd/upgd_plots
python examples/basic_usage.py
```

## Architecture

### Agent Hierarchy

```
BaseAgent
├── Data Management
│   ├── DataLoaderAgent - Load experiment JSON files
│   ├── UtilityHistogramLoaderAgent [Phase 2]
│   └── AggregatorAgent [Phase 2]
├── Statistical Analysis
│   ├── StatisticalTestAgent - Hypothesis testing
│   ├── MethodRankingAgent [Phase 2]
│   └── ComparatorAgent [Phase 2]
├── Plot Generation
│   ├── PlotGeneratorAgent (base)
│   ├── TimeSeriesPlotAgent - Training curves
│   ├── ComparisonBarPlotAgent [Phase 2]
│   └── UtilityHistogramPlotAgent [Phase 2]
└── Export [Phase 2]
    ├── ExportAgent - CSV, LaTeX, JSON
    └── ReportGeneratorAgent - HTML/PDF reports
```

## Available Agents (Phase 1)

### DataLoaderAgent

Load experiment data from JSON log files.

```python
loader = DataLoaderAgent(config=default_config)

# Load specific methods and seeds
data = loader.execute(
    dataset='mini_imagenet',  # or 'input_mnist', 'emnist', 'cifar10'
    methods=['UPGD (Full)', 'UPGD (Output Only)'],
    seeds=[1, 2, 3],
    metrics=['accuracy', 'loss', 'plasticity']
)

# Returns pandas DataFrame with columns:
# - dataset, method, seed, step, metric, value
```

**Key Methods:**
- `execute()`: Load data
- `get_available_methods(dataset)`: List available methods
- `get_available_seeds(dataset, method)`: List available seeds

### StatisticalTestAgent

Perform statistical hypothesis tests.

```python
tester = StatisticalTestAgent(config=default_config)

results = tester.execute(
    data=data,
    baseline_method='S&P',
    comparison_methods=['UPGD (Full)', 'UPGD (Output Only)'],
    metric='accuracy',
    test_type='wilcoxon',  # 'ttest', 'ttest_paired', 'wilcoxon', 'anova'
    correction='holm',      # 'bonferroni', 'holm', 'fdr', None
    alpha=0.05,
    effect_size_type='cohen_d'  # 'cohen_d', 'cliff_delta'
)

# Returns DataFrame with p-values, corrected p-values, significance flags
```

**Supported Tests:**
- Independent t-test (`ttest`)
- Paired t-test (`ttest_paired`)
- Wilcoxon signed-rank test (`wilcoxon`) - **default, non-parametric**
- ANOVA (`anova`)

**Correction Methods:**
- Bonferroni
- Holm-Bonferroni (default)
- Benjamini-Hochberg FDR

### TimeSeriesPlotAgent

Create publication-quality time series plots.

```python
plotter = TimeSeriesPlotAgent(config=default_config)

fig, path = plotter.execute(
    data=data,
    methods=['S&P', 'UPGD (Full)'],
    metric='accuracy',
    x_axis='step',
    confidence_level=0.95,
    show_bands=True,
    subsample=1000,  # Plot every 1000th point for speed
    title='Accuracy Comparison',
    subdir='mini_imagenet',
    filename='accuracy_time_series'
)
```

**Features:**
- Mean trajectory with confidence bands
- Multiple methods on same plot
- Subsampling for large datasets
- Automatic confidence interval calculation
- Consistent colors and styling from config

## Configuration

All agents use a centralized configuration system defined in `config/default_config.py`.

### Key Configuration Sections

```python
default_config = {
    'data': {
        'base_dir': '/scratch/gautschi/shin283/upgd',
        'logs_dir': '/scratch/gautschi/shin283/upgd/logs',
        'datasets': {
            'mini_imagenet': {...},
            'input_mnist': {...},
            'emnist': {...},
            'cifar10': {...}
        }
    },
    'stats': {
        'default_alpha': 0.05,
        'default_correction': 'holm',
        'default_test': 'wilcoxon',
        'confidence_level': 0.95
    },
    'plotting': {
        'dpi': 150,
        'colors': {...},
        'linestyles': {...},
        'output_dir': 'upgd_plots/figures'
    }
}
```

### Customizing Configuration

```python
# Option 1: Override specific settings
custom_config = default_config.copy()
custom_config['stats']['default_alpha'] = 0.01

# Option 2: Pass custom config to agent
loader = DataLoaderAgent(config=custom_config)
```

## Data Format

All agents operate on standardized pandas DataFrames:

```python
# Standard format
pd.DataFrame({
    'dataset': str,    # 'mini_imagenet', 'cifar10', etc.
    'method': str,     # 'UPGD (Full)', 'S&P', etc.
    'seed': int,       # Random seed
    'step': int,       # Training step (0 to n_samples-1)
    'metric': str,     # 'accuracy', 'loss', 'plasticity', etc.
    'value': float,    # Metric value
})
```

## Available Metrics

- `accuracy`: Classification accuracy
- `loss`: Training loss
- `plasticity`: Neural plasticity measure
- `n_dead_units`: Number of inactive neurons
- `weight_rank`: Weight matrix rank
- `weight_l2`, `weight_l1`: Weight norms
- `grad_l2`, `grad_l1`, `grad_l0`: Gradient norms

## File Organization

```
upgd_plots/
├── agents/                  # Agent modules
│   ├── base.py             # BaseAgent class
│   ├── data/               # Data management agents
│   │   └── loader.py
│   ├── stats/              # Statistical agents
│   │   └── tests.py
│   ├── plot/               # Plotting agents
│   │   ├── base_plotter.py
│   │   └── time_series.py
│   └── export/             # Export agents [Phase 2]
├── config/                  # Configuration
│   └── default_config.py
├── examples/                # Usage examples
│   └── basic_usage.py
├── figures/                 # Generated plots
│   ├── mini_imagenet/
│   ├── input_mnist/
│   ├── emnist/
│   └── cifar10/
└── scripts/                 # Legacy scripts (kept for compatibility)
```

## Roadmap

### Phase 1 ✅ (Completed)
- [x] BaseAgent infrastructure
- [x] DataLoaderAgent
- [x] StatisticalTestAgent
- [x] PlotGeneratorAgent (base)
- [x] TimeSeriesPlotAgent
- [x] Configuration system
- [x] Example scripts

### Phase 2 (Future)
- [ ] AggregatorAgent - Per-task aggregation
- [ ] ComparatorAgent - Multi-metric comparison tables
- [ ] ComparisonBarPlotAgent - Bar charts with significance markers
- [ ] UtilityHistogramLoaderAgent - Load utility data
- [ ] UtilityHistogramPlotAgent - Utility distribution plots
- [ ] MethodRankingAgent - Method ranking with critical difference tests
- [ ] ExportAgent - LaTeX tables, CSV export

### Phase 3 (Future)
- [ ] Pipeline orchestration
- [ ] Report generation
- [ ] Critical difference plots
- [ ] Heatmap plots

## Troubleshooting

### ImportError: No module named 'scipy'

```bash
source /scratch/gautschi/shin283/upgd/.upgd/bin/activate
pip install scipy pandas
```

### FileNotFoundError: Logs directory not found

Check that the `base_dir` in config points to your UPGD installation:

```python
from config import default_config
print(default_config['data']['base_dir'])
# Should be: /scratch/gautschi/shin283/upgd
```

### No data loaded for method

Verify the method name matches the configuration:

```python
loader = DataLoaderAgent(config=default_config)
available = loader.get_available_methods('mini_imagenet')
print(f"Available methods: {available}")
```

## Contributing

When adding new agents:

1. Inherit from `BaseAgent` (or `PlotGeneratorAgent` for plotting)
2. Implement the `execute()` method
3. Add comprehensive docstrings
4. Update the corresponding `__init__.py`
5. Add usage examples

## License

Part of the UPGD project.
