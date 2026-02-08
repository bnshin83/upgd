"""
Default configuration for UPGD agents.

Centralizes all dataset configurations, plotting styles, and analysis parameters.
"""

from pathlib import Path

# Base paths
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR.parent  # upgd_plots/
PROJECT_DIR = PLOTS_DIR.parent  # upgd/

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================
# Each dataset has different log paths, experiment configurations, and parameters

DATASET_CONFIGS = {
    # Mini-ImageNet: label-permuted binary classification tasks
    'mini_imagenet': {
        'display_name': 'Mini-ImageNet',
        'logs_subdir': 'label_permuted_mini_imagenet_stats',
        'steps_per_task': 2500,  # Default for label-permuted datasets
        'n_samples': 1000000,
        'experiments': {
            'S&P': {
                'path': 'sgd/fully_connected_relu_with_hooks/lr_0.005_sigma_0.01_beta_utility_0.9_weight_decay_0.002_n_samples_1000000',
                'color': '#7f7f7f',
                'linestyle': '--',
                'marker': 'o',
            },
            'UPGD (Full)': {
                'path': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
                'color': '#1f77b4',
                'linestyle': '-',
                'marker': 's',
            },
            'UPGD (Output Only)': {
                'path': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
                'color': '#2ca02c',
                'linestyle': '-',
                'marker': '^',
            },
            'UPGD (Hidden Only)': {
                'path': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
                'color': '#ff7f0e',
                'linestyle': '-',
                'marker': 'D',
            },
            'UPGD (Hidden+Output)': {
                'path': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
                'color': '#9467bd',
                'linestyle': '-',
                'marker': 'v',
            },
            'UPGD (Clamped 0.52)': {
                'path': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
                'color': '#d62728',
                'linestyle': '-',
                'marker': 'p',
            },
            'UPGD (Clamped 0.48-0.52)': {
                'path': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
                'color': '#8c564b',
                'linestyle': '-',
                'marker': 'h',
            },
            'UPGD (Clamped 0.44-0.56)': {
                'path': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
                'color': '#e377c2',
                'linestyle': '-',
                'marker': '*',
            },
        }
    },

    # Input-Permuted MNIST: permutation changes every 5000 steps
    'input_mnist': {
        'display_name': 'Input-Permuted MNIST',
        'logs_subdir': 'input_permuted_mnist_stats',
        'steps_per_task': 5000,  # Different from other datasets
        'n_samples': 1000000,
        'experiments': {
            'S&P': {
                'path': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_weight_decay_0.01_beta_utility_0.9999_n_samples_1000000',
                'color': '#7f7f7f',
                'linestyle': '--',
                'marker': 'o',
            },
            'UPGD (Full)': {
                'path': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
                'color': '#1f77b4',
                'linestyle': '-',
                'marker': 's',
            },
            'UPGD (Output Only)': {
                'path': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_output_only_n_samples_1000000',
                'color': '#2ca02c',
                'linestyle': '-',
                'marker': '^',
            },
            'UPGD (Hidden Only)': {
                'path': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_hidden_only_n_samples_1000000',
                'color': '#ff7f0e',
                'linestyle': '-',
                'marker': 'D',
            },
            'UPGD (Hidden+Output)': {
                'path': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_gating_mode_hidden_and_output_n_samples_1000000',
                'color': '#9467bd',
                'linestyle': '-',
                'marker': 'v',
            },
            'UPGD (Clamped 0.52)': {
                'path': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_n_samples_1000000',
                'color': '#d62728',
                'linestyle': '-',
                'marker': 'p',
            },
            'UPGD (Clamped 0.48-0.52)': {
                'path': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
                'color': '#8c564b',
                'linestyle': '-',
                'marker': 'h',
            },
            'UPGD (Clamped 0.44-0.56)': {
                'path': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.1_beta_utility_0.9999_weight_decay_0.01_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
                'color': '#e377c2',
                'linestyle': '-',
                'marker': '*',
            },
        }
    },

    # Label-Permuted EMNIST
    'emnist': {
        'display_name': 'Label-Permuted EMNIST',
        'logs_subdir': 'label_permuted_emnist_stats',
        'steps_per_task': 2500,
        'n_samples': 1000000,
        'experiments': {
            'S&P': {
                'path': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.9_weight_decay_0.001_n_samples_1000000',
                'color': '#7f7f7f',
                'linestyle': '--',
                'marker': 'o',
            },
            'UPGD (Full)': {
                'path': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
                'color': '#1f77b4',
                'linestyle': '-',
                'marker': 's',
            },
            'UPGD (Output Only)': {
                'path': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
                'color': '#2ca02c',
                'linestyle': '-',
                'marker': '^',
            },
            'UPGD (Hidden Only)': {
                'path': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
                'color': '#ff7f0e',
                'linestyle': '-',
                'marker': 'D',
            },
            'UPGD (Hidden+Output)': {
                'path': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
                'color': '#9467bd',
                'linestyle': '-',
                'marker': 'v',
            },
            'UPGD (Clamped 0.52)': {
                'path': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_n_samples_1000000',
                'color': '#d62728',
                'linestyle': '-',
                'marker': 'p',
            },
            'UPGD (Clamped 0.48-0.52)': {
                'path': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
                'color': '#8c564b',
                'linestyle': '-',
                'marker': 'h',
            },
            'UPGD (Clamped 0.44-0.56)': {
                'path': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.9_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
                'color': '#e377c2',
                'linestyle': '-',
                'marker': '*',
            },
        }
    },

    # Label-Permuted CIFAR-10
    'cifar10': {
        'display_name': 'Label-Permuted CIFAR-10',
        'logs_subdir': 'label_permuted_cifar10_stats',
        'steps_per_task': 2500,
        'n_samples': 1000000,
        'experiments': {
            'S&P': {
                'path': 'sgd/fully_connected_relu_with_hooks/lr_0.01_sigma_0.01_beta_utility_0.999_weight_decay_0.001_n_samples_1000000',
                'color': '#7f7f7f',
                'linestyle': '--',
                'marker': 'o',
            },
            'UPGD (Full)': {
                'path': 'upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_n_samples_1000000',
                'color': '#1f77b4',
                'linestyle': '-',
                'marker': 's',
            },
            'UPGD (Output Only)': {
                'path': 'upgd_fo_global_outputonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_output_only_n_samples_1000000',
                'color': '#2ca02c',
                'linestyle': '-',
                'marker': '^',
            },
            'UPGD (Hidden Only)': {
                'path': 'upgd_fo_global_hiddenonly/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_hidden_only_n_samples_1000000',
                'color': '#ff7f0e',
                'linestyle': '-',
                'marker': 'D',
            },
            'UPGD (Hidden+Output)': {
                'path': 'upgd_fo_global_hiddenandoutput/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_gating_mode_hidden_and_output_n_samples_1000000',
                'color': '#9467bd',
                'linestyle': '-',
                'marker': 'v',
            },
            'UPGD (Clamped 0.52)': {
                'path': 'upgd_fo_global_clamped052/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_n_samples_1000000',
                'color': '#d62728',
                'linestyle': '-',
                'marker': 'p',
            },
            'UPGD (Clamped 0.48-0.52)': {
                'path': 'upgd_fo_global_clamped_48_52/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_min_clamp_0.48_max_clamp_0.52_n_samples_1000000',
                'color': '#8c564b',
                'linestyle': '-',
                'marker': 'h',
            },
            'UPGD (Clamped 0.44-0.56)': {
                'path': 'upgd_fo_global_clamped_44_56/fully_connected_relu_with_hooks/lr_0.01_sigma_0.001_beta_utility_0.999_weight_decay_0.0_min_clamp_0.44_max_clamp_0.56_n_samples_1000000',
                'color': '#e377c2',
                'linestyle': '-',
                'marker': '*',
            },
        }
    },
}

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

default_config = {
    # Data configuration
    'data': {
        'base_dir': str(PROJECT_DIR),
        'logs_dir': str(PROJECT_DIR / 'logs'),
        'cache_dir': str(PLOTS_DIR / 'cache'),
        'datasets': DATASET_CONFIGS,
    },

    # Statistical testing configuration
    'stats': {
        'default_alpha': 0.05,
        'default_correction': 'holm',  # 'bonferroni', 'holm', 'fdr', None
        'default_test': 'wilcoxon',    # 'ttest', 'wilcoxon', 'anova'
        'confidence_level': 0.95,
        'effect_size_type': 'cohen_d',  # 'cohen_d', 'cliff_delta'
    },

    # Plotting configuration
    'plotting': {
        'style': 'seaborn-v0_8-whitegrid',
        'dpi': 150,
        'pdf_fonttype': 42,  # Embedded fonts for PDFs
        'ps_fonttype': 42,
        'font_size': 12,

        # Figure sizes (width, height) in inches
        'figure_sizes': {
            'time_series': (12, 7),
            'comparison': (10, 6),
            'histogram': (14, 7),
            'heatmap': (10, 8),
        },

        # Method colors (consistent across all plots)
        'colors': {
            'S&P': '#7f7f7f',
            'UPGD (Full)': '#1f77b4',
            'UPGD (Output Only)': '#2ca02c',
            'UPGD (Hidden Only)': '#ff7f0e',
            'UPGD (Hidden+Output)': '#9467bd',
            'UPGD (Clamped 0.52)': '#d62728',
            'UPGD (Clamped 0.48-0.52)': '#8c564b',
            'UPGD (Clamped 0.44-0.56)': '#e377c2',
        },

        # Line styles
        'linestyles': {
            'S&P': '--',
            'UPGD (Full)': '-',
            'UPGD (Output Only)': '-',
            'UPGD (Hidden Only)': '-',
            'UPGD (Hidden+Output)': '-',
            'UPGD (Clamped 0.52)': '-',
            'UPGD (Clamped 0.48-0.52)': '-',
            'UPGD (Clamped 0.44-0.56)': '-',
        },

        # Markers
        'markers': {
            'S&P': 'o',
            'UPGD (Full)': 's',
            'UPGD (Output Only)': '^',
            'UPGD (Hidden Only)': 'D',
            'UPGD (Hidden+Output)': 'v',
            'UPGD (Clamped 0.52)': 'p',
            'UPGD (Clamped 0.48-0.52)': 'h',
            'UPGD (Clamped 0.44-0.56)': '*',
        },

        # Output configuration
        'output_dir': str(PLOTS_DIR / 'figures'),
        'formats': ['png', 'pdf'],  # Export formats
    },

    # Metric configurations
    'metrics': {
        'available': [
            'accuracy', 'loss', 'plasticity', 'n_dead_units',
            'weight_rank', 'weight_l2', 'weight_l1',
            'grad_l2', 'grad_l1', 'grad_l0'
        ],
        'display_names': {
            'accuracy': 'Accuracy',
            'loss': 'Loss',
            'plasticity': 'Plasticity',
            'n_dead_units': 'Dead Units (%)',
            'weight_rank': 'Weight Rank',
            'weight_l2': 'Weight L2 Norm',
            'weight_l1': 'Weight L1 Norm',
            'grad_l2': 'Gradient L2 Norm',
            'grad_l1': 'Gradient L1 Norm',
            'grad_l0': 'Gradient L0 Ratio',
        },
        'higher_is_better': {
            'accuracy': True,
            'loss': False,
            'plasticity': True,
            'n_dead_units': False,
            'weight_rank': True,
            'weight_l2': False,
            'weight_l1': False,
            'grad_l2': False,
            'grad_l1': False,
            'grad_l0': False,
        }
    },
}
