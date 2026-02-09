# UPGD Project Context

## Goal
Investigate utility-gated perturbations for continual learning, demonstrating that layer-selective gating (especially output-only) achieves strong performance improvements.

## Key Research Questions
1. Does UPGD-W with output-only gating outperform baseline methods (Shrink & Perturb)?
2. What is the relationship between gating strategy and task regime (input-shift vs output-shift)?
3. Can RL experiments validate regime theory on complex control tasks?

## Target Venues
- Primary: UAI 2026 (current focus)
- Related: ICML 2025 paper in main research project

## Key Findings
- Output-only gating: Up to 104% improvement over Shrink & Perturb
- Ant-v4 (RL): Input-shift regime - hidden-only (4843) >> output-only (3229)
- Humanoid-v4 (RL): Currently running - 80 tasks across 2 clusters

## Methods Compared
- UPGD variants: full gating, output-only, hidden-only
- Baselines: Adam, SGD, Shrink & Perturb, EWC, SI, MAS

## Datasets/Environments
- Supervised: Input-permuted MNIST, Label-permuted EMNIST, CIFAR-10, Mini-ImageNet
- RL: Ant-v4, Humanoid-v4 (MuJoCo continuous control)
