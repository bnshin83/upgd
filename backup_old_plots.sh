#!/bin/bash
# Backup old plots (before Jan 15, 2026)

BACKUP_DIR="/scratch/gautschi/shin283/upgd/upgd_plots/figures_backup_20260115"

# Mini-ImageNet
echo "Processing mini_imagenet..."
ls -lt /scratch/gautschi/shin283/upgd/upgd_plots/figures/mini_imagenet/*.png | grep -v "Jan 15" | awk '{print $NF}' | while read file; do
    mv "$file" "$BACKUP_DIR/mini_imagenet/"
done
echo "Moved $(ls $BACKUP_DIR/mini_imagenet/*.png 2>/dev/null | wc -l) plots from mini_imagenet"

# Input-Permuted MNIST
echo "Processing input_mnist..."
ls -lt /scratch/gautschi/shin283/upgd/upgd_plots/figures/input_mnist/*.png | grep -v "Jan 15" | awk '{print $NF}' | while read file; do
    mv "$file" "$BACKUP_DIR/input_mnist/"
done
echo "Moved $(ls $BACKUP_DIR/input_mnist/*.png 2>/dev/null | wc -l) plots from input_mnist"

# EMNIST
echo "Processing emnist..."
ls -lt /scratch/gautschi/shin283/upgd/upgd_plots/figures/emnist/*.png | grep -v "Jan 15" | awk '{print $NF}' | while read file; do
    mv "$file" "$BACKUP_DIR/emnist/"
done
echo "Moved $(ls $BACKUP_DIR/emnist/*.png 2>/dev/null | wc -l) plots from emnist"

# CIFAR-10
echo "Processing cifar10..."
ls -lt /scratch/gautschi/shin283/upgd/upgd_plots/figures/cifar10/*.png | grep -v "Jan 15" | awk '{print $NF}' | while read file; do
    mv "$file" "$BACKUP_DIR/cifar10/"
done
echo "Moved $(ls $BACKUP_DIR/cifar10/*.png 2>/dev/null | wc -l) plots from cifar10"

echo ""
echo "Backup complete!"
echo "Old plots saved to: $BACKUP_DIR"
echo ""
echo "Summary:"
echo "  mini_imagenet: $(ls /scratch/gautschi/shin283/upgd/upgd_plots/figures/mini_imagenet/*.png 2>/dev/null | wc -l) plots remaining"
echo "  input_mnist: $(ls /scratch/gautschi/shin283/upgd/upgd_plots/figures/input_mnist/*.png 2>/dev/null | wc -l) plots remaining"
echo "  emnist: $(ls /scratch/gautschi/shin283/upgd/upgd_plots/figures/emnist/*.png 2>/dev/null | wc -l) plots remaining"
echo "  cifar10: $(ls /scratch/gautschi/shin283/upgd/upgd_plots/figures/cifar10/*.png 2>/dev/null | wc -l) plots remaining"
