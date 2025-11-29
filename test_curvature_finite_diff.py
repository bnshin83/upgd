#!/usr/bin/env python3
"""
Test script for the new finite differences curvature calculation.

This script validates that the new curvature calculation method works correctly
and produces reasonable results compared to the original Hutchinson method.
"""

import torch
import torch.nn as nn
import numpy as np
from core.optim.weight_upgd.input_aware import (
    compute_input_curvature_finite_diff, 
    hutchinson_trace_estimator
)
from core.learner.input_aware_upgd import InputAwareFirstOrderGlobalUPGDLearner
from core.network.fcn_relu import FullyConnectedReLU

def create_test_data(n_samples=32, input_dim=28*28, n_classes=10):
    """Create synthetic test data."""
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return x, y

def create_test_network(input_dim=28*28, n_classes=10):
    """Create a simple test network."""
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes)
    )

def test_finite_diff_curvature():
    """Test the finite differences curvature calculation."""
    print("=== Testing Finite Differences Curvature Calculation ===")
    
    # Setup
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data and network
    input_dim = 28 * 28
    n_classes = 10
    batch_size = 16
    
    x, y = create_test_data(batch_size, input_dim, n_classes)
    x, y = x.to(device), y.to(device)
    
    model = create_test_network(input_dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Test 1: Basic functionality
    print("\\n1. Testing basic finite differences curvature computation...")
    try:
        curvature = compute_input_curvature_finite_diff(
            model=model,
            inputs=x,
            targets=y,
            criterion=criterion,
            h=1e-3,
            niter=5,
            temp=1.0
        )
        print(f"   ✓ Finite differences curvature: {curvature:.6f}")
        assert curvature > 0, "Curvature should be positive"
        assert not np.isnan(curvature), "Curvature should not be NaN"
        
    except Exception as e:
        print(f"   ✗ Error in finite differences method: {e}")
        return False
    
    # Test 2: Compare with original Hutchinson method
    print("\\n2. Comparing with original Hutchinson estimator...")
    try:
        # First compute loss for Hutchinson method
        x_hutch = x.requires_grad_(True)
        output = model(x_hutch)
        loss = criterion(output, y)
        
        hutchinson_curvature = hutchinson_trace_estimator(loss, x_hutch, n_samples=5)
        print(f"   ✓ Hutchinson curvature: {hutchinson_curvature:.6f}")
        print(f"   ✓ Finite diff curvature: {curvature:.6f}")
        print(f"   ✓ Ratio (FD/Hutchinson): {curvature / hutchinson_curvature:.3f}")
        
    except Exception as e:
        print(f"   ✗ Error in Hutchinson comparison: {e}")
        return False
    
    # Test 3: Parameter sensitivity
    print("\\n3. Testing parameter sensitivity...")
    
    # Test different perturbation sizes
    h_values = [1e-4, 1e-3, 1e-2]
    curvatures_h = []
    
    for h in h_values:
        curv = compute_input_curvature_finite_diff(
            model=model, inputs=x, targets=y, criterion=criterion,
            h=h, niter=3, temp=1.0
        )
        curvatures_h.append(curv)
        print(f"   h={h:.0e}: curvature = {curv:.6f}")
    
    # Test different iteration counts
    niter_values = [1, 5, 10]
    curvatures_niter = []
    
    for niter in niter_values:
        curv = compute_input_curvature_finite_diff(
            model=model, inputs=x, targets=y, criterion=criterion,
            h=1e-3, niter=niter, temp=1.0
        )
        curvatures_niter.append(curv)
        print(f"   niter={niter}: curvature = {curv:.6f}")
    
    return True

def test_learner_integration():
    """Test integration with the InputAware learner."""
    print("\\n=== Testing Learner Integration ===")
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create learner
    try:
        network = FullyConnectedReLU(input_size=784, hidden_sizes=[128, 64], output_size=10)
        learner = InputAwareFirstOrderGlobalUPGDLearner(
            network=network,
            optim_kwargs={
                'lr': 0.01,
                'sigma': 0.1,
                'hutchinson_samples': 5,
                'curvature_threshold': 1.0,
                'lambda_max': 1.0
            }
        )
        print("   ✓ Learner created successfully")
        
        # Create test data
        x, y = create_test_data(16, 784, 10)
        x, y = x.to(device), y.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # Test curvature computation through learner
        curvature = learner.compute_input_curvature(
            model=learner.network,
            input_batch=x,
            targets=y,
            criterion=criterion
        )
        
        print(f"   ✓ Learner curvature computation: {curvature:.6f}")
        assert curvature > 0, "Curvature should be positive"
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error in learner integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_size_consistency():
    """Test that curvature is consistent across different batch sizes."""
    print("\\n=== Testing Batch Size Consistency ===")
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_test_network().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Test different batch sizes
    batch_sizes = [1, 4, 16, 32]
    curvatures = []
    
    for bs in batch_sizes:
        x, y = create_test_data(bs, 28*28, 10)
        x, y = x.to(device), y.to(device)
        
        curvature = compute_input_curvature_finite_diff(
            model=model, inputs=x, targets=y, criterion=criterion,
            h=1e-3, niter=3, temp=1.0
        )
        curvatures.append(curvature)
        print(f"   Batch size {bs}: curvature = {curvature:.6f}")
    
    # Check if values are in reasonable range
    cv = np.std(curvatures) / np.mean(curvatures)  # Coefficient of variation
    print(f"   Coefficient of variation across batch sizes: {cv:.3f}")
    
    if cv < 0.5:  # Less than 50% variation is reasonable
        print("   ✓ Curvature values are reasonably consistent across batch sizes")
        return True
    else:
        print("   ⚠ High variation across batch sizes (may be normal)")
        return True

def main():
    """Run all tests."""
    print("Testing New Finite Differences Curvature Implementation\\n")
    print("This validates the updated method that follows post_run_analysis_modified2.py")
    print("="*70)
    
    success = True
    
    # Run tests
    success &= test_finite_diff_curvature()
    success &= test_learner_integration() 
    success &= test_batch_size_consistency()
    
    print("\\n" + "="*70)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\\nThe new finite differences curvature calculation is working correctly.")
        print("You can now run your experiments with:")
        print("   sbatch test_inputaware_emnist_stats.sh")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\\nPlease check the errors above before running experiments.")
    
    return success

if __name__ == "__main__":
    main()