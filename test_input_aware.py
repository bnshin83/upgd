#!/usr/bin/env python3
"""
Test script for input-aware UPGD implementation.
This script verifies that the input-aware optimization works correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.optim.weight_upgd.input_aware import (
    InputAwareFirstOrderGlobalUPGD,
    hutchinson_trace_estimator
)
from core.network.fcn_relu import FullyConnectedReLU


def test_hutchinson_estimator():
    """Test the Hutchinson trace estimator."""
    print("Testing Hutchinson trace estimator...")
    
    # Create a simple quadratic loss: L = 0.5 * x^T * A * x
    # where A is a known matrix, so we can verify the trace
    dim = 10
    A = torch.randn(dim, dim)
    A = A @ A.T  # Make it symmetric positive definite
    x = torch.randn(dim, requires_grad=True)
    
    # Loss function
    loss = 0.5 * x @ A @ x
    
    # Compute trace using Hutchinson estimator
    estimated_trace = hutchinson_trace_estimator(loss, x, n_samples=100)
    
    # The Hessian of this loss is A, so tr(H^2) = tr(A^2)
    true_trace = torch.trace(A @ A).item()
    
    print(f"  True tr(H^2): {true_trace:.4f}")
    print(f"  Estimated tr(H^2): {estimated_trace:.4f}")
    print(f"  Relative error: {abs(estimated_trace - true_trace) / true_trace * 100:.2f}%")
    
    assert abs(estimated_trace - true_trace) / true_trace < 0.5, "Hutchinson estimator error too large"
    print("  ✓ Hutchinson estimator test passed\n")


def test_input_aware_optimizer():
    """Test the input-aware UPGD optimizer."""
    print("Testing Input-Aware UPGD optimizer...")
    
    # Create a simple network
    torch.manual_seed(42)
    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    # Create optimizer
    optimizer = InputAwareFirstOrderGlobalUPGD(
        model.named_parameters(),
        lr=0.01,
        beta_utility=0.99,
        sigma=0.1,
        curvature_threshold=1.0,
        lambda_max=0.5,
        lambda_scale=1.0,
        beta_curvature=0.9
    )
    
    # Test training step with varying curvature
    for i in range(5):
        # Create dummy data
        x = torch.randn(32, input_dim)
        y = torch.randint(0, output_dim, (32,))
        
        # Forward pass
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        # Simulate different curvature values
        if i < 2:
            # Low curvature (easy samples)
            optimizer.set_input_curvature(0.1)
            print(f"  Step {i+1}: Low curvature (0.1) - λ = {optimizer.compute_lambda():.4f}")
        else:
            # High curvature (hard samples)
            optimizer.set_input_curvature(10.0)
            print(f"  Step {i+1}: High curvature (10.0) - λ = {optimizer.compute_lambda():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        print(f"    Loss: {loss.item():.4f}")
    
    print("  ✓ Input-aware optimizer test passed\n")


def test_integration():
    """Test the full integration with a simple task."""
    print("Testing full integration...")
    
    from core.learner.input_aware_upgd import InputAwareFirstOrderGlobalUPGDLearner
    from core.network.fcn_relu import FullyConnectedReLU
    
    # Create learner
    learner = InputAwareFirstOrderGlobalUPGDLearner(
        network=FullyConnectedReLU(),
        optim_kwargs={
            'lr': 0.01,
            'beta_utility': 0.99,
            'sigma': 0.1,
            'curvature_threshold': 1.0,
            'lambda_max': 0.5,
            'hutchinson_samples': 1
        }
    )
    
    print("  Learner created: ", learner.name)
    
    # Create a mock task
    class MockTask:
        def __init__(self):
            self.n_inputs = 784
            self.n_outputs = 10
            self.criterion = 'cross_entropy'
    
    task = MockTask()
    learner.set_task(task)
    
    # Create optimizer
    optimizer = learner.optimizer(
        learner.parameters, **learner.optim_kwargs
    )
    learner.optimizer_instance = optimizer
    
    # Test a training step
    x = torch.randn(32, 784).to(learner.device)
    y = torch.randint(0, 10, (32,)).to(learner.device)
    
    # Forward pass
    output = learner.predict(x)
    loss = F.cross_entropy(output, y)
    
    # Compute input curvature (with gradient tracking)
    x_grad = x.requires_grad_(True)
    output_grad = learner.network(x_grad)
    loss_grad = F.cross_entropy(output_grad, y)
    curvature = learner.compute_input_curvature(loss_grad, x_grad)
    
    print(f"  Computed input curvature: {curvature:.4f}")
    
    # Update optimizer with curvature
    learner.update_optimizer_curvature(curvature)
    
    # Standard backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Loss after step: {loss.item():.4f}")
    print("  ✓ Integration test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Input-Aware UPGD Implementation Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_hutchinson_estimator()
        test_input_aware_optimizer()
        test_integration()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe input-aware UPGD implementation is working correctly.")
        print("You can now run the full experiments using:")
        print("  python experiments/label_permuted_emnist_input_aware.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
