#!/usr/bin/env python3
"""
Quick integration test for the curvature tracking system.
"""

import torch
import torch.nn as nn
import sys
import os

# Add path
sys.path.insert(0, '/scratch/gautschi/shin283/upgd')

def test_basic_integration():
    """Test basic integration of the curvature system."""
    print("Testing basic integration...")
    
    try:
        # Test imports
        from core.optim.weight_upgd.input_aware import compute_input_curvature_finite_diff
        from core.learner.input_aware_upgd import InputAwareFirstOrderGlobalUPGDLearner
        from core.network.fcn_relu import FullyConnectedReLU
        print("✓ Imports successful")
        
        # Test basic curvature computation
        device = torch.device('cpu')  # Use CPU for testing
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)
        
        # Create test data (single sample as in real usage)
        x = torch.randn(1, 784, device=device)  # Single sample
        y = torch.randint(0, 10, (1,), device=device)
        criterion = nn.CrossEntropyLoss()
        
        # Test finite differences curvature
        curvatures = compute_input_curvature_finite_diff(
            model=model,
            inputs=x,
            targets=y,
            criterion=criterion,
            h=1e-3,
            niter=2,  # Small for testing
            temp=1.0
        )
        
        print(f"✓ Curvature computation successful: {curvatures} (shape: {curvatures.shape})")
        assert curvatures.shape == (1,), f"Expected shape (1,), got {curvatures.shape}"
        assert curvatures[0] >= 0, f"Curvature should be non-negative, got {curvatures[0]}"
        
        # Test learner integration
        from core.utils import tasks
        task = tasks['label_permuted_emnist_stats']()
        
        learner = InputAwareFirstOrderGlobalUPGDLearner(
            network=FullyConnectedReLU,  # Pass class, not instance
            optim_kwargs={
                'lr': 0.01,
                'sigma': 0.1,
                'hutchinson_samples': 2,
                'curvature_threshold': 1.0,
                'lambda_max': 1.0
            }
        )
        
        # Set task to initialize network
        learner.set_task(task)
        
        # Test learner curvature computation (should return scalar by default)
        learner_curvature = learner.compute_input_curvature(
            model=learner.network,
            input_batch=x,
            targets=y,
            criterion=criterion
        )
        
        print(f"✓ Learner curvature computation successful: {learner_curvature}")
        import numpy as np
        assert isinstance(learner_curvature, (float, int, np.floating, np.integer)), f"Expected scalar, got {type(learner_curvature)}"
        assert learner_curvature >= 0, f"Curvature should be non-negative, got {learner_curvature}"
        
        print("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_integration()
    if success:
        print("\n🎉 Integration test passed! The script should be ready to run.")
    else:
        print("\n⚠️  Integration test failed. Please check the errors above.")