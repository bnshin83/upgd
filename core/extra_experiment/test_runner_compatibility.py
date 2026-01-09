#!/usr/bin/env python3
"""
Test script to verify that run_stats_with_curvature.py works correctly 
with both input-aware and regular learners.
"""

import sys
import os
sys.path.insert(1, os.getcwd())

from core.run.run_stats_with_curvature import RunStatsWithCurvature

def test_learner_compatibility():
    """Test that the runner works with different learner types."""
    
    print("="*60)
    print("Testing Enhanced Runner Compatibility")
    print("="*60)
    
    # Test parameters - use minimal samples for quick testing
    test_configs = [
        {
            'name': 'Input-Aware UPGD',
            'learner': 'input_aware_upgd_fo_global',
            'expected_curvature': True
        },
        {
            'name': 'Regular UPGD', 
            'learner': 'upgd_fo_global',
            'expected_curvature': False
        },
        {
            'name': 'Adam',
            'learner': 'adam',
            'expected_curvature': False
        }
    ]
    
    common_params = {
        'task': 'input_permuted_mnist_stats',
        'network': 'fully_connected_relu_with_hooks', 
        'n_samples': 100,  # Very small for testing
        'seed': 42,
        'lr': 0.01,
        'sigma': 0.001,
        'save_path': 'test_logs',
        'compute_curvature_every': 10,
        # Input-aware specific params (will be ignored by regular learners)
        'curvature_threshold': 0.05,
        'lambda_max': 1.0,
        'hutchinson_samples': 3
    }
    
    results = []
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} ---")
        
        try:
            # Create runner with specific learner
            runner = RunStatsWithCurvature(
                learner=config['learner'],
                **common_params
            )
            
            # Verify input-aware detection
            expected = config['expected_curvature']
            actual = runner.is_input_aware
            
            print(f"  Learner: {config['learner']}")
            print(f"  Expected input-aware: {expected}")
            print(f"  Detected input-aware: {actual}")
            
            # Check if learner has curvature computation method
            has_compute_method = hasattr(runner.learner, 'compute_input_curvature')
            print(f"  Has compute_input_curvature method: {has_compute_method}")
            
            # For input-aware learners, verify method exists
            if expected and not has_compute_method:
                print(f"  WARNING: Expected input-aware learner but missing method!")
                results.append(f"FAIL - {config['name']}: Missing compute_input_curvature method")
            elif expected and actual and has_compute_method:
                print(f"  SUCCESS: Input-aware learner correctly detected and configured")
                results.append(f"PASS - {config['name']}: Correct input-aware detection")
            elif not expected and not actual:
                print(f"  SUCCESS: Regular learner correctly detected")  
                results.append(f"PASS - {config['name']}: Correct regular learner detection")
            else:
                print(f"  WARNING: Detection mismatch!")
                results.append(f"WARN - {config['name']}: Detection mismatch")
                
            # Brief functionality test - just initialize, don't run full experiment
            print(f"  Configuration successful: ✓")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(f"ERROR - {config['name']}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Results Summary:")
    print(f"{'='*60}")
    for result in results:
        print(f"  {result}")
    
    # Count results
    passes = len([r for r in results if r.startswith('PASS')])
    total = len(results) 
    print(f"\nOverall: {passes}/{total} tests passed")
    
    if passes == total:
        print("🎉 All compatibility tests passed!")
        return True
    else:
        print("⚠️  Some tests failed or had warnings")
        return False

if __name__ == "__main__":
    success = test_learner_compatibility()
    sys.exit(0 if success else 1)