#!/usr/bin/env python3
"""
Quick test script to verify UPGD implementation works with incremental CIFAR-100.
This runs a minimal test (1 epoch, small batch) to catch any import or basic errors.
"""

import sys
import os
import torch
import json

# Add paths
sys.path.insert(0, '/scratch/gautschi/shin283/loss-of-plasticity')
sys.path.insert(0, '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar')

def test_upgd_import():
    """Test that UPGD can be imported"""
    print("=" * 60)
    print("Test 1: UPGD Import")
    print("=" * 60)
    try:
        from lop.algos.upgd import UPGD
        print("✓ UPGD imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import UPGD: {e}")
        return False

def test_upgd_optimizer():
    """Test that UPGD optimizer can be instantiated and used"""
    print("\n" + "=" * 60)
    print("Test 2: UPGD Optimizer Instantiation")
    print("=" * 60)
    try:
        from lop.algos.upgd import UPGD

        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )

        # Create UPGD optimizer
        optimizer = UPGD(
            model.parameters(),
            lr=0.01,
            weight_decay=0.0005,
            beta_utility=0.999,
            sigma=0.001
        )

        # Set parameter names
        optimizer.set_param_names(model.named_parameters())

        # Test forward/backward pass
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        # Test utility stats
        stats = optimizer.get_utility_stats()
        gating_stats = optimizer.get_gating_stats()

        print("✓ UPGD optimizer works correctly")
        print(f"  - Utility stats keys: {list(stats.keys())}")
        print(f"  - Gating stats keys: {list(gating_stats.keys())}")
        print(f"  - Global max utility: {stats.get('global_max_utility', 0.0):.6f}")
        print(f"  - Mean gate value: {gating_stats.get('mean_gate_value', 0.0):.4f}")
        return True
    except Exception as e:
        print(f"✗ Failed to use UPGD optimizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test that UPGD config files can be loaded"""
    print("\n" + "=" * 60)
    print("Test 3: Config File Loading")
    print("=" * 60)

    config_files = [
        '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_baseline.json',
        '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_with_cbp.json'
    ]

    all_passed = True
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            assert config.get('use_upgd') == True, "use_upgd should be True"
            assert 'upgd_beta_utility' in config, "upgd_beta_utility missing"
            assert 'upgd_sigma' in config, "upgd_sigma missing"

            print(f"✓ Config loaded: {os.path.basename(config_file)}")
            print(f"  - beta_utility: {config['upgd_beta_utility']}")
            print(f"  - sigma: {config['upgd_sigma']}")
            print(f"  - use_cbp: {config.get('use_cbp', False)}")
        except Exception as e:
            print(f"✗ Failed to load {config_file}: {e}")
            all_passed = False

    return all_passed

def test_experiment_initialization():
    """Test that the experiment can be initialized with UPGD"""
    print("\n" + "=" * 60)
    print("Test 4: Experiment Initialization (may take a moment...)")
    print("=" * 60)

    try:
        from incremental_cifar_experiment import IncrementalCIFARExperiment

        # Load UPGD config
        config_file = '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/cfg/upgd_baseline.json'
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Set paths
        config['data_path'] = '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data'
        config['results_dir'] = '/tmp/upgd_test_results'
        config['num_epochs'] = 1  # Override for quick test

        # Create experiment
        exp = IncrementalCIFARExperiment(
            exp_params=config,
            results_dir=config['results_dir'],
            run_index=0,
            verbose=False
        )

        # Check that UPGD optimizer was created
        from lop.algos.upgd import UPGD
        assert isinstance(exp.optim, UPGD), f"Expected UPGD optimizer, got {type(exp.optim)}"

        print("✓ Experiment initialized successfully with UPGD")
        print(f"  - Optimizer type: {type(exp.optim).__name__}")
        print(f"  - Network: {type(exp.net).__name__}")
        print(f"  - Device: {exp.device}")

        # Clean up
        import shutil
        if os.path.exists('/tmp/upgd_test_results'):
            shutil.rmtree('/tmp/upgd_test_results')

        return True
    except Exception as e:
        print(f"✗ Failed to initialize experiment: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("UPGD Incremental CIFAR-100 Integration Tests")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("UPGD Import", test_upgd_import()))
    results.append(("UPGD Optimizer", test_upgd_optimizer()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Experiment Init", test_experiment_initialization()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        print("UPGD is ready for full experiments.")
    else:
        print("Some tests failed! ✗")
        print("Please fix the issues before running full experiments.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
