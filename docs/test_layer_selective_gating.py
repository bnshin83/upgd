#!/usr/bin/env python3.8
"""
Test script for layer-selective UPGD gating on incremental CIFAR-100.

Tests:
1. UPGD optimizer initialization with different gating modes
2. Layer name detection (ResNet18)
3. Gating application logic
4. Parameter counting for gated vs non-gated layers
"""

import sys
sys.path.insert(0, '/scratch/gautschi/shin283/loss-of-plasticity')

import torch
from lop.algos.upgd import UPGD
from lop.nets.torchvision_modified_resnet import build_resnet18

def test_gating_modes():
    """Test UPGD initialization with different gating modes."""
    print("=" * 60)
    print("Test 1: UPGD Initialization with Different Gating Modes")
    print("=" * 60)

    # Create a simple model
    model = build_resnet18(num_classes=100)

    gating_modes = ['full', 'output_only', 'hidden_only']

    for mode in gating_modes:
        print(f"\nTesting gating_mode='{mode}'...")
        try:
            optimizer = UPGD(
                model.parameters(),
                lr=0.1,
                gating_mode=mode,
                non_gated_scale=0.5
            )
            optimizer.set_param_names(model.named_parameters())
            print(f"  ✓ UPGD with gating_mode='{mode}' initialized successfully")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False

    return True


def test_layer_names():
    """Test ResNet18 layer naming and print all parameter names."""
    print("\n" + "=" * 60)
    print("Test 2: ResNet18 Layer Names")
    print("=" * 60)

    model = build_resnet18(num_classes=100)

    print("\nAll parameter names in ResNet18:")
    print("-" * 60)
    for name, param in model.named_parameters():
        print(f"  {name:40s} shape={list(param.shape)}")

    print("\n" + "-" * 60)
    print("Final layer parameters (fc.*):")
    for name, param in model.named_parameters():
        if name.startswith('fc.'):
            print(f"  {name:40s} shape={list(param.shape)}")

    return True


def test_gating_logic():
    """Test _should_apply_gating logic for each gating mode."""
    print("\n" + "=" * 60)
    print("Test 3: Gating Logic for Each Mode")
    print("=" * 60)

    model = build_resnet18(num_classes=100)

    # Test each gating mode
    gating_modes = ['full', 'output_only', 'hidden_only']

    for mode in gating_modes:
        print(f"\n{mode.upper()} MODE:")
        print("-" * 60)

        optimizer = UPGD(
            model.parameters(),
            lr=0.1,
            gating_mode=mode,
            non_gated_scale=0.5
        )
        optimizer.set_param_names(model.named_parameters())

        # Sample parameters to test
        test_params = [
            'conv1.weight',
            'layer1.0.conv1.weight',
            'layer4.1.bn2.bias',
            'fc.weight',
            'fc.bias'
        ]

        for param_name in test_params:
            should_gate = optimizer._should_apply_gating(param_name, mode)
            status = "GATED" if should_gate else "NON-GATED"
            print(f"  {param_name:30s} -> {status}")

    return True


def test_parameter_counts():
    """Count gated vs non-gated parameters for each mode."""
    print("\n" + "=" * 60)
    print("Test 4: Parameter Counts (Gated vs Non-Gated)")
    print("=" * 60)

    model = build_resnet18(num_classes=100)

    gating_modes = ['full', 'output_only', 'hidden_only']

    for mode in gating_modes:
        print(f"\n{mode.upper()} MODE:")
        print("-" * 60)

        optimizer = UPGD(
            model.parameters(),
            lr=0.1,
            gating_mode=mode,
            non_gated_scale=0.5
        )
        optimizer.set_param_names(model.named_parameters())

        gated_count = 0
        non_gated_count = 0

        for name, param in model.named_parameters():
            should_gate = optimizer._should_apply_gating(name, mode)
            if should_gate:
                gated_count += param.numel()
            else:
                non_gated_count += param.numel()

        total_params = gated_count + non_gated_count
        gated_pct = (gated_count / total_params) * 100
        non_gated_pct = (non_gated_count / total_params) * 100

        print(f"  Gated parameters:     {gated_count:,} ({gated_pct:.2f}%)")
        print(f"  Non-gated parameters: {non_gated_count:,} ({non_gated_pct:.2f}%)")
        print(f"  Total parameters:     {total_params:,}")

    return True


def test_forward_backward_pass():
    """Test that optimizer works with forward/backward pass."""
    print("\n" + "=" * 60)
    print("Test 5: Forward/Backward Pass with UPGD")
    print("=" * 60)

    model = build_resnet18(num_classes=100)

    gating_modes = ['full', 'output_only', 'hidden_only']

    for mode in gating_modes:
        print(f"\nTesting {mode} mode...")

        try:
            optimizer = UPGD(
                model.parameters(),
                lr=0.1,
                gating_mode=mode,
                non_gated_scale=0.5
            )
            optimizer.set_param_names(model.named_parameters())

            # Create dummy data
            x = torch.randn(4, 3, 32, 32)
            y = torch.randint(0, 100, (4,))

            # Forward pass
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Get statistics
            gating_stats = optimizer.get_gating_stats()

            print(f"  ✓ Forward/backward pass successful")
            print(f"    Gating mode: {gating_stats['gating_mode']}")
            print(f"    Gated params: {gating_stats['gated_params']:,}")
            print(f"    Non-gated params: {gating_stats['non_gated_params']:,}")
            print(f"    Mean gate value: {gating_stats['mean_gate_value']:.4f}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("UPGD Layer-Selective Gating Test Suite")
    print("=" * 60)

    tests = [
        ("Gating Modes", test_gating_modes),
        ("Layer Names", test_layer_names),
        ("Gating Logic", test_gating_logic),
        ("Parameter Counts", test_parameter_counts),
        ("Forward/Backward Pass", test_forward_backward_pass),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name:30s} {status}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
