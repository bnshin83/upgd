#!/usr/bin/env python3
"""
Standalone Curvature Calculation Example

This script demonstrates how input curvature is calculated step-by-step,
showing the mathematical concepts in action.

Run this to understand the mechanics before using the full training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def simple_hutchinson_estimator(loss, inputs, n_samples=10):
    """
    Standalone implementation of Hutchinson trace estimator.
    
    Computes: tr(H_x^2) ≈ E_v[||H_x v||^2]
    where H_x is the Hessian of loss w.r.t. inputs
    """
    trace_estimate = 0.0
    
    print(f"Computing Hutchinson estimate with {n_samples} samples...")
    
    for i in range(n_samples):
        # Step 1: Sample random vector v ~ N(0, I)
        v = torch.randn_like(inputs, requires_grad=False)
        
        # Step 2: Compute gradient of loss w.r.t. inputs
        grad_loss = torch.autograd.grad(loss, inputs, create_graph=True, retain_graph=True)[0]
        
        # Step 3: Compute Hessian-vector product H_x @ v
        hvp = torch.autograd.grad(grad_loss, inputs, grad_outputs=v, retain_graph=True)[0]
        
        # Step 4: Compute ||H_x v||^2
        hvp_squared_norm = (hvp * hvp).sum().item()
        trace_estimate += hvp_squared_norm
        
        print(f"  Sample {i+1}: ||H_x v||^2 = {hvp_squared_norm:.6f}")
    
    trace_estimate /= n_samples
    print(f"Final estimate: tr(H_x^2) ≈ {trace_estimate:.6f}")
    
    return trace_estimate

def create_test_network():
    """Create a simple network for demonstration."""
    return nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

def generate_test_data(n_samples=100):
    """Generate test data with different curvature characteristics."""
    # Easy samples: well-separated clusters
    easy_x = torch.cat([
        torch.randn(n_samples//2, 2) + torch.tensor([2.0, 2.0]),  # Cluster 1
        torch.randn(n_samples//2, 2) + torch.tensor([-2.0, -2.0])  # Cluster 2
    ])
    easy_y = torch.cat([torch.ones(n_samples//2), torch.zeros(n_samples//2)])
    
    # Hard samples: overlapping/boundary samples
    hard_x = torch.cat([
        torch.randn(n_samples//4, 2) * 0.5 + torch.tensor([0.5, 0.5]),   # Near boundary
        torch.randn(n_samples//4, 2) * 0.5 + torch.tensor([-0.5, -0.5]), # Near boundary
        torch.randn(n_samples//4, 2) * 0.1 + torch.tensor([0.0, 0.0]),   # Right on boundary
        torch.randn(n_samples//4, 2) * 2.0 + torch.tensor([3.0, -3.0])   # Outliers
    ])
    hard_y = torch.cat([torch.ones(n_samples//2), torch.zeros(n_samples//2)])
    
    return easy_x, easy_y, hard_x, hard_y

def demonstrate_curvature_calculation():
    """Demonstrate curvature calculation on different types of samples."""
    print("=== Curvature Calculation Demonstration ===\\n")
    
    # Create network and data
    network = create_test_network()
    easy_x, easy_y, hard_x, hard_y = generate_test_data()
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Test on easy vs hard samples
    print("1. EASY SAMPLES (well-separated clusters)")
    easy_curvatures = []
    
    for i in range(5):  # Test 5 easy samples
        x = easy_x[i:i+1].requires_grad_(True)
        y = easy_y[i:i+1]
        
        output = network(x)
        loss = criterion(output, y.unsqueeze(1))
        
        print(f"\\nEasy sample {i+1}:")
        print(f"  Input: [{x[0, 0].item():.3f}, {x[0, 1].item():.3f}]")
        print(f"  Output: {output.item():.3f}, Target: {y.item():.0f}")
        print(f"  Loss: {loss.item():.6f}")
        
        curvature = simple_hutchinson_estimator(loss, x, n_samples=3)
        easy_curvatures.append(curvature)
    
    print("\\n" + "="*50)
    print("2. HARD SAMPLES (boundary/overlapping regions)")
    hard_curvatures = []
    
    for i in range(5):  # Test 5 hard samples
        x = hard_x[i:i+1].requires_grad_(True)
        y = hard_y[i:i+1]
        
        output = network(x)
        loss = criterion(output, y.unsqueeze(1))
        
        print(f"\\nHard sample {i+1}:")
        print(f"  Input: [{x[0, 0].item():.3f}, {x[0, 1].item():.3f}]")
        print(f"  Output: {output.item():.3f}, Target: {y.item():.0f}")
        print(f"  Loss: {loss.item():.6f}")
        
        curvature = simple_hutchinson_estimator(loss, x, n_samples=3)
        hard_curvatures.append(curvature)
    
    # Compare results
    print("\\n" + "="*50)
    print("3. COMPARISON")
    print(f"Easy samples - Average curvature: {np.mean(easy_curvatures):.6f} ± {np.std(easy_curvatures):.6f}")
    print(f"Hard samples - Average curvature: {np.mean(hard_curvatures):.6f} ± {np.std(hard_curvatures):.6f}")
    print(f"Ratio (hard/easy): {np.mean(hard_curvatures) / np.mean(easy_curvatures):.2f}x")
    
    return easy_curvatures, hard_curvatures

def demonstrate_lambda_mapping():
    """Show how input curvature maps to protection strength."""
    print("\\n=== Lambda Mapping Demonstration ===\\n")
    
    # Test different curvature values
    curvature_values = np.logspace(-3, 2, 50)  # 0.001 to 100
    
    # Parameters (matching your config)
    threshold = 0.07
    lambda_max = 1.0
    lambda_scale = 1.0
    
    lambda_values = []
    for curvature in curvature_values:
        # Sigmoid mapping: λ(x) = λ_max * sigmoid((tr(H_x^2) - τ) / α)
        normalized = (curvature - threshold) / lambda_scale
        lambda_val = lambda_max * torch.sigmoid(torch.tensor(normalized)).item()
        lambda_values.append(lambda_val)
    
    # Plot the mapping
    plt.figure(figsize=(10, 6))
    plt.semilogx(curvature_values, lambda_values, 'b-', linewidth=2)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.axhline(y=lambda_max/2, color='g', linestyle='--', label=f'λ_max/2 = {lambda_max/2}')
    plt.xlabel('Input Curvature tr(H_x²)')
    plt.ylabel('Protection Strength λ(x)')
    plt.title('Curvature to Protection Mapping')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, lambda_max * 1.1)
    
    # Print key values
    print(f"Threshold curvature: {threshold}")
    print(f"Lambda at threshold: {lambda_max * torch.sigmoid(torch.tensor(0.0)).item():.6f}")
    print(f"Low curvature (0.001): λ = {lambda_max * torch.sigmoid(torch.tensor((0.001 - threshold) / lambda_scale)).item():.6f}")
    print(f"High curvature (10.0): λ = {lambda_max * torch.sigmoid(torch.tensor((10.0 - threshold) / lambda_scale)).item():.6f}")
    
    plt.tight_layout()
    plt.savefig('lambda_mapping.png', dpi=300, bbox_inches='tight')
    print("\\nLambda mapping plot saved as 'lambda_mapping.png'")
    
    return curvature_values, lambda_values

def show_implementation_details():
    """Show exactly how this integrates with the training loop."""
    print("\\n=== Integration with Training Loop ===\\n")
    
    print("In the training loop (run_stats_with_curvature.py):")
    print("""
# Step 1: Ensure input requires gradients
input.requires_grad_(True)

# Step 2: Forward pass and loss computation  
output = learner.predict(input)
loss = criterion(output, target)

# Step 3: Compute input curvature
if step % compute_curvature_every == 0:
    curvature = learner.compute_input_curvature(loss, input)
    
    # Step 4: Update optimizer with curvature
    learner.update_optimizer_curvature(curvature)
    
    # Step 5: Get dynamic lambda
    lambda_val = optimizer.compute_lambda()
    
    # Step 6: Log values
    print(f"Step {step}: Curvature = {curvature:.6f}, Lambda = {lambda_val:.6f}")

# Step 7: Regular backward pass and optimization
loss.backward()
optimizer.step()
""")
    
    print("The optimizer uses lambda to modulate parameter updates:")
    print("""
# In the optimizer step() method:
lambda_val = self.compute_lambda()  # Based on current curvature

for group in self.param_groups:
    for p in group['params']:
        if p.grad is None:
            continue
            
        # Get parameter utility (how important this weight is)
        utility = self.get_utility(p)
        
        # Dynamic protection: high utility + high lambda = strong protection
        protection = lambda_val * utility
        
        # Modulated update
        update = grad / (1 + protection)
        p.data.add_(update, alpha=-lr)
""")

def main():
    """Run all demonstrations."""
    torch.manual_seed(42)  # For reproducibility
    
    # 1. Show basic curvature calculation
    easy_curv, hard_curv = demonstrate_curvature_calculation()
    
    # 2. Show lambda mapping
    demonstrate_lambda_mapping()
    
    # 3. Show integration details
    show_implementation_details()
    
    print("\\n=== Summary ===")
    print("✓ Input curvature tr(H_x²) measures loss landscape sharpness")
    print("✓ Hutchinson estimator provides efficient approximation")
    print("✓ Hard samples typically have higher curvature")
    print("✓ Dynamic lambda provides curvature-gated protection")
    print("✓ Integration preserves plasticity on easy samples")
    print("\\nRun your experiment with: sbatch test_inputaware_emnist_stats.sh")

if __name__ == "__main__":
    main()