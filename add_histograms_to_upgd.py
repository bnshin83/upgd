#!/usr/bin/env python3
"""
Script to add gradient, weight, and raw utility histograms to UPGD optimizers.
Updates both FirstOrderNonprotectingGlobalUPGD and FirstOrderGlobalUPGD classes.
"""

import re

def add_histograms_to_upgd():
    file_path = '/scratch/gautschi/shin283/upgd/core/optim/weight_upgd/first_order.py'

    with open(file_path, 'r') as f:
        content = f.read()

    # Pattern 1: Update collection lists (for both classes)
    old_pattern1 = r'(\s+# Collect scaled utilities for norm computation\n\s+all_scaled_utilities = \[\]\n\n\s+for group in self\.param_groups:)'
    new_pattern1 = r'''        # Collect scaled utilities, gradients, weights, and raw utilities for statistics
        all_scaled_utilities = []
        all_gradients = []
        all_weights = []
        all_raw_utilities = []

        for group in self.param_groups:'''

    content = re.sub(old_pattern1, new_pattern1, content)

    # Pattern 2: Update collection append (for both classes)
    old_pattern2 = r'(\s+# Collect scaled utilities for norm computation\n\s+all_scaled_utilities\.append\(scaled_utility\.flatten\(\)\))'
    new_pattern2 = r'''                # Collect for statistics
                all_scaled_utilities.append(scaled_utility.flatten())
                all_gradients.append(p.grad.flatten())
                all_weights.append(p.data.flatten())
                all_raw_utilities.append((state["avg_utility"] / bias_correction).flatten())'''

    content = re.sub(old_pattern2, new_pattern2, content)

    # Pattern 3: Replace histogram computation sections
    # This pattern targets the specific section in each class
    old_histogram_section = r'''        # Compute norms and histograms on scaled utilities
        if all_scaled_utilities:
            scaled_utility_tensor = torch\.cat\(all_scaled_utilities\)
            self\.utility_L1 = torch\.norm\(scaled_utility_tensor, p=1\)\.item\(\)
            self\.utility_L2 = torch\.norm\(scaled_utility_tensor, p=2\)\.item\(\)
            self\.utility_L4 = torch\.norm\(scaled_utility_tensor, p=4\)\.item\(\)
            self\.utility_L5 = torch\.norm\(scaled_utility_tensor, p=5\)\.item\(\)
            self\.utility_L10 = torch\.norm\(scaled_utility_tensor, p=10\)\.item\(\)

            # Compute histogram statistics \(5 bins: \[0, 0\.2\), \[0\.2, 0\.4\), \[0\.4, 0\.6\), \[0\.6, 0\.8\), \[0\.8, 1\.0\)\)
            total_params = scaled_utility_tensor\.numel\(\)
            self\.utility_hist_0_20 = \(\(scaled_utility_tensor >= 0\.0\) & \(scaled_utility_tensor < 0\.2\)\)\.sum\(\)\.item\(\)
            self\.utility_hist_20_40 = \(\(scaled_utility_tensor >= 0\.2\) & \(scaled_utility_tensor < 0\.4\)\)\.sum\(\)\.item\(\)
            self\.utility_hist_40_60 = \(\(scaled_utility_tensor >= 0\.4\) & \(scaled_utility_tensor < 0\.6\)\)\.sum\(\)\.item\(\)
            self\.utility_hist_60_80 = \(\(scaled_utility_tensor >= 0\.6\) & \(scaled_utility_tensor < 0\.8\)\)\.sum\(\)\.item\(\)
            self\.utility_hist_80_100 = \(\(scaled_utility_tensor >= 0\.8\) & \(scaled_utility_tensor <= 1\.0\)\)\.sum\(\)\.item\(\)

            # Store percentages as well
            self\.utility_hist_0_20_pct = \(self\.utility_hist_0_20 / total_params\) \* 100
            self\.utility_hist_20_40_pct = \(self\.utility_hist_20_40 / total_params\) \* 100
            self\.utility_hist_40_60_pct = \(self\.utility_hist_40_60 / total_params\) \* 100
            self\.utility_hist_60_80_pct = \(self\.utility_hist_60_80 / total_params\) \* 100
            self\.utility_hist_80_100_pct = \(self\.utility_hist_80_100 / total_params\) \* 100
            self\.utility_total_params = total_params'''

    new_histogram_section = '''        # Compute norms and histograms on scaled utilities, gradients, weights, and raw utilities
        if all_scaled_utilities:
            scaled_utility_tensor = torch.cat(all_scaled_utilities)
            gradient_tensor = torch.cat(all_gradients)
            weight_tensor = torch.cat(all_weights)
            raw_utility_tensor = torch.cat(all_raw_utilities)

            total_params = scaled_utility_tensor.numel()

            # Scaled utility statistics
            self.utility_L1 = torch.norm(scaled_utility_tensor, p=1).item()
            self.utility_L2 = torch.norm(scaled_utility_tensor, p=2).item()
            self.utility_L4 = torch.norm(scaled_utility_tensor, p=4).item()
            self.utility_L5 = torch.norm(scaled_utility_tensor, p=5).item()
            self.utility_L10 = torch.norm(scaled_utility_tensor, p=10).item()

            # Scaled utility histogram (5 bins: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0])
            self.utility_hist_0_20 = ((scaled_utility_tensor >= 0.0) & (scaled_utility_tensor < 0.2)).sum().item()
            self.utility_hist_20_40 = ((scaled_utility_tensor >= 0.2) & (scaled_utility_tensor < 0.4)).sum().item()
            self.utility_hist_40_60 = ((scaled_utility_tensor >= 0.4) & (scaled_utility_tensor < 0.6)).sum().item()
            self.utility_hist_60_80 = ((scaled_utility_tensor >= 0.6) & (scaled_utility_tensor < 0.8)).sum().item()
            self.utility_hist_80_100 = ((scaled_utility_tensor >= 0.8) & (scaled_utility_tensor <= 1.0)).sum().item()
            self.utility_hist_0_20_pct = (self.utility_hist_0_20 / total_params) * 100
            self.utility_hist_20_40_pct = (self.utility_hist_20_40 / total_params) * 100
            self.utility_hist_40_60_pct = (self.utility_hist_40_60 / total_params) * 100
            self.utility_hist_60_80_pct = (self.utility_hist_60_80 / total_params) * 100
            self.utility_hist_80_100_pct = (self.utility_hist_80_100 / total_params) * 100
            self.utility_total_params = total_params

            # Gradient histogram (log scale: <1e-4, [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), >=1e-1)
            grad_abs = torch.abs(gradient_tensor)
            self.grad_hist_lt_1e4 = (grad_abs < 1e-4).sum().item()
            self.grad_hist_1e4_1e3 = ((grad_abs >= 1e-4) & (grad_abs < 1e-3)).sum().item()
            self.grad_hist_1e3_1e2 = ((grad_abs >= 1e-3) & (grad_abs < 1e-2)).sum().item()
            self.grad_hist_1e2_1e1 = ((grad_abs >= 1e-2) & (grad_abs < 1e-1)).sum().item()
            self.grad_hist_gte_1e1 = (grad_abs >= 1e-1).sum().item()
            self.grad_hist_lt_1e4_pct = (self.grad_hist_lt_1e4 / total_params) * 100
            self.grad_hist_1e4_1e3_pct = (self.grad_hist_1e4_1e3 / total_params) * 100
            self.grad_hist_1e3_1e2_pct = (self.grad_hist_1e3_1e2 / total_params) * 100
            self.grad_hist_1e2_1e1_pct = (self.grad_hist_1e2_1e1 / total_params) * 100
            self.grad_hist_gte_1e1_pct = (self.grad_hist_gte_1e1 / total_params) * 100

            # Weight histogram (log scale: <1e-4, [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), >=1e-1)
            weight_abs = torch.abs(weight_tensor)
            self.weight_hist_lt_1e4 = (weight_abs < 1e-4).sum().item()
            self.weight_hist_1e4_1e3 = ((weight_abs >= 1e-4) & (weight_abs < 1e-3)).sum().item()
            self.weight_hist_1e3_1e2 = ((weight_abs >= 1e-3) & (weight_abs < 1e-2)).sum().item()
            self.weight_hist_1e2_1e1 = ((weight_abs >= 1e-2) & (weight_abs < 1e-1)).sum().item()
            self.weight_hist_gte_1e1 = (weight_abs >= 1e-1).sum().item()
            self.weight_hist_lt_1e4_pct = (self.weight_hist_lt_1e4 / total_params) * 100
            self.weight_hist_1e4_1e3_pct = (self.weight_hist_1e4_1e3 / total_params) * 100
            self.weight_hist_1e3_1e2_pct = (self.weight_hist_1e3_1e2 / total_params) * 100
            self.weight_hist_1e2_1e1_pct = (self.weight_hist_1e2_1e1 / total_params) * 100
            self.weight_hist_gte_1e1_pct = (self.weight_hist_gte_1e1 / total_params) * 100

            # Raw utility histogram (centered around 0: <-0.01, [-0.01,-0.001), [-0.001,0.001), [0.001,0.01), >=0.01)
            self.raw_util_hist_lt_m01 = (raw_utility_tensor < -0.01).sum().item()
            self.raw_util_hist_m01_m001 = ((raw_utility_tensor >= -0.01) & (raw_utility_tensor < -0.001)).sum().item()
            self.raw_util_hist_m001_p001 = ((raw_utility_tensor >= -0.001) & (raw_utility_tensor <= 0.001)).sum().item()
            self.raw_util_hist_p001_p01 = ((raw_utility_tensor > 0.001) & (raw_utility_tensor <= 0.01)).sum().item()
            self.raw_util_hist_gt_p01 = (raw_utility_tensor > 0.01).sum().item()
            self.raw_util_hist_lt_m01_pct = (self.raw_util_hist_lt_m01 / total_params) * 100
            self.raw_util_hist_m01_m001_pct = (self.raw_util_hist_m01_m001 / total_params) * 100
            self.raw_util_hist_m001_p001_pct = (self.raw_util_hist_m001_p001 / total_params) * 100
            self.raw_util_hist_p001_p01_pct = (self.raw_util_hist_p001_p01 / total_params) * 100
            self.raw_util_hist_gt_p01_pct = (self.raw_util_hist_gt_p01 / total_params) * 100'''

    content = re.sub(old_histogram_section, new_histogram_section, content)

    # Write back
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Successfully updated {file_path}")
    print("Added gradient, weight, and raw utility histograms to UPGD optimizers")

if __name__ == '__main__':
    add_histograms_to_upgd()
