Yes! There are several ways to track utility. Let me check what's currently being logged:

[1 tool called]

Now let me check if the runner script logs utility:

[1 tool called]

I can see that utility is **not currently being logged**. Let me add utility tracking to the optimizer. Here's what we can do:

## **Option 1: Add Utility Logging to the Optimizer (Recommended)**

Add a method to return utility statistics and log them in the runner:

[4 tools called]

Now let me add a helper method to get utility statistics:

[1 tool called]

Now add the same for the second-order optimizer:

[1 tool called]

Perfect! Now let me add utility logging to the runner script:

[1 tool called]

Now let me add utility logging to the runner:

[1 tool called]

Perfect! Now utility will be tracked. Let me create a summary of what's been added:

## **✅ Utility Tracking Added!**

### **What's Now Tracked:**

When you run your experiments, WandB will now log these utility metrics every step:

1. **`utility/global_max`**: The maximum utility value across all parameters (used for normalization)
2. **`utility/mean`**: Average scaled utility across all parameter groups
3. **`utility/min`**: Minimum scaled utility value
4. **`utility/max`**: Maximum scaled utility value

### **How It Works:**

1. **In the optimizer** (`input_aware.py`):
   - Stores `global_max_util` after each step
   - Provides `get_utility_stats()` method to compute statistics

2. **In the runner** (`run_stats_with_curvature.py`):
   - Calls `get_utility_stats()` every logging step
   - Adds utility metrics to WandB logs

### **What You'll See in WandB:**

You'll now have a new "utility" section in your WandB dashboard with 4 charts:
- **utility/global_max**: Tracks how the maximum utility evolves
- **utility/mean**: Average utility across parameters (should be ~0.5 with sigmoid)
- **utility/min**: Minimum utility (low utility = high plasticity)
- **utility/max**: Maximum utility (high utility = strong protection)

### **Interpretation:**

- **High utility** → Parameter is important → Gets protected (low gating factor)
- **Low utility** → Parameter is unimportant → Gets full updates (high gating factor)
- **Utility growing over time** → Parameters becoming more important (less plastic)
- **Utility staying low** → Parameters remaining plastic

The utility tracking is now active for all your input-aware experiments! 🎯📊