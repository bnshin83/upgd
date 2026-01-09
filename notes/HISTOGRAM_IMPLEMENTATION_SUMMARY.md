# Histogram Implementation Summary

## Completed
✅ **SGD Optimizer** (`core/optim/sgd.py`) - Fully implemented with:
- Gradient histograms (5 bins, log scale)
- Weight histograms (5 bins, log scale)
- Raw utility histograms (5 bins, centered around 0)
- All percentages and counts
- Integrated into `get_utility_stats()` method

## Histogram Bins Defined

### Gradient & Weight Histograms (Absolute Values, Log Scale)
- `< 1e-4`: Very small values
- `[1e-4, 1e-3)`: Small values
- `[1e-3, 1e-2)`: Medium-small values
- `[1e-2, 1e-1)`: Medium values
- `≥ 1e-1`: Large values

### Raw Utility Histograms (Signed Values, Fine-Grained)
- `< -0.001`: Negative utility
- `[-0.001, -0.0002)`: Small negative utility
- `[-0.0002, 0.0002]`: Near-zero utility
- `(0.0002, 0.001]`: Small positive utility
- `> 0.001`: Positive utility

### Scaled Utility Histograms (Existing, kept unchanged)
- `[0, 0.2)`, `[0.2, 0.4)`, `[0.4, 0.6)`, `[0.6, 0.8)`, `[0.8, 1.0]`

## WandB Logging Keys

All histograms are logged via `get_utility_stats()` with these keys:

```python
# Gradient histograms
'gradient/hist_lt_1e4'          # count
'gradient/hist_1e4_1e3'         # count
'gradient/hist_1e3_1e2'         # count
'gradient/hist_1e2_1e1'         # count
'gradient/hist_gte_1e1'         # count
'gradient/hist_lt_1e4_pct'      # percentage
'gradient/hist_1e4_1e3_pct'     # percentage
'gradient/hist_1e3_1e2_pct'     # percentage
'gradient/hist_1e2_1e1_pct'     # percentage
'gradient/hist_gte_1e1_pct'     # percentage

# Weight histograms
'weight/hist_lt_1e4'            # count
'weight/hist_1e4_1e3'           # count
'weight/hist_1e3_1e2'           # count
'weight/hist_1e2_1e1'           # count
'weight/hist_gte_1e1'           # count
'weight/hist_lt_1e4_pct'        # percentage
'weight/hist_1e4_1e3_pct'       # percentage
'weight/hist_1e3_1e2_pct'       # percentage
'weight/hist_1e2_1e1_pct'       # percentage
'weight/hist_gte_1e1_pct'       # percentage

# Raw utility histograms
'raw_utility/hist_lt_m001'       # count
'raw_utility/hist_m001_m0002'     # count
'raw_utility/hist_m0002_p0002'    # count
'raw_utility/hist_p0002_p001'     # count
'raw_utility/hist_gt_p001'       # count
'raw_utility/hist_lt_m001_pct'   # percentage
'raw_utility/hist_m001_m0002_pct' # percentage
'raw_utility/hist_m0002_p0002_pct'# percentage
'raw_utility/hist_p0002_p001_pct' # percentage
'raw_utility/hist_gt_p001_pct'   # percentage
```

## Pending - UPGD Optimizers

The UPGD file `core/optim/weight_upgd/first_order.py` contains 4 optimizer classes that need similar updates:

1. ✅ `FirstOrderNonprotectingGlobalUPGD` - Collection lists updated (lines 190-194)
2. ✅ `FirstOrderNonprotectingGlobalUPGD` - Collection append updated (lines 205-209)
3. ⚠️  `FirstOrderNonprotectingGlobalUPGD` - Histogram computation needs update
4. ⚠️  `FirstOrderNonprotectingLocalUPGD` - All sections need update
5. ✅ `FirstOrderGlobalUPGD` - Collection lists updated (lines 190-194)
6. ✅ `FirstOrderGlobalUPGD` - Collection append updated (lines 205-209)
7. ⚠️  `FirstOrderGlobalUPGD` - Histogram computation needs update
8. ⚠️  `FirstOrderLocalUPGD` - All sections need update

### Required Changes for Each UPGD Class

**Step 1:** Add collection lists (after `global_max_util` computation):
```python
# Collect scaled utilities, gradients, weights, and raw utilities for statistics
all_scaled_utilities = []
all_gradients = []
all_weights = []
all_raw_utilities = []
```

**Step 2:** Update collection in loop:
```python
# Collect for statistics
all_scaled_utilities.append(scaled_utility.flatten())
all_gradients.append(p.grad.flatten())
all_weights.append(p.data.flatten())
all_raw_utilities.append((state["avg_utility"] / bias_correction).flatten())
```

**Step 3:** Expand histogram computation (replace existing histogram section):
- Concatenate all tensors
- Add gradient histogram computation (5 bins)
- Add weight histogram computation (5 bins)
- Add raw utility histogram computation (5 bins)

**Step 4:** Update `get_utility_stats()` method (add to existing return dict):
- Add gradient histogram stats
- Add weight histogram stats
- Add raw utility histogram stats

## Implementation Code Reference

See `core/optim/sgd.py` lines 33-123 and lines 209-246 for complete working implementation.

## Benefits

1. **Gradient distribution analysis**: Understand if gradients are vanishing/exploding
2. **Weight distribution analysis**: Track weight magnitude evolution
3. **Raw utility distribution**: See unscaled utility values before sigmoid squashing
4. **Better debugging**: Identify why scaled utilities cluster in [0.4, 0.6]
5. **Hypothesis testing**: Verify mathematical predictions about utility magnitudes

## Next Steps

To complete UPGD histogram implementation:
1. Copy the histogram computation code from SGD to each UPGD class
2. Update the defaults in the `else` block
3. Update `get_utility_stats()` methods in each class
4. Test with a small run to verify logging works
