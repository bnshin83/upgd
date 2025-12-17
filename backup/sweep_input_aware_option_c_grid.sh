#!/bin/bash
set -euo pipefail

# Grid sweep for Option C gating parameters: a (C_A), b (C_B), and minimum gating value
# Submits one SLURM job per combination by overriding environment variables inline.

SCRIPT_PATH="/scratch/gautschi/shin283/upgd/test_input_permuted_mnist_stats_upgd_input_aware_fo_global.sh"

# Adjust these grids as needed
A_VALUES=(0.5 1.0 2.0)
B_VALUES=(0.5 1.0 2.0)
MIN_G_VALUES=(0.0 0.02 0.05 0.1)

total_jobs=$(( ${#A_VALUES[@]} * ${#B_VALUES[@]} * ${#MIN_G_VALUES[@]} ))
echo "Submitting grid: |A|=${#A_VALUES[@]} |B|=${#B_VALUES[@]} |min_g|=${#MIN_G_VALUES[@]} => total=${total_jobs}"

for a in "${A_VALUES[@]}"; do
  for b in "${B_VALUES[@]}"; do
    for ming in "${MIN_G_VALUES[@]}"; do
      a_tag=${a//./p}
      b_tag=${b//./p}
      ming_tag=${ming//./p}
      job_name="upgd_cgrid_a_${a_tag}_b_${b_tag}_ming_${ming_tag}"

      echo "Submitting ${job_name} (a=${a}, b=${b}, min_g=${ming})"
      UPGD_LAMBDA_MAPPING=centered_linear \
      UPGD_GATING_STRATEGY=option_c \
      UPGD_OPTION_C_A="${a}" \
      UPGD_OPTION_C_B="${b}" \
      UPGD_MIN_GATING="${ming}" \
      sbatch --parsable --job-name="${job_name}" "${SCRIPT_PATH}"
    done
  done
done

echo "All ${total_jobs} jobs submitted."


