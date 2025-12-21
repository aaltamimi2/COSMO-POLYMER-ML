#!/bin/bash
# ======================================================================
# Active Learning Sweep Script
#
# Systematically tests different configurations of polymer_al_real_data.py
# to explore the parameter space and compare results.
#
# Usage: bash run_al_sweep.sh
# ======================================================================

set -e  # Exit on error

# Configuration
DATA_DIR="REF-DATA"
OUTPUT_BASE="al_sweep_results"
LOG_FILE="${OUTPUT_BASE}/sweep_log.txt"
SUMMARY_FILE="${OUTPUT_BASE}/sweep_summary.csv"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Initialize log
echo "========================================" | tee "${LOG_FILE}"
echo "Active Learning Sweep Started" | tee -a "${LOG_FILE}"
echo "Date: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Initialize summary CSV
echo "experiment,temperature,acquisition,method,iterations,initial_samples,best_polymer,best_score,efficiency_rank,n_polymers_tested,runtime_sec,results_path" > "${SUMMARY_FILE}"

# Counter for experiments
exp_count=0

# ======================================================================
# Helper function to run experiment
# ======================================================================
run_experiment() {
    local exp_name=$1
    local temp=$2
    local acq=$3
    local method=$4
    local iterations=$5
    local initial=$6
    shift 6
    local extra_args="$@"

    exp_count=$((exp_count + 1))

    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "Experiment ${exp_count}: ${exp_name}" | tee -a "${LOG_FILE}"
    echo "  Temperature: ${temp}°C" | tee -a "${LOG_FILE}"
    echo "  Acquisition: ${acq}" | tee -a "${LOG_FILE}"
    echo "  Method: ${method}" | tee -a "${LOG_FILE}"
    echo "  Iterations: ${iterations}" | tee -a "${LOG_FILE}"
    echo "  Initial: ${initial}" | tee -a "${LOG_FILE}"
    echo "  Extra args: ${extra_args}" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"

    # Create experiment directory with organized structure
    # Format: OUTPUT_BASE/temp_XXX/method_YYY/acq_ZZZ/exp_name/
    temp_dir="${OUTPUT_BASE}/temp_${temp}C"
    method_dir="${temp_dir}/method_${method}"
    acq_dir="${method_dir}/acq_${acq}"
    exp_dir="${acq_dir}/${exp_name}"

    mkdir -p "${exp_dir}"

    # Run experiment and capture timing
    start_time=$(date +%s)

    # Use --output to directly save to experiment directory
    python polymer_al_real_data.py \
        --data "${DATA_DIR}" \
        --temperature "${temp}" \
        --acquisition "${acq}" \
        --method "${method}" \
        --iterations "${iterations}" \
        --initial "${initial}" \
        --output "${exp_dir}/results" \
        ${extra_args} \
        > "${exp_dir}/output.log" 2>&1

    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    # Extract results from campaign history
    if [ -f "${exp_dir}/results/campaign_history.csv" ]; then
        # Get last row (final results)
        last_row=$(tail -n 1 "${exp_dir}/results/campaign_history.csv")
        best_polymer=$(echo "${last_row}" | cut -d',' -f4)
        best_score=$(echo "${last_row}" | cut -d',' -f5)
        n_tested=$(echo "${last_row}" | cut -d',' -f6)

        # Extract efficiency rank from output log
        efficiency_rank=$(grep "Efficiency: Found rank" "${exp_dir}/output.log" | grep -oP 'rank \K\d+' || echo "N/A")

        echo "  Results: ${best_polymer} (score=${best_score}, rank=${efficiency_rank}/${n_tested})" | tee -a "${LOG_FILE}"
        echo "  Runtime: ${runtime}s" | tee -a "${LOG_FILE}"

        # Add to summary with full path for reference
        echo "${exp_name},${temp},${acq},${method},${iterations},${initial},${best_polymer},${best_score},${efficiency_rank},${n_tested},${runtime},${exp_dir}" >> "${SUMMARY_FILE}"
    else
        echo "  ERROR: No results generated" | tee -a "${LOG_FILE}"
        echo "${exp_name},${temp},${acq},${method},${iterations},${initial},ERROR,N/A,N/A,N/A,${runtime},${exp_dir}" >> "${SUMMARY_FILE}"
    fi

    echo "" | tee -a "${LOG_FILE}"
}

# ======================================================================
# SWEEP 1: Temperature Sweep (same acquisition, method)
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP 1: Temperature Variation" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for temp in 25 50 80 100 120; do
    run_experiment "temp${temp}_ei_versatility" "${temp}" "ei" "versatility" 8 2
done

# ======================================================================
# SWEEP 2: Acquisition Function Sweep (same temp, method)
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP 2: Acquisition Function Variation" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for acq in ei ucb pi greedy; do
    run_experiment "temp25_${acq}_versatility" 25 "${acq}" "versatility" 8 2
done

# ======================================================================
# SWEEP 3: Scoring Method Sweep
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP 3: Scoring Method Variation" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for method in versatility max mean; do
    run_experiment "temp25_ei_${method}" 25 "ei" "${method}" 8 2
done

# Count method with different thresholds
for threshold in 5 10 20; do
    run_experiment "temp25_ei_count_thresh${threshold}" 25 "ei" "count" 8 2 "--threshold ${threshold}"
done

# ======================================================================
# SWEEP 4: Specific Solvent Optimization
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP 4: Solvent-Specific Optimization" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Get list of solvents from one of the data files
solvents=("propanol" "2-propanol" "acetone" "toluene" "chloroform" "THF" "water")

for solvent in "${solvents[@]}"; do
    # URL-encode spaces if needed
    solvent_clean=$(echo "${solvent}" | tr ' ' '_')
    run_experiment "temp25_ei_specific_${solvent_clean}" 25 "ei" "specific" 8 2 "--solvent ${solvent}"
done

# ======================================================================
# SWEEP 5: Iteration Count Variation
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP 5: Iteration Count Variation" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for iters in 3 5 8 10 15; do
    run_experiment "temp25_ei_versatility_iter${iters}" 25 "ei" "versatility" "${iters}" 2
done

# ======================================================================
# SWEEP 6: Initial Sample Size Variation
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP 6: Initial Sample Size Variation" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for initial in 1 2 3 4; do
    run_experiment "temp25_ei_versatility_init${initial}" 25 "ei" "versatility" 8 "${initial}"
done

# ======================================================================
# SWEEP 7: Combined High-Temperature + Different Acquisition
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP 7: High-Temperature Experiments" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for temp in 100 120 140 160; do
    for acq in ei ucb; do
        run_experiment "temp${temp}_${acq}_versatility" "${temp}" "${acq}" "versatility" 8 2
    done
done

# ======================================================================
# SWEEP 8: Best Practices (recommended settings)
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP 8: Best Practice Configurations" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# EI with more iterations and initial samples
run_experiment "best_practice_ei_comprehensive" 25 "ei" "versatility" 12 3

# UCB with higher exploration
run_experiment "best_practice_ucb_exploration" 25 "ucb" "versatility" 12 3

# Mid-temperature for better solubility
run_experiment "best_practice_midtemp_ei" 80 "ei" "versatility" 10 3

# ======================================================================
# Generate Summary Report
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "GENERATING SUMMARY REPORT" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Create Python script to analyze results
cat > "${OUTPUT_BASE}/analyze_results.py" << 'PYTHON_EOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read summary
df = pd.read_csv('al_sweep_results/sweep_summary.csv')

# Filter out errors
df = df[df['best_polymer'] != 'ERROR'].copy()
df['best_score'] = pd.to_numeric(df['best_score'], errors='coerce')
df['efficiency_rank'] = pd.to_numeric(df['efficiency_rank'], errors='coerce')

print("=" * 70)
print("ACTIVE LEARNING SWEEP SUMMARY")
print("=" * 70)
print(f"\nTotal experiments: {len(df)}")
print(f"Average runtime: {df['runtime_sec'].mean():.1f}s")
print(f"\nBest polymer found most often: {df['best_polymer'].mode()[0] if len(df) > 0 else 'N/A'}")
print(f"Average best score: {df['best_score'].mean():.3f}")
print(f"Average efficiency rank: {df['efficiency_rank'].mean():.1f}")

# Group by acquisition function
print("\n" + "-" * 70)
print("Performance by Acquisition Function:")
print("-" * 70)
acq_stats = df.groupby('acquisition').agg({
    'best_score': ['mean', 'std'],
    'efficiency_rank': 'mean',
    'runtime_sec': 'mean'
}).round(3)
print(acq_stats)

# Group by temperature
print("\n" + "-" * 70)
print("Performance by Temperature:")
print("-" * 70)
temp_stats = df.groupby('temperature').agg({
    'best_score': ['mean', 'std'],
    'efficiency_rank': 'mean',
    'best_polymer': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
}).round(3)
print(temp_stats)

# Group by method
print("\n" + "-" * 70)
print("Performance by Scoring Method:")
print("-" * 70)
method_stats = df.groupby('method').agg({
    'best_score': ['mean', 'std'],
    'efficiency_rank': 'mean'
}).round(3)
print(method_stats)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Best score by acquisition
ax = axes[0, 0]
df.boxplot(column='best_score', by='acquisition', ax=ax)
ax.set_title('Best Score by Acquisition Function')
ax.set_xlabel('Acquisition Function')
ax.set_ylabel('Best Score')
plt.sca(ax)
plt.xticks(rotation=45)

# 2. Efficiency rank by acquisition
ax = axes[0, 1]
df.boxplot(column='efficiency_rank', by='acquisition', ax=ax)
ax.set_title('Efficiency Rank by Acquisition Function')
ax.set_xlabel('Acquisition Function')
ax.set_ylabel('Rank (lower is better)')
plt.sca(ax)
plt.xticks(rotation=45)

# 3. Best score by temperature
ax = axes[0, 2]
temp_grouped = df.groupby('temperature')['best_score'].mean()
ax.plot(temp_grouped.index, temp_grouped.values, 'o-', linewidth=2, markersize=8)
ax.set_title('Best Score vs Temperature')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Average Best Score')
ax.grid(True, alpha=0.3)

# 4. Runtime by configuration
ax = axes[1, 0]
df.boxplot(column='runtime_sec', by='acquisition', ax=ax)
ax.set_title('Runtime by Acquisition Function')
ax.set_xlabel('Acquisition Function')
ax.set_ylabel('Runtime (seconds)')
plt.sca(ax)
plt.xticks(rotation=45)

# 5. Best polymer distribution
ax = axes[1, 1]
polymer_counts = df['best_polymer'].value_counts()
ax.bar(range(len(polymer_counts)), polymer_counts.values)
ax.set_xticks(range(len(polymer_counts)))
ax.set_xticklabels(polymer_counts.index, rotation=45, ha='right')
ax.set_title('Best Polymer Found (Frequency)')
ax.set_ylabel('Count')

# 6. Score vs efficiency rank
ax = axes[1, 2]
for acq in df['acquisition'].unique():
    subset = df[df['acquisition'] == acq]
    ax.scatter(subset['efficiency_rank'], subset['best_score'],
              label=acq, alpha=0.7, s=100)
ax.set_xlabel('Efficiency Rank')
ax.set_ylabel('Best Score')
ax.set_title('Score vs Efficiency by Acquisition')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('al_sweep_results/sweep_analysis.png', dpi=200, bbox_inches='tight')
print(f"\n[OK] Saved visualization: al_sweep_results/sweep_analysis.png")

# Save detailed stats
with open('al_sweep_results/detailed_stats.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("DETAILED STATISTICS\n")
    f.write("=" * 70 + "\n\n")

    f.write("Overall Statistics:\n")
    f.write("-" * 70 + "\n")
    f.write(df.describe().to_string())
    f.write("\n\n")

    f.write("Top 10 Experiments by Best Score:\n")
    f.write("-" * 70 + "\n")
    f.write(df.nlargest(10, 'best_score')[['experiment', 'temperature', 'acquisition',
                                            'method', 'best_polymer', 'best_score',
                                            'efficiency_rank']].to_string(index=False))
    f.write("\n\n")

    f.write("Top 10 Experiments by Efficiency:\n")
    f.write("-" * 70 + "\n")
    f.write(df.nsmallest(10, 'efficiency_rank')[['experiment', 'temperature', 'acquisition',
                                                   'method', 'best_polymer', 'best_score',
                                                   'efficiency_rank']].to_string(index=False))
    f.write("\n")

print(f"[OK] Saved detailed stats: al_sweep_results/detailed_stats.txt")
PYTHON_EOF

# Run analysis
python "${OUTPUT_BASE}/analyze_results.py" | tee -a "${LOG_FILE}"

# ======================================================================
# Final Summary
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "SWEEP COMPLETE" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "Total experiments run: ${exp_count}" | tee -a "${LOG_FILE}"
echo "Results saved in: ${OUTPUT_BASE}/" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Files generated:" | tee -a "${LOG_FILE}"
echo "  - ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "  - ${SUMMARY_FILE}" | tee -a "${LOG_FILE}"
echo "  - ${OUTPUT_BASE}/sweep_analysis.png" | tee -a "${LOG_FILE}"
echo "  - ${OUTPUT_BASE}/detailed_stats.txt" | tee -a "${LOG_FILE}"
echo "  - Individual experiment folders in ${OUTPUT_BASE}/" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Date completed: $(date)" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
