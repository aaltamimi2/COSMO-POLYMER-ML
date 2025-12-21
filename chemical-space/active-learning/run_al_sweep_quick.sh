#!/bin/bash
# ======================================================================
# Active Learning Quick Sweep Script (Fast Test Version)
#
# Runs a smaller subset of experiments for quick testing
# Results organized by: temp/method/acquisition/experiment
#
# Usage: bash run_al_sweep_quick.sh
# ======================================================================

set -e  # Exit on error

# Configuration
DATA_DIR="REF-DATA"
OUTPUT_BASE="al_sweep_quick"
LOG_FILE="${OUTPUT_BASE}/sweep_log.txt"
SUMMARY_FILE="${OUTPUT_BASE}/sweep_summary.csv"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Initialize log
echo "========================================" | tee "${LOG_FILE}"
echo "Active Learning Quick Sweep Started" | tee -a "${LOG_FILE}"
echo "Date: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Initialize summary CSV
echo "experiment,temperature,acquisition,method,iterations,initial_samples,best_polymer,best_score,efficiency_rank,n_polymers_tested,runtime_sec,results_path" > "${SUMMARY_FILE}"

# Counter for experiments
exp_count=0

# Helper function to run experiment
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
    echo "  Temperature: ${temp}°C | Acquisition: ${acq} | Method: ${method}" | tee -a "${LOG_FILE}"

    # Create organized directory structure
    temp_dir="${OUTPUT_BASE}/temp_${temp}C"
    method_dir="${temp_dir}/method_${method}"
    acq_dir="${method_dir}/acq_${acq}"
    exp_dir="${acq_dir}/${exp_name}"

    mkdir -p "${exp_dir}"

    start_time=$(date +%s)

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

    # Extract results
    if [ -f "${exp_dir}/results/campaign_history.csv" ]; then
        last_row=$(tail -n 1 "${exp_dir}/results/campaign_history.csv")
        best_polymer=$(echo "${last_row}" | cut -d',' -f4)
        best_score=$(echo "${last_row}" | cut -d',' -f5)
        n_tested=$(echo "${last_row}" | cut -d',' -f6)
        efficiency_rank=$(grep "Efficiency: Found rank" "${exp_dir}/output.log" | grep -oP 'rank \K\d+' || echo "N/A")

        echo "  Result: ${best_polymer} (score=${best_score}, rank=${efficiency_rank}) [${runtime}s]" | tee -a "${LOG_FILE}"
        echo "${exp_name},${temp},${acq},${method},${iterations},${initial},${best_polymer},${best_score},${efficiency_rank},${n_tested},${runtime},${exp_dir}" >> "${SUMMARY_FILE}"
    else
        echo "  ERROR: No results generated" | tee -a "${LOG_FILE}"
        echo "${exp_name},${temp},${acq},${method},${iterations},${initial},ERROR,N/A,N/A,N/A,${runtime},${exp_dir}" >> "${SUMMARY_FILE}"
    fi
}

# ======================================================================
# QUICK TESTS (fewer iterations, smaller parameter space)
# ======================================================================

echo "" | tee -a "${LOG_FILE}"
echo "Test 1: Temperature Sweep (3 temps)" | tee -a "${LOG_FILE}"
for temp in 25 80 120; do
    run_experiment "quick_temp${temp}" "${temp}" "ei" "versatility" 5 2
done

echo "" | tee -a "${LOG_FILE}"
echo "Test 2: Acquisition Functions (all 4)" | tee -a "${LOG_FILE}"
for acq in ei ucb pi greedy; do
    run_experiment "quick_acq_${acq}" 25 "${acq}" "versatility" 5 2
done

echo "" | tee -a "${LOG_FILE}"
echo "Test 3: Scoring Methods (3 methods)" | tee -a "${LOG_FILE}"
for method in versatility max mean; do
    run_experiment "quick_method_${method}" 25 "ei" "${method}" 5 2
done

echo "" | tee -a "${LOG_FILE}"
echo "Test 4: Specific Solvents (3 examples)" | tee -a "${LOG_FILE}"
for solvent in propanol acetone toluene; do
    run_experiment "quick_solvent_${solvent}" 25 "ei" "specific" 5 2 "--solvent ${solvent}"
done

# ======================================================================
# Summary
# ======================================================================
echo "" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "QUICK SWEEP COMPLETE" | tee -a "${LOG_FILE}"
echo "======================================================================" | tee -a "${LOG_FILE}"
echo "Total experiments: ${exp_count}" | tee -a "${LOG_FILE}"
echo "Results organized in: ${OUTPUT_BASE}/" | tee -a "${LOG_FILE}"
echo "  Structure: temp_XXC/method_YYY/acq_ZZZ/experiment_name/results/" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Directory tree overview
echo "Directory structure:" | tee -a "${LOG_FILE}"
tree -L 4 -d "${OUTPUT_BASE}" 2>/dev/null | tee -a "${LOG_FILE}" || \
    find "${OUTPUT_BASE}" -type d -print | sed 's|[^/]*/| |g' | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Quick Summary:" | tee -a "${LOG_FILE}"
echo "--------------" | tee -a "${LOG_FILE}"
tail -n +2 "${SUMMARY_FILE}" | while IFS=',' read -r exp temp acq method iters init polymer score rank tested time path; do
    if [ "$polymer" != "ERROR" ]; then
        echo "  ${exp}: ${polymer} (${score})" | tee -a "${LOG_FILE}"
    fi
done

echo "" | tee -a "${LOG_FILE}"
echo "Summary CSV: ${SUMMARY_FILE}" | tee -a "${LOG_FILE}"
echo "Done! Check ${OUTPUT_BASE}/ for detailed results." | tee -a "${LOG_FILE}"
