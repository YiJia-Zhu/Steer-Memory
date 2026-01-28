#!/bin/bash

# Comprehensive test script to verify all 24 configurations work correctly
# Tests 4 models × 6 datasets with a small sample (3 examples per dataset)
# This is a quick smoke test before running the full batch

set -e  # Exit on error

# Specify which GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model paths
MODEL_ROOT="/storage/szn_data/model_weights"
MODELS=(
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-7B-Instruct"
    "DeepSeek-R1-Distill-Qwen-1.5B"
    "DeepSeek-R1-Distill-Qwen-7B"
)

# Dataset names
DATASETS=(
    "aime_2024"
    "aime25"
    "amc23"
    "arc-c"
    "math500"
    "openbookqa"
)

# Test parameters
MAX_TOKENS=16384
TEST_EXAMPLES=3  # Small number for quick testing
TEST_ROOT="results/test_all"

# Function to get short model name
get_model_short_name() {
    case "$1" in
        "DeepSeek-R1-Distill-Qwen-1.5B")
            echo "DS-R1-1.5B"
            ;;
        "DeepSeek-R1-Distill-Qwen-7B")
            echo "DS-R1-7B"
            ;;
        "Qwen2.5-3B-Instruct")
            echo "Qwen2.5-3B"
            ;;
        "Qwen2.5-7B-Instruct")
            echo "Qwen2.5-7B"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

# Clean up previous test results
if [ -d "${TEST_ROOT}" ]; then
    echo "Cleaning up previous test results..."
    rm -rf "${TEST_ROOT}"
fi

mkdir -p "${TEST_ROOT}"

# Log file
LOG_FILE="${TEST_ROOT}/test_all_configs.log"
SUMMARY_FILE="${TEST_ROOT}/test_summary.txt"

echo "=========================================" | tee "${LOG_FILE}"
echo "SEAL Baseline - Testing All 24 Configurations" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "Test examples per dataset: ${TEST_EXAMPLES}" | tee -a "${LOG_FILE}"
echo "Start time: $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Initialize counters
TOTAL_CONFIGS=0
PASSED_CONFIGS=0
FAILED_CONFIGS=0

# Initialize summary
echo "Configuration Test Summary" > "${SUMMARY_FILE}"
echo "=========================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Test all configurations
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
    MODEL_SHORT=$(get_model_short_name "${MODEL_NAME}")
    
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Testing Model: ${MODEL_NAME}" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    for DATASET in "${DATASETS[@]}"; do
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
        
        echo "----------------------------------------" | tee -a "${LOG_FILE}"
        echo "Config ${TOTAL_CONFIGS}/24: ${MODEL_SHORT} + ${DATASET}" | tee -a "${LOG_FILE}"
        echo "Start: $(date)" | tee -a "${LOG_FILE}"
        
        TEST_DIR="${TEST_ROOT}/${MODEL_SHORT}/${DATASET}"
        mkdir -p "${TEST_DIR}"
        
        # Test baseline evaluation
        echo "Running baseline evaluation..." | tee -a "${LOG_FILE}"
        
        if python eval_MATH_vllm.py \
            --model_name_or_path "${MODEL_PATH}" \
            --dataset "${DATASET}" \
            --max_tokens "${MAX_TOKENS}" \
            --max_examples ${TEST_EXAMPLES} \
            --save_dir "${TEST_DIR}" \
            --use_chat_format \
            --remove_bos \
            >> "${LOG_FILE}" 2>&1; then
            
            # Check if predictions file exists
            if [ -f "${TEST_DIR}/predictions.jsonl" ]; then
                PASSED_CONFIGS=$((PASSED_CONFIGS + 1))
                echo "✓ PASSED: ${MODEL_SHORT} + ${DATASET}" | tee -a "${LOG_FILE}"
                echo "✓ ${MODEL_SHORT} + ${DATASET}" >> "${SUMMARY_FILE}"
            else
                FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                echo "✗ FAILED: ${MODEL_SHORT} + ${DATASET} (no predictions file)" | tee -a "${LOG_FILE}"
                echo "✗ ${MODEL_SHORT} + ${DATASET} (no predictions)" >> "${SUMMARY_FILE}"
            fi
        else
            FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
            echo "✗ FAILED: ${MODEL_SHORT} + ${DATASET} (evaluation error)" | tee -a "${LOG_FILE}"
            echo "✗ ${MODEL_SHORT} + ${DATASET} (evaluation error)" >> "${SUMMARY_FILE}"
        fi
        
        echo "End: $(date)" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
    done
    
    echo "" | tee -a "${LOG_FILE}"
done

# Print summary
echo "" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "Test Summary" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "Total configurations tested: ${TOTAL_CONFIGS}" | tee -a "${LOG_FILE}"
echo "Passed: ${PASSED_CONFIGS}" | tee -a "${LOG_FILE}"
echo "Failed: ${FAILED_CONFIGS}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Add summary to summary file
echo "" >> "${SUMMARY_FILE}"
echo "=========================" >> "${SUMMARY_FILE}"
echo "Total: ${TOTAL_CONFIGS}" >> "${SUMMARY_FILE}"
echo "Passed: ${PASSED_CONFIGS}" >> "${SUMMARY_FILE}"
echo "Failed: ${FAILED_CONFIGS}" >> "${SUMMARY_FILE}"

if [ ${FAILED_CONFIGS} -eq 0 ]; then
    echo "✓ ALL TESTS PASSED!" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    echo "You can now run the full batch scripts:" | tee -a "${LOG_FILE}"
    echo "  bash scripts/run_all_baseline.sh" | tee -a "${LOG_FILE}"
    echo "  bash scripts/run_all_vector_generation.sh" | tee -a "${LOG_FILE}"
    echo "  bash scripts/run_all_steering.sh" | tee -a "${LOG_FILE}"
    exit 0
else
    echo "✗ SOME TESTS FAILED" | tee -a "${LOG_FILE}"
    echo "Check ${LOG_FILE} for details" | tee -a "${LOG_FILE}"
    echo "Summary saved in ${SUMMARY_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi

