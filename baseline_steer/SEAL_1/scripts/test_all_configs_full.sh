#!/bin/bash

# FULL END-TO-END test script for all 24 configurations
# Tests: Baseline → Hidden States → Vector Generation → Steering Evaluation
# Uses 3 examples per dataset for quick testing

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
TEST_EXAMPLES=3
STEERING_LAYER=20
STEERING_COEF=1.0
TEST_ROOT="results/test_all_full"

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
LOG_FILE="${TEST_ROOT}/test_all_full.log"
SUMMARY_FILE="${TEST_ROOT}/test_summary.txt"

echo "=========================================" | tee "${LOG_FILE}"
echo "SEAL Baseline - FULL E2E Test (24 Configs)" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "Test examples per dataset: ${TEST_EXAMPLES}" | tee -a "${LOG_FILE}"
echo "Steering layer: ${STEERING_LAYER}" | tee -a "${LOG_FILE}"
echo "Steering coefficient: ${STEERING_COEF}" | tee -a "${LOG_FILE}"
echo "Start time: $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Initialize counters
TOTAL_CONFIGS=0
BASELINE_PASS=0
VECTOR_PASS=0
STEERING_PASS=0
FAILED_CONFIGS=0

# Initialize summary
echo "Full E2E Configuration Test Summary" > "${SUMMARY_FILE}"
echo "===================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Test all configurations
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
    MODEL_SHORT=$(get_model_short_name "${MODEL_NAME}")
    
    # Check if model supports steering (only Qwen models for now)
    SUPPORTS_STEERING=false
    if [[ "${MODEL_NAME}" =~ "Qwen" ]] || [[ "${MODEL_NAME}" =~ "qwen" ]]; then
        SUPPORTS_STEERING=true
    fi
    
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Testing Model: ${MODEL_NAME}" | tee -a "${LOG_FILE}"
    echo "Steering support: ${SUPPORTS_STEERING}" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    for DATASET in "${DATASETS[@]}"; do
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
        CONFIG_NAME="${MODEL_SHORT} + ${DATASET}"
        
        echo "----------------------------------------" | tee -a "${LOG_FILE}"
        echo "Config ${TOTAL_CONFIGS}/24: ${CONFIG_NAME}" | tee -a "${LOG_FILE}"
        echo "Start: $(date)" | tee -a "${LOG_FILE}"
        
        TEST_DIR="${TEST_ROOT}/${MODEL_SHORT}/${DATASET}"
        mkdir -p "${TEST_DIR}"
        
        BASELINE_OK=false
        VECTOR_OK=false
        STEERING_OK=false
        
        # ============================================
        # STEP 1: Baseline Evaluation
        # ============================================
        echo "[1/4] Running baseline evaluation..." | tee -a "${LOG_FILE}"
        
        if python eval_MATH_vllm.py \
            --model_name_or_path "${MODEL_PATH}" \
            --dataset "${DATASET}" \
            --max_tokens "${MAX_TOKENS}" \
            --max_examples ${TEST_EXAMPLES} \
            --save_dir "${TEST_DIR}" \
            --use_chat_format \
            --remove_bos \
            >> "${LOG_FILE}" 2>&1; then
            
            if [ -f "${TEST_DIR}/predictions.jsonl" ]; then
                BASELINE_OK=true
                BASELINE_PASS=$((BASELINE_PASS + 1))
                echo "  ✓ Baseline evaluation passed" | tee -a "${LOG_FILE}"
            else
                echo "  ✗ Baseline evaluation failed (no predictions)" | tee -a "${LOG_FILE}"
                echo "✗ ${CONFIG_NAME} - Baseline failed (no predictions)" >> "${SUMMARY_FILE}"
                FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                continue
            fi
        else
            echo "  ✗ Baseline evaluation failed (error)" | tee -a "${LOG_FILE}"
            echo "✗ ${CONFIG_NAME} - Baseline failed (error)" >> "${SUMMARY_FILE}"
            FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
            continue
        fi
        
        # ============================================
        # STEP 2: Vector Generation (if model supports steering)
        # ============================================
        if [ "${SUPPORTS_STEERING}" = true ]; then
            echo "[2/4] Generating hidden states for correct samples..." | tee -a "${LOG_FILE}"
            
            if python hidden_analysis.py \
                --model_path "${MODEL_PATH}" \
                --data_dir "${TEST_DIR}" \
                --data_path "${TEST_DIR}/predictions.jsonl" \
                --type correct \
                >> "${LOG_FILE}" 2>&1; then
                
                echo "  ✓ Hidden states for correct samples generated" | tee -a "${LOG_FILE}"
            else
                echo "  ✗ Failed to generate hidden states for correct samples" | tee -a "${LOG_FILE}"
                echo "✗ ${CONFIG_NAME} - Vector gen failed (correct)" >> "${SUMMARY_FILE}"
                FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                continue
            fi
            
            echo "[3/4] Generating hidden states for incorrect samples..." | tee -a "${LOG_FILE}"
            
            if python hidden_analysis.py \
                --model_path "${MODEL_PATH}" \
                --data_dir "${TEST_DIR}" \
                --data_path "${TEST_DIR}/predictions.jsonl" \
                --type incorrect \
                >> "${LOG_FILE}" 2>&1; then
                
                echo "  ✓ Hidden states for incorrect samples generated" | tee -a "${LOG_FILE}"
            else
                echo "  ✗ Failed to generate hidden states for incorrect samples" | tee -a "${LOG_FILE}"
                echo "✗ ${CONFIG_NAME} - Vector gen failed (incorrect)" >> "${SUMMARY_FILE}"
                FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                continue
            fi
            
            echo "[4/4] Generating steering vector..." | tee -a "${LOG_FILE}"
            
            if python vector_generation.py \
                --data_dir "${TEST_DIR}" \
                --prefixs correct incorrect \
                --layers ${STEERING_LAYER} \
                --save_prefix "test_full" \
                --overwrite \
                >> "${LOG_FILE}" 2>&1; then
                
                VECTOR_PATH="${TEST_DIR}/vector_test_full/layer_${STEERING_LAYER}_transition_reflection_steervec.pt"
                
                if [ -f "${VECTOR_PATH}" ]; then
                    VECTOR_OK=true
                    VECTOR_PASS=$((VECTOR_PASS + 1))
                    echo "  ✓ Vector generation passed" | tee -a "${LOG_FILE}"
                else
                    echo "  ✗ Vector file not created" | tee -a "${LOG_FILE}"
                    echo "✗ ${CONFIG_NAME} - Vector gen failed (no file)" >> "${SUMMARY_FILE}"
                    FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                    continue
                fi
            else
                echo "  ✗ Vector generation failed" | tee -a "${LOG_FILE}"
                echo "✗ ${CONFIG_NAME} - Vector gen failed (error)" >> "${SUMMARY_FILE}"
                FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                continue
            fi
            
            # ============================================
            # STEP 3: Steering Evaluation
            # ============================================
            echo "[5/5] Running steering evaluation..." | tee -a "${LOG_FILE}"
            
            if python eval_MATH_steering.py \
                --model_name_or_path "${MODEL_PATH}" \
                --dataset "${DATASET}" \
                --max_tokens "${MAX_TOKENS}" \
                --max_examples ${TEST_EXAMPLES} \
                --batch_size 1 \
                --save_dir "${TEST_DIR}/steering_test" \
                --steering \
                --steering_vector "${VECTOR_PATH}" \
                --steering_layer ${STEERING_LAYER} \
                --steering_coef ${STEERING_COEF} \
                --use_chat_format \
                --remove_bos \
                >> "${LOG_FILE}" 2>&1; then
                
                # Check if predictions file exists anywhere in the steering_test directory
                PRED_FILE=$(find "${TEST_DIR}/steering_test" -name "predictions.jsonl" 2>/dev/null | head -n 1)
                if [ -n "${PRED_FILE}" ] && [ -f "${PRED_FILE}" ]; then
                    STEERING_OK=true
                    STEERING_PASS=$((STEERING_PASS + 1))
                    echo "  ✓ Steering evaluation passed" | tee -a "${LOG_FILE}"
                    echo "✓ ${CONFIG_NAME} - FULL E2E PASS" >> "${SUMMARY_FILE}"
                else
                    echo "  ✗ Steering evaluation failed (no predictions)" | tee -a "${LOG_FILE}"
                    echo "✗ ${CONFIG_NAME} - Steering failed (no predictions)" >> "${SUMMARY_FILE}"
                    FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                    continue
                fi
            else
                echo "  ✗ Steering evaluation failed (error)" | tee -a "${LOG_FILE}"
                echo "✗ ${CONFIG_NAME} - Steering failed (error)" >> "${SUMMARY_FILE}"
                FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
                continue
            fi
        else
            # Model doesn't support steering, only baseline counts as pass
            echo "[2-5] Skipping vector generation and steering (model not supported)" | tee -a "${LOG_FILE}"
            echo "✓ ${CONFIG_NAME} - Baseline only (no steering support)" >> "${SUMMARY_FILE}"
        fi
        
        echo "End: $(date)" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
    done
    
    echo "" | tee -a "${LOG_FILE}"
done

# Print summary
echo "" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "Full E2E Test Summary" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "Total configurations tested: ${TOTAL_CONFIGS}" | tee -a "${LOG_FILE}"
echo "Baseline passed: ${BASELINE_PASS}" | tee -a "${LOG_FILE}"
echo "Vector generation passed: ${VECTOR_PASS}" | tee -a "${LOG_FILE}"
echo "Steering evaluation passed: ${STEERING_PASS}" | tee -a "${LOG_FILE}"
echo "Failed: ${FAILED_CONFIGS}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Add summary to summary file
echo "" >> "${SUMMARY_FILE}"
echo "===================================" >> "${SUMMARY_FILE}"
echo "Total: ${TOTAL_CONFIGS}" >> "${SUMMARY_FILE}"
echo "Baseline passed: ${BASELINE_PASS}" >> "${SUMMARY_FILE}"
echo "Vector generation passed: ${VECTOR_PASS}" >> "${SUMMARY_FILE}"
echo "Steering evaluation passed: ${STEERING_PASS}" >> "${SUMMARY_FILE}"
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

