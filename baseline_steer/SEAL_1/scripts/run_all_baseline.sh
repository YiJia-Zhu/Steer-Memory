#!/bin/bash

# Batch script to run SEAL baseline on all 24 configurations
# 4 models × 6 datasets = 24 configurations

# Specify which GPU to use (set to specific GPU or leave empty to use all available)
# Example: export CUDA_VISIBLE_DEVICES=0  (use only GPU 0)
# Example: export CUDA_VISIBLE_DEVICES=0,1  (use GPU 0 and 1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Model paths
MODEL_ROOT="/storage/szn_data/model_weights"
MODELS=(
    "DeepSeek-R1-Distill-Qwen-1.5B"
    "DeepSeek-R1-Distill-Qwen-7B"
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-7B-Instruct"
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

# Configuration
MAX_TOKENS=16384
USE_CHAT_FORMAT="--use_chat_format"
REMOVE_BOS="--remove_bos"

# Results root directory
RESULTS_ROOT="results/baseline"

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

# Create results directory
mkdir -p "${RESULTS_ROOT}"

# Log file
LOG_FILE="${RESULTS_ROOT}/run_all_baseline.log"
echo "Starting baseline evaluation: $(date)" > "${LOG_FILE}"

# Run all configurations
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
    MODEL_SHORT=$(get_model_short_name "${MODEL_NAME}")
    
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Model: ${MODEL_NAME} (${MODEL_SHORT})" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    for DATASET in "${DATASETS[@]}"; do
        echo "----------------------------------------" | tee -a "${LOG_FILE}"
        echo "Dataset: ${DATASET}" | tee -a "${LOG_FILE}"
        echo "Start time: $(date)" | tee -a "${LOG_FILE}"
        
        # Set save directory
        SAVE_DIR="${RESULTS_ROOT}/${MODEL_SHORT}/${DATASET}"
        mkdir -p "${SAVE_DIR}"
        
        # Run evaluation
        echo "Running evaluation..." | tee -a "${LOG_FILE}"
        python eval_MATH_vllm.py \
            --model_name_or_path "${MODEL_PATH}" \
            --dataset "${DATASET}" \
            --max_tokens "${MAX_TOKENS}" \
            --save_dir "${SAVE_DIR}" \
            ${USE_CHAT_FORMAT} \
            ${REMOVE_BOS} \
            2>&1 | tee -a "${LOG_FILE}"
        
        EXIT_CODE=${PIPESTATUS[0]}
        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "✓ Evaluation completed successfully" | tee -a "${LOG_FILE}"
        else
            echo "✗ Evaluation failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
        fi
        
        echo "End time: $(date)" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
    done
done

echo "All baseline evaluations completed: $(date)" | tee -a "${LOG_FILE}"
echo "Results saved in: ${RESULTS_ROOT}" | tee -a "${LOG_FILE}"

