#!/bin/bash

# Script to run SEAL steering evaluation on all 24 configurations
# This should be run AFTER run_all_vector_generation.sh completes successfully

# Specify which GPU to use
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
BATCH_SIZE=1

# Steering coefficients to try
STEERING_COEFS=(0.5 1.0 1.5 2.0)

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

# Function to get layer number for steering
get_steering_layer() {
    case "$1" in
        "DeepSeek-R1-Distill-Qwen-1.5B")
            echo "20"
            ;;
        "DeepSeek-R1-Distill-Qwen-7B")
            echo "20"
            ;;
        "Qwen2.5-3B-Instruct")
            echo "20"
            ;;
        "Qwen2.5-7B-Instruct")
            echo "20"
            ;;
        *)
            echo "20"
            ;;
    esac
}

# Log file
LOG_FILE="${RESULTS_ROOT}/run_steering.log"
echo "Starting steering evaluation: $(date)" > "${LOG_FILE}"

# Run steering for all configurations
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
    MODEL_SHORT=$(get_model_short_name "${MODEL_NAME}")
    LAYER=$(get_steering_layer "${MODEL_NAME}")
    
    # Skip non-Qwen models for now (steering implementation is Qwen-specific)
    if [[ ! "${MODEL_NAME}" =~ "Qwen" ]] && [[ ! "${MODEL_NAME}" =~ "qwen" ]]; then
        echo "Skipping ${MODEL_NAME} - steering only implemented for Qwen models" | tee -a "${LOG_FILE}"
        continue
    fi
    
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Model: ${MODEL_NAME} (${MODEL_SHORT})" | tee -a "${LOG_FILE}"
    echo "Steering layer: ${LAYER}" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    for DATASET in "${DATASETS[@]}"; do
        DATA_DIR="${RESULTS_ROOT}/${MODEL_SHORT}/${DATASET}"
        VECTOR_PATH="${DATA_DIR}/vector_seal_100/layer_${LAYER}_transition_reflection_steervec.pt"
        
        # Check if vector exists
        if [ ! -f "${VECTOR_PATH}" ]; then
            echo "✗ Vector not found: ${VECTOR_PATH}" | tee -a "${LOG_FILE}"
            echo "   Skipping steering for ${DATASET}" | tee -a "${LOG_FILE}"
            continue
        fi
        
        echo "----------------------------------------" | tee -a "${LOG_FILE}"
        echo "Dataset: ${DATASET}" | tee -a "${LOG_FILE}"
        
        for COEF in "${STEERING_COEFS[@]}"; do
            echo "Steering coefficient: ${COEF}" | tee -a "${LOG_FILE}"
            echo "Start time: $(date)" | tee -a "${LOG_FILE}"
            
            SAVE_DIR="${DATA_DIR}/steering_coef_${COEF}"
            mkdir -p "${SAVE_DIR}"
            
            # Run steering evaluation
            echo "Running steering evaluation..." | tee -a "${LOG_FILE}"
            python eval_MATH_steering.py \
                --model_name_or_path "${MODEL_PATH}" \
                --dataset "${DATASET}" \
                --max_tokens "${MAX_TOKENS}" \
                --batch_size ${BATCH_SIZE} \
                --save_dir "${SAVE_DIR}" \
                --steering \
                --steering_vector "${VECTOR_PATH}" \
                --steering_layer ${LAYER} \
                --steering_coef ${COEF} \
                ${USE_CHAT_FORMAT} \
                ${REMOVE_BOS} \
                2>&1 | tee -a "${LOG_FILE}"
            
            EXIT_CODE=${PIPESTATUS[0]}
            if [ ${EXIT_CODE} -eq 0 ]; then
                echo "✓ Steering evaluation completed successfully" | tee -a "${LOG_FILE}"
            else
                echo "✗ Steering evaluation failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
            fi
            
            echo "End time: $(date)" | tee -a "${LOG_FILE}"
            echo "" | tee -a "${LOG_FILE}"
        done
    done
done

echo "All steering evaluations completed: $(date)" | tee -a "${LOG_FILE}"
echo "Results saved in: ${RESULTS_ROOT}" | tee -a "${LOG_FILE}"

