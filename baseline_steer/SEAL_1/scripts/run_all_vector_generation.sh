#!/bin/bash

# Script to generate steering vectors for all 24 configurations
# This should be run AFTER run_all_baseline.sh completes successfully

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

# Function to get layer number for steering (different models may have different optimal layers)
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
LOG_FILE="${RESULTS_ROOT}/run_vector_generation.log"
echo "Starting vector generation: $(date)" > "${LOG_FILE}"

# Generate vectors for all configurations
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
    MODEL_SHORT=$(get_model_short_name "${MODEL_NAME}")
    LAYER=$(get_steering_layer "${MODEL_NAME}")
    
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Model: ${MODEL_NAME} (${MODEL_SHORT})" | tee -a "${LOG_FILE}"
    echo "Steering layer: ${LAYER}" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    for DATASET in "${DATASETS[@]}"; do
        echo "----------------------------------------" | tee -a "${LOG_FILE}"
        echo "Dataset: ${DATASET}" | tee -a "${LOG_FILE}"
        echo "Start time: $(date)" | tee -a "${LOG_FILE}"
        
        DATA_DIR="${RESULTS_ROOT}/${MODEL_SHORT}/${DATASET}"
        
        # Check if predictions exist
        if [ ! -f "${DATA_DIR}/predictions.jsonl" ]; then
            echo "✗ Predictions file not found: ${DATA_DIR}/predictions.jsonl" | tee -a "${LOG_FILE}"
            echo "   Skipping vector generation for this configuration" | tee -a "${LOG_FILE}"
            continue
        fi
        
        # Step 1: Generate hidden states for correct samples
        echo "Step 1: Generating hidden states for correct samples..." | tee -a "${LOG_FILE}"
        python hidden_analysis.py \
            --model_path "${MODEL_PATH}" \
            --data_dir "${DATA_DIR}" \
            --data_path "${DATA_DIR}/predictions.jsonl" \
            --type correct \
            2>&1 | tee -a "${LOG_FILE}"
        
        if [ $? -ne 0 ]; then
            echo "✗ Failed to generate hidden states for correct samples" | tee -a "${LOG_FILE}"
            continue
        fi
        
        # Step 2: Generate hidden states for incorrect samples
        echo "Step 2: Generating hidden states for incorrect samples..." | tee -a "${LOG_FILE}"
        python hidden_analysis.py \
            --model_path "${MODEL_PATH}" \
            --data_dir "${DATA_DIR}" \
            --data_path "${DATA_DIR}/predictions.jsonl" \
            --type incorrect \
            2>&1 | tee -a "${LOG_FILE}"
        
        if [ $? -ne 0 ]; then
            echo "✗ Failed to generate hidden states for incorrect samples" | tee -a "${LOG_FILE}"
            continue
        fi
        
        # Step 3: Generate steering vector
        echo "Step 3: Generating steering vector for layer ${LAYER}..." | tee -a "${LOG_FILE}"
        python vector_generation.py \
            --data_dir "${DATA_DIR}" \
            --prefixs correct incorrect \
            --layers ${LAYER} \
            --save_prefix "seal_100" \
            --overwrite \
            2>&1 | tee -a "${LOG_FILE}"
        
        if [ $? -eq 0 ]; then
            echo "✓ Vector generation completed successfully" | tee -a "${LOG_FILE}"
            echo "   Vector saved in: ${DATA_DIR}/vector_seal_100/" | tee -a "${LOG_FILE}"
        else
            echo "✗ Vector generation failed" | tee -a "${LOG_FILE}"
        fi
        
        echo "End time: $(date)" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
    done
done

echo "All vector generation completed: $(date)" | tee -a "${LOG_FILE}"
echo "Results saved in: ${RESULTS_ROOT}" | tee -a "${LOG_FILE}"

