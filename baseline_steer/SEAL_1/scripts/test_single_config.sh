#!/bin/bash

# Test script to verify a single configuration works correctly
# Use this to test before running the full batch

set -e  # Exit on error

# Specify which GPU to use (only GPU 0 is available)
export CUDA_VISIBLE_DEVICES=0

# Configuration for testing (using smallest model and dataset)
MODEL_PATH="/storage/szn_data/model_weights/Qwen2.5-3B-Instruct"
DATASET="amc23"
MAX_TOKENS=16384
TEST_DIR="results/test"

echo "========================================="
echo "SEAL Baseline Configuration Test"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET}"
echo "Test directory: ${TEST_DIR}"
echo ""

# Clean up previous test
if [ -d "${TEST_DIR}" ]; then
    echo "Cleaning up previous test results..."
    rm -rf "${TEST_DIR}"
fi

mkdir -p "${TEST_DIR}"

# Step 1: Test baseline evaluation
echo "========================================="
echo "Step 1: Testing baseline evaluation"
echo "========================================="
python eval_MATH_vllm.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --max_tokens "${MAX_TOKENS}" \
    --max_examples 10 \
    --save_dir "${TEST_DIR}" \
    --use_chat_format \
    --remove_bos

if [ $? -ne 0 ]; then
    echo "✗ Baseline evaluation failed!"
    exit 1
fi

echo "✓ Baseline evaluation successful"
echo ""

# Check if predictions file exists
if [ ! -f "${TEST_DIR}/predictions.jsonl" ]; then
    echo "✗ Predictions file not created!"
    exit 1
fi

echo "✓ Predictions file created: ${TEST_DIR}/predictions.jsonl"
echo ""

# Step 2: Test vector generation (only if using Qwen models with steering support)
if [[ "${MODEL_PATH}" =~ "Qwen" ]] || [[ "${MODEL_PATH}" =~ "qwen" ]]; then
    echo "========================================="
    echo "Step 2: Testing vector generation"
    echo "========================================="
    
    # Generate hidden states for correct samples
    echo "Generating hidden states for correct samples..."
    python hidden_analysis.py \
        --model_path "${MODEL_PATH}" \
        --data_dir "${TEST_DIR}" \
        --data_path "${TEST_DIR}/predictions.jsonl" \
        --type correct
    
    if [ $? -ne 0 ]; then
        echo "✗ Failed to generate hidden states for correct samples"
        exit 1
    fi
    
    echo "✓ Hidden states for correct samples generated"
    
    # Generate hidden states for incorrect samples
    echo "Generating hidden states for incorrect samples..."
    python hidden_analysis.py \
        --model_path "${MODEL_PATH}" \
        --data_dir "${TEST_DIR}" \
        --data_path "${TEST_DIR}/predictions.jsonl" \
        --type incorrect
    
    if [ $? -ne 0 ]; then
        echo "✗ Failed to generate hidden states for incorrect samples"
        exit 1
    fi
    
    echo "✓ Hidden states for incorrect samples generated"
    
    # Generate steering vector
    echo "Generating steering vector..."
    python vector_generation.py \
        --data_dir "${TEST_DIR}" \
        --prefixs correct incorrect \
        --layers 20 \
        --save_prefix "test" \
        --overwrite
    
    if [ $? -ne 0 ]; then
        echo "✗ Vector generation failed"
        exit 1
    fi
    
    echo "✓ Vector generation successful"
    echo ""
    
    # Step 3: Test steering evaluation
    echo "========================================="
    echo "Step 3: Testing steering evaluation"
    echo "========================================="
    
    VECTOR_PATH="${TEST_DIR}/vector_test/layer_20_transition_reflection_steervec.pt"
    
    if [ ! -f "${VECTOR_PATH}" ]; then
        echo "✗ Vector file not found: ${VECTOR_PATH}"
        exit 1
    fi
    
    python eval_MATH_steering.py \
        --model_name_or_path "${MODEL_PATH}" \
        --dataset "${DATASET}" \
        --max_tokens "${MAX_TOKENS}" \
        --max_examples 10 \
        --batch_size 1 \
        --save_dir "${TEST_DIR}/steering_test" \
        --steering \
        --steering_vector "${VECTOR_PATH}" \
        --steering_layer 20 \
        --steering_coef 1.0 \
        --use_chat_format \
        --remove_bos
    
    if [ $? -ne 0 ]; then
        echo "✗ Steering evaluation failed"
        exit 1
    fi
    
    echo "✓ Steering evaluation successful"
else
    echo "Skipping vector generation and steering (model not supported)"
fi

echo ""
echo "========================================="
echo "All tests passed! ✓"
echo "========================================="
echo ""
echo "Test results saved in: ${TEST_DIR}"
echo ""
echo "You can now run the full batch scripts:"
echo "  bash scripts/run_all_baseline.sh"
echo "  bash scripts/run_all_vector_generation.sh"
echo "  bash scripts/run_all_steering.sh"

