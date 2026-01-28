#!/bin/bash

# Quick test version - uses only 10 examples per dataset for testing

# Default configuration
DEFAULT_GPUS="0,1,2,3,4,5,6"
DEFAULT_OUTPUT="results/baseline_quick"
DEFAULT_MODEL_ROOT="/private/zhenningshi/model_weights"
DEFAULT_METHODS="deer"
DEFAULT_MAX_TOKENS=16384
ARC_OPEN_MAX_TOKENS=1024
DATASETS="aime_2024,aime25,amc23,arc-c,math500,openbookqa"

# Parse arguments
GPUS="${1:-$DEFAULT_GPUS}"
OUTPUT_BASE="${2:-$DEFAULT_OUTPUT}"
MODEL_ROOT="${3:-$DEFAULT_MODEL_ROOT}"
METHODS="${4:-$DEFAULT_METHODS}"

# Add timestamp to output directory to avoid overwriting previous runs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

echo "========================================="
echo "SEAL Baseline - Quick Test (10 examples)"
echo "========================================="
echo "GPUs: ${GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "Model root: ${MODEL_ROOT}"
echo "Datasets: ${DATASETS}"
echo "Methods: ${METHODS}"
echo ""
echo "NOTE: Make sure you have activated the correct"
echo "      Python environment before running this script"
echo ""
echo "Press Ctrl+C at any time to stop all tasks"
echo "========================================="
echo ""

# Change to SEAL_1 directory
cd "$(dirname "$0")/.." || exit 1

declare -a FAILED_METHODS=()
declare -a SUCCESS_METHODS=()

for METHOD in ${METHODS//,/ }; do
    echo ""
    echo "========================================="
    echo "Running quick test method: ${METHOD}"
    echo "========================================="

    METHOD_OUTPUT="${OUTPUT_DIR}/${METHOD}"

    python run_baseline_distributed.py \
        --gpus "${GPUS}" \
        --output_dir "${METHOD_OUTPUT}" \
        --model_root "${MODEL_ROOT}" \
        --models "Qwen2.5-3B-Instruct,DeepSeek-R1-Distill-Qwen-1.5B" \
        --datasets "${DATASETS}" \
        --max_tokens "${DEFAULT_MAX_TOKENS}" \
        --max_examples 10 \
        --steering_layer 20 \
        --steering_coef 1.0 \
        --method "${METHOD}"

    METHOD_EXIT=$?

    if [ $METHOD_EXIT -eq 0 ]; then
        SUCCESS_METHODS+=("${METHOD}")
    else
        FAILED_METHODS+=("${METHOD}")
    fi
done

if [ ${#FAILED_METHODS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Methods with failures: ${FAILED_METHODS[*]}"
    exit 1
fi

echo ""
echo "✅ Quick test completed for methods: ${SUCCESS_METHODS[*]}"

# Generate combined CSV across all runs under results/
python scripts/summarize_results.py --scan_root "results" --combined_output "results/all_runs_metrics.csv"
COMBINED_EXIT=$?
if [ $COMBINED_EXIT -ne 0 ]; then
    echo "⚠️  Combined summary generation failed for results/"
fi

exit 0
