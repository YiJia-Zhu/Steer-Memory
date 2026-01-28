#!/bin/bash

# SEAL Baseline - Full Pipeline Runner
# This is a convenience wrapper for the distributed runner

# Default configuration
DEFAULT_GPUS="0,1,2,3,4,5,6"
DEFAULT_OUTPUT="results/baseline_full"
DEFAULT_MODEL_ROOT="/private/zhenningshi/model_weights"
DEFAULT_MAX_TOKENS=16384
ARC_OPEN_MAX_TOKENS=1024
DATASETS="aime_2024,aime25,amc23,arc-c,math500,openbookqa"
# DATASETS="aime_2024"
# DEFAULT_METHODS="vanilla,seal,cod,deer"
DEFAULT_METHODS="deer"

# Parse arguments
GPUS="${1:-$DEFAULT_GPUS}"
OUTPUT_BASE="${2:-$DEFAULT_OUTPUT}"
MODEL_ROOT="${3:-$DEFAULT_MODEL_ROOT}"
METHODS="${4:-$DEFAULT_METHODS}"
# Optional override for max tokens
MAX_TOKENS="${5:-$DEFAULT_MAX_TOKENS}"

# Add timestamp to output directory to avoid overwriting previous runs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

echo "========================================="
echo "SEAL Baseline - Full Pipeline"
echo "========================================="
echo "GPUs: ${GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "Model root: ${MODEL_ROOT}"
echo "Datasets: ${DATASETS}"
echo "Max tokens (default): ${MAX_TOKENS}"
echo "Max tokens (arc/openbookqa): ${ARC_OPEN_MAX_TOKENS}"
echo "Methods: ${METHODS}"
echo ""
echo "NOTE: Make sure you have activated the correct"
echo "      Python environment before running this script"
echo ""
echo "Press Ctrl+C at any time to stop all tasks"
echo "========================================="
echo ""
        # --models "Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct,DeepSeek-R1-Distill-Qwen-1.5B,DeepSeek-R1-Distill-Qwen-7B" \

# Change to SEAL_1 directory
cd "$(dirname "$0")/.." || exit 1

declare -a FAILED_METHODS=()
declare -a SUCCESS_METHODS=()

for METHOD in ${METHODS//,/ }; do
    echo ""
    echo "========================================="
    echo "Running method: ${METHOD}"
    echo "========================================="

    METHOD_OUTPUT="${OUTPUT_DIR}/${METHOD}"

    python run_baseline_distributed.py \
        --gpus "${GPUS}" \
        --output_dir "${METHOD_OUTPUT}" \
        --model_root "${MODEL_ROOT}" \
        --models "Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct,DeepSeek-R1-Distill-Qwen-1.5B,DeepSeek-R1-Distill-Qwen-7B" \
        --datasets "${DATASETS}" \
        --max_tokens "${MAX_TOKENS}" \
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

OVERALL_EXIT=0
if [ ${#FAILED_METHODS[@]} -gt 0 ]; then
    OVERALL_EXIT=1
fi

# If at least one method succeeded, generate results summary
if [ ${#SUCCESS_METHODS[@]} -gt 0 ]; then
    echo ""
    echo "========================================="
    echo "Generating Results Summary"
    echo "========================================="
    echo ""
    
    python scripts/summarize_results.py \
        --output_dir "${OUTPUT_DIR}" \
        --models "Qwen2.5-3B,Qwen2.5-7B,DS-R1-1.5B,DS-R1-7B" \
        --datasets "${DATASETS}" \
        --methods "${METHODS}"
    
    SUMMARY_EXIT_CODE=$?
    
    if [ $SUMMARY_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "✅ Pipeline and Summary Completed Successfully!"
        echo "========================================="
        echo ""
        echo "Results location: ${OUTPUT_DIR}"
        echo "Summary files:"
        echo "  - ${OUTPUT_DIR}/results_summary.csv"
        echo "  - ${OUTPUT_DIR}/results_by_model.csv"
        echo "  - ${OUTPUT_DIR}/results_by_dataset.csv"
        echo "  - ${OUTPUT_DIR}/results_by_method_model.csv"
        echo "  - ${OUTPUT_DIR}/results_by_method_dataset.csv"
        echo "========================================="

        # Generate combined CSV across all runs under results/
        python scripts/summarize_results.py --scan_root "results" --combined_output "results/all_runs_metrics.csv"
        COMBINED_EXIT=$?
        if [ $COMBINED_EXIT -ne 0 ]; then
            echo "⚠️  Combined summary generation failed for results/"
        fi
    else
        echo ""
        echo "⚠️  Pipeline completed but summary generation failed"
        echo "You can manually run: python scripts/summarize_results.py --output_dir ${OUTPUT_DIR} --methods ${METHODS}"
        OVERALL_EXIT=1
    fi
fi

if [ ${#FAILED_METHODS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Methods with failures: ${FAILED_METHODS[*]}"
fi

exit $OVERALL_EXIT
