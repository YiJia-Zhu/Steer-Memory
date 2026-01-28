#!/bin/bash

# SEAL Baseline - Results Summarizer (Standalone)
# This script only generates summaries from existing results

# Parse arguments
RESULTS_DIR="${1}"

if [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 <results_directory>"
    echo ""
    echo "Example:"
    echo "  $0 results/baseline_full_20260122_111424"
    echo ""
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory $RESULTS_DIR does not exist"
    exit 1
fi

# Change to SEAL_1 directory
cd "$(dirname "$0")/.." || exit 1

echo "========================================="
echo "SEAL Baseline - Results Summarizer"
echo "========================================="
echo "Results directory: ${RESULTS_DIR}"
echo ""
echo "Generating summaries..."
echo "========================================="
echo ""

# Run the summarizer
python scripts/summarize_results.py \
    --output_dir "${RESULTS_DIR}" \
    --models "Qwen2.5-3B,Qwen2.5-7B,DS-R1-1.5B,DS-R1-7B" \
    --datasets "aime_2024,aime25,amc23,arc-c,math500,openbookqa"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Summary Generation Completed!"
    echo "========================================="
    echo ""
    echo "Results location: ${RESULTS_DIR}"
    echo "Summary files:"
    echo "  - ${RESULTS_DIR}/results_summary.csv"
    echo "  - ${RESULTS_DIR}/results_by_model.csv"
    echo "  - ${RESULTS_DIR}/results_by_dataset.csv"
    echo "  - ${RESULTS_DIR}/overall_statistics.json"
    echo "========================================="
else
    echo ""
    echo "❌ Summary generation failed with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE

