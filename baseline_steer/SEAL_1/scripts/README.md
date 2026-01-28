# SEAL Baseline Batch Evaluation Scripts

This directory contains scripts to run SEAL baseline experiments on 24 configurations (4 models × 6 datasets).

## Models
1. DeepSeek-R1-Distill-Qwen-1.5B
2. DeepSeek-R1-Distill-Qwen-7B
3. Qwen2.5-3B-Instruct
4. Qwen2.5-7B-Instruct

## Datasets
1. aime_2024
2. aime25
3. amc23
4. arc-c (ARC-Challenge)
5. math500 (MATH-500)
6. openbookqa (OpenBookQA)

## Workflow

### Step 1: Run Baseline Evaluation
This runs vanilla (non-steered) evaluation on all configurations to get initial predictions:

```bash
cd /storage/zhangx_data/steer_memory_baseline/SEAL_1
bash scripts/run_all_baseline.sh
```

**Output**: 
- Predictions saved in `results/baseline/{MODEL}/{DATASET}/predictions.jsonl`
- Evaluation metrics in `results/baseline/{MODEL}/{DATASET}/metrics.json`

### Step 2: Generate Steering Vectors
This generates steering vectors for each configuration using the baseline predictions:

```bash
bash scripts/run_all_vector_generation.sh
```

**Requirements**: Step 1 must complete successfully first.

**Process for each configuration**:
1. Extract hidden states from correct predictions (max 100 samples)
2. Extract hidden states from incorrect predictions (max 100 samples)
3. Compute steering vector as: `mean(correct+incorrect) - mean(other)`
4. Save vector for layer 20 (configurable per model)

**Output**: 
- Steering vectors saved in `results/baseline/{MODEL}/{DATASET}/vector_seal_100/`

### Step 3: Run Steering Evaluation
This runs steered evaluation using the generated vectors:

```bash
bash scripts/run_all_steering.sh
```

**Requirements**: Step 2 must complete successfully first.

**Note**: Currently only supports Qwen-based models due to implementation constraints.

**Steering coefficients tested**: 0.5, 1.0, 1.5, 2.0

**Output**: 
- Steered predictions in `results/baseline/{MODEL}/{DATASET}/steering_coef_{COEF}/`

## Configuration

### Key Parameters
- **Max tokens**: 16384 (as per requirements)
- **Max samples for vector generation**: 100 per type (correct/incorrect)
- **Steering layer**: 20 (can be adjusted per model in scripts)
- **Batch size**: 1 (for steering evaluation)

### Modifying Model Paths
Edit the `MODEL_ROOT` variable in each script:
```bash
MODEL_ROOT="/storage/szn_data/model_weights"
```

### Modifying Datasets
Edit the `DATASETS` array in each script to add/remove datasets.

### Modifying Steering Coefficients
Edit the `STEERING_COEFS` array in `run_all_steering.sh`.

## Monitoring Progress

Each script creates a log file in `results/baseline/`:
- `run_all_baseline.log` - Baseline evaluation log
- `run_vector_generation.log` - Vector generation log
- `run_steering.log` - Steering evaluation log

Monitor progress:
```bash
tail -f results/baseline/run_all_baseline.log
```

## GPU Usage

The scripts will automatically use available GPUs. For parallel execution across multiple GPUs, consider:
1. Splitting the configuration list
2. Running separate instances with different GPU assignments
3. Using `CUDA_VISIBLE_DEVICES` environment variable

Example for parallel execution:
```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_subset_1.sh

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 bash scripts/run_subset_2.sh
```

## Expected Runtime

Approximate times per configuration (depends on hardware):
- Baseline evaluation: 10-30 minutes (depends on dataset size and model)
- Vector generation: 5-15 minutes
- Steering evaluation: 10-30 minutes (per coefficient)

Total estimated time for all 24 configurations: **8-24 hours** (sequential execution)

## Troubleshooting

### Missing datasets
Ensure datasets are available at:
```
/storage/zhangx_data/steer_memory_baseline/Steer-Memory-114/datasets/
```

### Model loading errors
Verify model paths in `/storage/szn_data/model_weights/`

### Memory issues
- Reduce batch size
- Use smaller models first
- Enable gradient checkpointing (requires code modification)

### Vector generation fails
Check that `predictions.jsonl` and `math_eval.jsonl` exist and contain valid data.

## Results Analysis

After completion, results can be analyzed using:
```python
import json
import pandas as pd

# Load metrics for a configuration
with open('results/baseline/DS-R1-1.5B/aime_2024/metrics.json') as f:
    metrics = json.load(f)
    print(f"Accuracy: {metrics['acc']:.3f}")
```

## Note on SEAL Method

SEAL (Steerable Reasoning Calibration) works by:
1. Identifying reasoning transitions in model outputs
2. Computing steering vectors from correct vs incorrect examples
3. Applying these vectors during inference to calibrate reasoning

This implementation follows the original SEAL paper methodology.

