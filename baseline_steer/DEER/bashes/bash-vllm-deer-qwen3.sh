

CUDA_VISIBLE_DEVICES=1 python ../vllm-deer-qwen3.py \
    --model_name_or_path "./Qwen3-4B" \
    --dataset_dir "./data/" \
    --output_path "./outputs" \
    --dataset "math" \
    --threshold 0.95 \
    --max_generated_tokens 16000 \
    --think_ratio 0.8 \
    --policy avg2 \
    --batch_size 2000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
