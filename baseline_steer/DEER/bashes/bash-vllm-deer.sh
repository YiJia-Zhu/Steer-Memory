

CUDA_VISIBLE_DEVICES=1 python ../vllm-deer.py \
    --model_name_or_path "./DeepSeek-R1-Distill-Qwen-14B" \
    --dataset_dir "./data/" \
    --output_path "./outputs" \
    --dataset "math" \
    --threshold 0.95 \
    --max_generated_tokens 16000 \
    --policy avg1 \
    --think_ratio 0.6 \
    --batch_size 2000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
