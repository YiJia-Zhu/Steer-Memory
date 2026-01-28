

CUDA_VISIBLE_DEVICES='1' \
 python ../vanilla_deer.py \
 --model_name_or_path ./DeepSeek-R1-Distill-Qwen-14B \
 --threshold 0.95 \
 --max_len 16384 \
 --dataset math \
