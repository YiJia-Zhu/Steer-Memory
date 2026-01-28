#!/bin/bash
# 测试数据分割是否正确实现
# 只测试 Qwen2.5-3B + math500，验证：
# 1. Baseline使用前100个样本
# 2. Steering使用后400个样本（索引100-499）

set -e  # 遇到错误立即退出

# 确保在正确的目录
cd /storage/zhangx_data/steer_memory_baseline/SEAL_1

# 设置使用GPU 0
export CUDA_VISIBLE_DEVICES=0

# 配置
MODEL="Qwen2.5-3B-Instruct"
MODEL_PATH="/storage/szn_data/model_weights/${MODEL}"
DATASET="math500"
OUTPUT_DIR="results/test_data_split"
MAX_TOKENS=16384

echo "=========================================="
echo "测试数据分割 - ${MODEL} + ${DATASET}"
echo "=========================================="
echo "模型: ${MODEL_PATH}"
echo "数据集: ${DATASET}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# 清理旧结果
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Step 1: Baseline评估 (前100个样本，用于向量提取)
echo "Step 1/3: Baseline评估 (前100个样本)..."
python eval_MATH_vllm.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --max_tokens ${MAX_TOKENS} \
    --save_dir "${OUTPUT_DIR}/baseline" \
    --max_examples 100 \
    --offset 0 \
    --use_chat_format \
    --remove_bos

# 检查结果
if [ ! -f "${OUTPUT_DIR}/baseline/predictions.jsonl" ]; then
    echo "❌ Baseline评估失败"
    exit 1
fi

BASELINE_COUNT=$(wc -l < "${OUTPUT_DIR}/baseline/predictions.jsonl")
echo "✓ Baseline完成: ${BASELINE_COUNT} 个样本"

if [ "${BASELINE_COUNT}" -ne 100 ]; then
    echo "⚠️  警告: 期望100个样本，实际得到${BASELINE_COUNT}个"
fi

echo ""

# Step 2: 生成向量
echo "Step 2/3: 生成向量..."

# 2.1 提取correct样本的hidden states
python hidden_analysis.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${OUTPUT_DIR}/baseline" \
    --data_path "${OUTPUT_DIR}/baseline/predictions.jsonl" \
    --type correct

# 2.2 提取incorrect样本的hidden states
python hidden_analysis.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${OUTPUT_DIR}/baseline" \
    --data_path "${OUTPUT_DIR}/baseline/predictions.jsonl" \
    --type incorrect

# 2.3 生成steering vector
python vector_generation.py \
    --data_dir "${OUTPUT_DIR}/baseline" \
    --prefixs correct incorrect \
    --layers 20 \
    --save_prefix seal \
    --overwrite

VECTOR_PATH="${OUTPUT_DIR}/baseline/vector_seal/layer_20_transition_reflection_steervec.pt"
if [ ! -f "${VECTOR_PATH}" ]; then
    echo "❌ 向量生成失败"
    exit 1
fi

echo "✓ 向量生成完成"
echo ""

# Step 3: Steering评估 (后400个样本，索引100-499)
echo "Step 3/3: Steering评估 (后400个样本，从索引100开始)..."
python eval_MATH_steering.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --max_tokens ${MAX_TOKENS} \
    --batch_size 1 \
    --save_dir "${OUTPUT_DIR}/steering" \
    --max_examples 400 \
    --offset 100 \
    --steering \
    --steering_vector "${VECTOR_PATH}" \
    --steering_layer 20 \
    --steering_coef 1.0 \
    --use_chat_format \
    --remove_bos

# 检查结果
if [ ! -f "${OUTPUT_DIR}/steering/predictions.jsonl" ]; then
    echo "❌ Steering评估失败"
    exit 1
fi

STEERING_COUNT=$(wc -l < "${OUTPUT_DIR}/steering/predictions.jsonl")
echo "✓ Steering完成: ${STEERING_COUNT} 个样本"

if [ "${STEERING_COUNT}" -ne 400 ]; then
    echo "⚠️  警告: 期望400个样本，实际得到${STEERING_COUNT}个"
fi

echo ""
echo "=========================================="
echo "✅ 数据分割测试完成！"
echo "=========================================="
echo "Baseline (训练集): ${BASELINE_COUNT} 个样本 (期望100)"
echo "Steering (测试集): ${STEERING_COUNT} 个样本 (期望400)"
echo "=========================================="
echo ""
echo "验证数据不重复："
echo "- Baseline使用: 索引 0-99"
echo "- Steering使用: 索引 100-499"
echo ""
echo "结果保存在: ${OUTPUT_DIR}"

