# SEAL Baseline - 分布式多GPU运行指南

## 📋 概述

分布式运行器可以自动将24个配置任务（4个模型 × 6个数据集）分配到多张GPU卡上并行运行，支持：

- ✅ 自动任务队列管理
- ✅ 多GPU并行执行
- ✅ 优雅的Ctrl+C中断处理
- ✅ 实时进度监控
- ✅ 完整的日志记录

## 🚀 快速开始

### 1. 使用便捷脚本（推荐）

#### 完整运行（所有样本）
```bash
# 使用GPU 0,1,2,3
bash scripts/run_full_baseline.sh "0,1,2,3"

# 或者指定不同的GPU
bash scripts/run_full_baseline.sh "0,1"

# 指定GPU和输出目录
bash scripts/run_full_baseline.sh "4,5,6,7" "results/baseline_h100"
```

#### 快速测试（每个数据集10个样本）
```bash
# 使用GPU 0,1进行快速测试
bash scripts/run_quick_test_distributed.sh "0,1"

# 使用所有4张卡
bash scripts/run_quick_test_distributed.sh "0,1,2,3"
```

### 2. 直接使用Python脚本（高级）

```bash
python run_baseline_distributed.py \
    --gpus "0,1,2,3" \
    --output_dir "results/baseline_full" \
    --models "Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct,DeepSeek-R1-Distill-Qwen-1.5B,DeepSeek-R1-Distill-Qwen-7B" \
    --datasets "aime_2024,aime25,amc23,arc-c,math500,openbookqa" \
    --max_tokens 16384 \
    --steering_layer 20 \
    --steering_coef 1.0
```

## 📊 工作原理

### 任务分配机制

1. **任务队列**：所有24个配置任务放入队列
2. **Worker进程**：每张GPU启动一个worker进程
3. **动态分配**：Worker完成一个任务后，自动从队列取下一个任务
4. **负载均衡**：快速完成的配置会让GPU更快地处理下一个任务

### 示例：4张GPU的工作流程

```
GPU 0: [Task 1] → [Task 5] → [Task 9]  → [Task 13] → ...
GPU 1: [Task 2] → [Task 6] → [Task 10] → [Task 14] → ...
GPU 2: [Task 3] → [Task 7] → [Task 11] → [Task 15] → ...
GPU 3: [Task 4] → [Task 8] → [Task 12] → [Task 16] → ...
```

## 🎯 完整Pipeline

每个配置会依次执行：

1. **Baseline评估** (`eval_MATH_vllm.py`)
   - 使用vLLM进行推理
   - 生成 `predictions.jsonl`

2. **Vector生成** (`hidden_analysis.py` + `vector_generation.py`)
   - 提取正确/错误样本的hidden states
   - 计算steering vector
   - 生成 `vector_seal/layer_20_transition_reflection_steervec.pt`

3. **Steering评估** (`eval_MATH_steering.py`)
   - 应用steering vector进行推理
   - 生成 `steering_eval/predictions.jsonl`

## 📁 输出结构

```
results/baseline_full/
├── run_summary.json              # 总体运行摘要
├── Qwen2.5-3B/
│   ├── aime_2024/
│   │   ├── run.log               # 详细日志
│   │   ├── predictions.jsonl     # Baseline结果
│   │   ├── metrics.json          # Baseline准确率
│   │   ├── math_eval.jsonl       # 评估详情
│   │   ├── hidden_correct/       # 正确样本hidden states
│   │   ├── hidden_incorrect/     # 错误样本hidden states
│   │   ├── vector_seal/          # Steering vectors
│   │   └── steering_eval/        # Steering评估结果
│   │       └── ...
│   ├── aime25/
│   └── ...
├── Qwen2.5-7B/
├── DS-R1-1.5B/
└── DS-R1-7B/
```

## 🛑 中断和恢复

### 优雅中断（Ctrl+C）

按 `Ctrl+C` 会：
1. ✅ 捕获中断信号
2. ✅ 停止接受新任务
3. ✅ 终止所有正在运行的GPU进程
4. ✅ 保存当前进度到summary文件
5. ✅ 清理所有资源

```bash
# 运行中按Ctrl+C
$ bash scripts/run_full_baseline.sh "0,1,2,3"
...
^C
===========================================================
🛑 Ctrl+C detected! Shutting down gracefully...
===========================================================
Terminating process 12345...
Terminating process 12346...
✓ All processes stopped
```

### 恢复运行

如果需要恢复中断的运行：

1. **查看已完成的配置**：检查 `results/baseline_full/` 中哪些已完成
2. **修改配置**：从 `--models` 或 `--datasets` 中移除已完成的
3. **重新运行**：使用相同的 `--output_dir`

```bash
# 示例：只运行剩余的模型
python run_baseline_distributed.py \
    --gpus "0,1,2,3" \
    --output_dir "results/baseline_full" \
    --models "DeepSeek-R1-Distill-Qwen-7B" \
    --datasets "aime_2024,aime25,amc23,arc-c,math500,openbookqa"
```

## 🔧 高级配置

### 自定义参数

```bash
python run_baseline_distributed.py \
    --gpus "0,1,2,3,4,5,6,7" \          # 8张H100
    --output_dir "results/baseline_h100" \
    --models "Qwen2.5-3B-Instruct" \     # 只测试一个模型
    --datasets "aime_2024,amc23" \       # 只测试两个数据集
    --max_examples 50 \                  # 每个数据集50个样本
    --max_tokens 32768 \                 # 更长的token限制
    --steering_layer 15 \                # 使用第15层
    --steering_coef 2.0                  # 更强的steering系数
```

### 只运行特定配置

```bash
# 只测试Qwen模型（支持steering）
python run_baseline_distributed.py \
    --gpus "0,1" \
    --models "Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct" \
    --datasets "aime_2024,aime25,amc23,arc-c,math500,openbookqa"

# 只测试数学数据集
python run_baseline_distributed.py \
    --gpus "0,1,2,3" \
    --datasets "aime_2024,aime25,amc23,math500"
```

## 📈 监控进度

### 实时输出

脚本会实时显示：
- 每个GPU正在处理的任务
- 每个步骤的完成状态
- 总体进度（已完成/总数）
- 已用时间和预计剩余时间（ETA）

```
============================================================
[GPU 0] Task 5/24: Qwen2.5-3B + math500
Start: 2026-01-22 10:15:30
============================================================

[GPU 0] [Qwen2.5-3B + math500] Step 1/3: Running baseline evaluation...
[GPU 0] [Qwen2.5-3B + math500] ✓ Baseline evaluation completed
[GPU 0] [Qwen2.5-3B + math500] Step 2/3: Generating steering vectors...
[GPU 0] [Qwen2.5-3B + math500] ✓ Vector generation completed
[GPU 0] [Qwen2.5-3B + math500] Step 3/3: Running steering evaluation...
[GPU 0] [Qwen2.5-3B + math500] ✓ Steering evaluation completed

[GPU 0] [Qwen2.5-3B + math500] ✅ ALL STEPS COMPLETED
End: 2026-01-22 10:45:20

============================================================
Progress: 5/24 (✓ 5, ✗ 0)
Elapsed: 15.5 min, ETA: 58.5 min
============================================================
```

### 查看详细日志

每个配置的详细日志保存在：
```bash
# 查看特定配置的日志
tail -f results/baseline_full/Qwen2.5-3B/aime_2024/run.log
```

## 📋 完成后检查结果

### 查看总结

```bash
cat results/baseline_full/run_summary.json
```

```json
{
  "total_tasks": 24,
  "completed": 24,
  "failed": 0,
  "total_time_seconds": 7230.5,
  "avg_time_per_task": 301.3,
  "success_configs": [
    "Qwen2.5-3B + aime_2024",
    "Qwen2.5-3B + aime25",
    ...
  ],
  "failed_configs": [],
  "gpus_used": [0, 1, 2, 3],
  "timestamp": "2026-01-22T12:30:45.123456"
}
```

### 收集所有结果

```bash
# 查看所有baseline准确率
find results/baseline_full -name "metrics.json" -path "*/metrics.json" -not -path "*/steering_eval/*" -exec echo {} \; -exec cat {} \; -exec echo \;

# 查看所有steering准确率
find results/baseline_full -path "*/steering_eval/*/metrics.json" -exec echo {} \; -exec cat {} \; -exec echo \;
```

## 💡 最佳实践

### H100服务器推荐配置

```bash
# 8张H100，完整运行
bash scripts/run_full_baseline.sh "0,1,2,3,4,5,6,7" "results/baseline_h100"

# 4张H100，快速测试
bash scripts/run_quick_test_distributed.sh "0,1,2,3" "results/test_h100"
```

### 调试建议

1. **先做快速测试**：
   ```bash
   bash scripts/run_quick_test_distributed.sh "0,1" "results/debug"
   ```

2. **检查单个配置**：
   ```bash
   python run_baseline_distributed.py \
       --gpus "0" \
       --models "Qwen2.5-3B-Instruct" \
       --datasets "amc23" \
       --max_examples 5
   ```

3. **查看详细错误**：
   ```bash
   # 查看失败配置的日志
   cat results/baseline_full/Qwen2.5-3B/aime_2024/run.log
   ```

## ⚠️ 常见问题

### Q1: 如何确定使用哪些GPU？

```bash
# 查看GPU状态
nvidia-smi

# 查看GPU显存使用
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

### Q2: 任务会重复运行吗？

不会。脚本会覆盖已有的结果目录，如果想避免重复运行，请：
- 使用不同的 `--output_dir`
- 或者从 `--models`/`--datasets` 中移除已完成的配置

### Q3: 出现OOM错误怎么办？

减少并行任务数：
```bash
# 只使用2张GPU
bash scripts/run_full_baseline.sh "0,1"
```

或减少样本数：
```bash
python run_baseline_distributed.py --gpus "0,1,2,3" --max_examples 100
```

### Q4: 如何查看单个配置的准确率？

```bash
# Baseline准确率
cat results/baseline_full/Qwen2.5-3B/aime_2024/metrics.json

# Steering准确率
find results/baseline_full/Qwen2.5-3B/aime_2024/steering_eval -name "metrics.json" -exec cat {} \;
```

## 🎓 完整示例

### 场景1: H100服务器完整运行

```bash
cd /storage/zhangx_data/steer_memory_baseline/SEAL_1

# 使用8张H100，完整运行所有24个配置
bash scripts/run_full_baseline.sh "0,1,2,3,4,5,6,7" "results/baseline_h100_full"

# 预计时间：约2-4小时（取决于数据集大小）
```

### 场景2: 快速验证

```bash
cd /storage/zhangx_data/steer_memory_baseline/SEAL_1

# 使用2张GPU，每个数据集10个样本
bash scripts/run_quick_test_distributed.sh "0,1" "results/quick_test"

# 预计时间：约20-30分钟
```

### 场景3: 自定义配置

```bash
cd /storage/zhangx_data/steer_memory_baseline/SEAL_1

# 只测试大模型，使用4张GPU
python run_baseline_distributed.py \
    --gpus "0,1,2,3" \
    --models "Qwen2.5-7B-Instruct,DeepSeek-R1-Distill-Qwen-7B" \
    --datasets "aime_2024,aime25,amc23,arc-c,math500,openbookqa" \
    --output_dir "results/large_models_only" \
    --max_tokens 16384

# 12个配置，预计时间：约1-2小时
```

## 📞 获取帮助

```bash
# 查看所有可用参数
python run_baseline_distributed.py --help
```

---

**Happy Experimenting! 🚀**

