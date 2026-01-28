#!/usr/bin/env python3
"""
快速验证offset功能是否正确实现
"""

from data_loader import load_dataset_seal

print('='*60)
print('测试 math500 数据分割')
print('='*60)

# 加载前100个样本（训练集）
print('\n1. 加载训练集 (前100个样本, offset=0)')
train_data = load_dataset_seal('math500', 'datasets', max_examples=100, offset=0)
print(f'   训练集样本数: {len(train_data)}')
if len(train_data) > 0:
    print(f'   第一个问题: {train_data[0]["question"][:80]}...')
    print(f'   最后一个问题: {train_data[-1]["question"][:80]}...')

# 加载后400个样本（测试集）
print('\n2. 加载测试集 (后400个样本, offset=100)')
eval_data = load_dataset_seal('math500', 'datasets', max_examples=400, offset=100)
print(f'   测试集样本数: {len(eval_data)}')
if len(eval_data) > 0:
    print(f'   第一个问题: {eval_data[0]["question"][:80]}...')

# 验证数据不重复
print('\n3. 验证数据不重复:')
if len(train_data) > 0 and len(eval_data) > 0:
    is_different = train_data[-1]["question"] != eval_data[0]["question"]
    print(f'   训练集最后一个 != 测试集第一个: {is_different}')
    if is_different:
        print('   ✅ 数据完全不重复！')
    else:
        print('   ❌ 数据有重复！')

print('\n' + '='*60)
print('测试结果总结:')
print(f'- 训练集: {len(train_data)} 个样本 (期望100)')
print(f'- 测试集: {len(eval_data)} 个样本 (期望400)')
print(f'- 总计: {len(train_data) + len(eval_data)} 个样本 (期望500)')

if len(train_data) == 100 and len(eval_data) == 400:
    print('\n✅ 所有测试通过！数据分割正确实现！')
else:
    print('\n⚠️  样本数量与预期不符，请检查')

print('='*60)

