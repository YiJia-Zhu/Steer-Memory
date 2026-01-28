#!/usr/bin/env python3
"""
测试所有数据集的offset功能
"""

from data_loader import load_dataset_seal

# 数据集配置（与run_baseline_distributed.py中的DATASET_CONFIG一致）
DATASET_CONFIG = {
    'math500': {'train': 100, 'eval': 400, 'total': 500},
    'aime_2024': {'train': 10, 'eval': 20, 'total': 30},
    'aime25': {'train': 10, 'eval': 20, 'total': 30},
    'amc23': {'train': 10, 'eval': 30, 'total': 40},
    'arc-c': {'train': 100, 'eval': 199, 'total': 299},
    'openbookqa': {'train': 100, 'eval': 400, 'total': 500},
}

print('='*80)
print('测试所有数据集的数据分割功能')
print('='*80)
print()

all_passed = True
results = []

for dataset_name, config in DATASET_CONFIG.items():
    print(f'📊 测试数据集: {dataset_name}')
    print('-'*80)
    
    try:
        # 加载训练集（前N个样本）
        train_data = load_dataset_seal(
            dataset_name, 
            'datasets', 
            max_examples=config['train'], 
            offset=0
        )
        train_count = len(train_data)
        
        # 加载测试集（后M个样本）
        eval_data = load_dataset_seal(
            dataset_name, 
            'datasets', 
            max_examples=config['eval'], 
            offset=config['train']
        )
        eval_count = len(eval_data)
        
        # 验证数据不重复
        is_different = True
        if train_count > 0 and eval_count > 0:
            is_different = train_data[-1]["question"] != eval_data[0]["question"]
        
        # 检查结果
        train_ok = train_count == config['train']
        eval_ok = eval_count == config['eval']
        total_ok = (train_count + eval_count) == config['total']
        all_ok = train_ok and eval_ok and total_ok and is_different
        
        status = '✅' if all_ok else '❌'
        results.append({
            'dataset': dataset_name,
            'passed': all_ok,
            'train': train_count,
            'eval': eval_count,
            'no_overlap': is_different
        })
        
        print(f'  训练集: {train_count}/{config["train"]} {"✅" if train_ok else "❌"}')
        print(f'  测试集: {eval_count}/{config["eval"]} {"✅" if eval_ok else "❌"}')
        print(f'  总计: {train_count + eval_count}/{config["total"]} {"✅" if total_ok else "❌"}')
        print(f'  数据不重复: {"✅" if is_different else "❌"}')
        print(f'  结果: {status}')
        
        if not all_ok:
            all_passed = False
            
    except Exception as e:
        print(f'  ❌ 错误: {str(e)}')
        results.append({
            'dataset': dataset_name,
            'passed': False,
            'train': 0,
            'eval': 0,
            'no_overlap': False
        })
        all_passed = False
    
    print()

# 打印总结
print('='*80)
print('测试结果总结')
print('='*80)
print()

print(f'{"数据集":<15} {"训练集":<12} {"测试集":<12} {"总计":<10} {"不重复":<8} {"状态":<6}')
print('-'*80)

for r in results:
    cfg = DATASET_CONFIG[r['dataset']]
    train_str = f"{r['train']}/{cfg['train']}"
    eval_str = f"{r['eval']}/{cfg['eval']}"
    total_str = f"{r['train']+r['eval']}/{cfg['total']}"
    overlap_str = "✅" if r['no_overlap'] else "❌"
    status_str = "✅ PASS" if r['passed'] else "❌ FAIL"
    
    print(f"{r['dataset']:<15} {train_str:<12} {eval_str:<12} {total_str:<10} {overlap_str:<8} {status_str:<6}")

print()
print('='*80)

if all_passed:
    print('🎉 所有测试通过！数据分割功能正确实现！')
else:
    print('⚠️  部分测试失败，请检查数据集或配置')

print('='*80)

