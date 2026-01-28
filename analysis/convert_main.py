import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('main_full.csv', encoding='utf-8-sig')
df = df.rename(columns={df.columns[0]: '行标签'})

# 定义数据集和它们在CSV中的列索引
datasets = [
    ('AIME 24', 1, 2),
    ('AIME 25', 3, 4),
    ('AMC 23', 5, 6),
    ('MATH500', 9, 10),
    ('ARC-C', 7, 8),
    ('OBQA', 11, 12),
]

# 模型配置 - 使用原始字符串避免转义混乱
models = [
    {'key': 'DeepSeek-R1-Distill-Qwen-1.5B', 'display': r'DeepSeek-R1}\\\textbf{Distill-Qwen}\\\textbf{1.5B'},
    {'key': 'DeepSeek-R1-Distill-Qwen-7B', 'display': r'DeepSeek-R1}\\\textbf{Distill-Qwen}\\\textbf{7B'},
    {'key': 'Qwen2.5-3B-Instruct', 'display': r'Qwen2.5}\\\textbf{3B-Instruct'},
    {'key': 'Qwen2.5-7B-Instruct', 'display': r'Qwen2.5}\\\textbf{7B-Instruct'}
]

# 方法映射 - (csv_key, display_name, is_stir, is_best)
methods = [
    ('greedy', 'Vanilla', False, False),
    ('Self-Discover', 'Self-Discover', False, False),
    ('Self-Consist', 'Self-Consistency', False, False),
    ('deer', 'DEER', False, False),
    ('seal', 'SEAL', False, False),
    ('med', 'STIR', True, False),  # 只保留STIR文字，后面单独加下标
    ('high', 'STIR', True, True)   # 最优结果，需要加粗数值
]

def format_res(acc, tokens, bold=False):
    """格式化结果"""
    if pd.isna(acc) or pd.isna(tokens):
        return "\\res{0.0}{0}"
    try:
        acc_pct = float(acc) * 100
        tokens_int = int(float(tokens))
        if bold:
            return f"\\res{{\\textbf{{{acc_pct:.1f}}}}}{{{tokens_int:,}}}"
        return f"\\res{{{acc_pct:.1f}}}{{{tokens_int:,}}}"
    except:
        return "\\res{0.0}{0}"

def format_avg(acc, tokens, baseline_acc=None, baseline_tokens=None):
    """格式化平均结果"""
    if pd.isna(acc) or pd.isna(tokens):
        return "\\baseavg{0.0}{0}"
    
    try:
        acc = float(acc)
        tokens = float(tokens)
    except:
        return "\\baseavg{0.0}{0}"
    
    acc_pct = acc * 100
    tokens_int = int(tokens)
    
    if baseline_acc is None:
        return f"\\baseavg{{{acc_pct:.1f}}}{{{tokens_int:,}}}"
    
    try:
        baseline_acc = float(baseline_acc)
        baseline_tokens = float(baseline_tokens)
    except:
        return f"\\baseavg{{{acc_pct:.1f}}}{{{tokens_int:,}}}"
    
    delta = (acc - baseline_acc) * 100
    cost_change = ((baseline_tokens - tokens) / baseline_tokens) * 100
    
    if delta >= 0:
        delta_str = f"\\inc{{{delta:.1f}}}"
    else:
        delta_str = f"\\dec{{{abs(delta):.1f}}}"
    
    if cost_change > 0:
        cost_str = f"\\saving{{{int(cost_change)}\\%}}"
    elif cost_change < 0:
        if abs(cost_change) >= 100:
            multiplier = tokens / baseline_tokens
            cost_str = f"\\overhead{{x{multiplier:.1f}}}"
        else:
            cost_str = f"\\overhead{{-{int(abs(cost_change))}\\%}}"
    else:
        cost_str = "\\saving{0\\%}"
    
    return f"\\avgres{{{acc_pct:.1f}}}{{{delta_str}}}{{{tokens_int:,}}}{{{cost_str}}}"

def format_method_name(method_display, is_stir, method_key):
    """格式化方法名 - 返回用于插入\textbf{}中的内容"""
    if not is_stir:
        return method_display
    
    # STIR方法需要特殊处理 - 这里返回 STIR}$_{...}$，其中}用于闭合\textbf{
    if method_key == 'med':
        return "STIR}$_{{{\\alpha=1.0}}}}$"
    elif method_key == 'high':
        return "STIR}$_{{{\\alpha=0.75}}}}$"
    return method_display

# 生成LaTeX
output = []
output.append("\\resizebox{\\textwidth}{!}{%")
output.append("\\begin{tabular}{llccccccc}")
output.append("\\toprule")
output.append("\\multirow{2}{*}{\\textbf{Model}} & \\multirow{2}{*}{\\textbf{Method}} & \\multicolumn{4}{c}{\\textbf{Math Reasoning}} & \\multicolumn{2}{c}{\\textbf{General QA}} & \\textbf{Average Stats} \\\\")
output.append("\\cmidrule(lr){3-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-9}")
output.append("& & \\textbf{AIME 24} & \\textbf{AIME 25} & \\textbf{AMC 23} & \\textbf{MATH500} & \\textbf{ARC-C} & \\textbf{OBQA} & \\textbf{Acc ($\\Delta$) / Cost (Drop\\%)} \\\\")
output.append("\\midrule")

for model in models:
    output.append("")
    output.append(f"% --- Model: {model['key']} ---")
    
    model_row_idx = df[df['行标签'] == model['key']].index
    if len(model_row_idx) == 0:
        continue
    
    start_idx = model_row_idx[0] + 1
    model_methods = []
    for i in range(start_idx, len(df)):
        row_label = df.iloc[i]['行标签']
        if pd.isna(row_label):
            continue
        if row_label in [m['key'] for m in models]:
            break
        model_methods.append(i)
    
    baseline_row = None
    for idx in model_methods:
        if df.iloc[idx]['行标签'] == 'greedy':
            baseline_row = df.iloc[idx]
            break
    
    if baseline_row is None:
        continue
    
    baseline_avg_acc = baseline_row[df.columns[13]]
    baseline_avg_tokens = baseline_row[df.columns[14]]
    
    for row_idx in model_methods:
        row = df.iloc[row_idx]
        method_key = row['行标签']
        
        method_config = None
        for m in methods:
            if m[0] == method_key:
                method_config = m
                break
        
        if method_config is None:
            continue
        
        method_display = method_config[1]
        is_stir = method_config[2]
        is_best = method_config[3]
        
        parts = []
        
        # 格式化方法名
        formatted_method = format_method_name(method_display, is_stir, method_key)
        
        # 添加行首部分
        if is_stir:
            # STIR方法特殊处理
            if method_key == 'med':
                method_text = "\\textbf{STIR}$_{\\alpha=1.0}$"
            else:  # high
                method_text = "\\textbf{STIR}$_{\\alpha=0.75}$"
            
            if row_idx == model_methods[-1]:
                parts.append(f"\\rowcolor{{blue!5}} \\multirow{{-{len(model_methods)}}}{{*}}{{\\shortstack[l]{{\\textbf{{{model['display']}}}}}}} & {method_text}")
            else:
                parts.append(f"\\rowcolor{{blue!5}}  & {method_text}")
        else:
            # 非STIR方法
            if row_idx == model_methods[-1]:
                parts.append(f"\\rowcolor{{blue!5}} \\multirow{{-{len(model_methods)}}}{{*}}{{\\shortstack[l]{{\\textbf{{{model['display']}}}}}}} & {formatted_method}")
            else:
                parts.append(f"& {formatted_method}")
        
        # 添加各数据集结果
        for ds_name, acc_col, token_col in datasets:
            acc = row[df.columns[acc_col]]
            tokens = row[df.columns[token_col]]
            parts.append(format_res(acc, tokens, bold=is_best))
        
        # 添加平均值
        avg_acc = row[df.columns[13]]
        avg_tokens = row[df.columns[14]]
        
        if method_key == 'greedy':
            parts.append(format_avg(avg_acc, avg_tokens))
        else:
            parts.append(format_avg(avg_acc, avg_tokens, baseline_avg_acc, baseline_avg_tokens))
        
        output.append(" & ".join(parts) + " \\\\")
    
    output.append("\\midrule")

if output[-1] == "\\midrule":
    output[-1] = "\\bottomrule"

output.append("\\end{tabular}%")
output.append("}")
output.append("\\end{table*}")

# 保存
output_file = 'latex_table_final.tex'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print(f"LaTeX表格已生成: {output_file}")
print(f"总行数: {len(output)}")