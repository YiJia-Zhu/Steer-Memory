import json
import os
from collections import defaultdict
import re

base_dir = "/private/zhenningshi/Steer-Memory-114/outputs"

# Collect data
data = defaultdict(lambda: {"tokens": [], "correct": [], "incorrect": []})

# Find all per_example.jsonl files in amc23 experiments
for root, dirs, files in os.walk(base_dir):
    if "amc23" in root and "per_example.jsonl" in files:
        jsonl_path = os.path.join(root, "per_example.jsonl")
        
        # Extract model and hyperparams from path
        path_parts = root.split("/")
        
        # Find the experiment directory (contains "amc23")
        exp_idx = [i for i, p in enumerate(path_parts) if "amc23" in p][0]
        exp_dir = path_parts[exp_idx]
        run_id = path_parts[exp_idx + 1]
        
        # Parse model name
        if "ds_r1_qwen_1p5b" in exp_dir:
            model = "qwen-1.5b"
        elif "ds_r1_qwen_7b" in exp_dir:
            model = "qwen-7b"
        elif "qwen2p5_3b" in exp_dir:
            model = "qwen2.5-3b"
        elif "qwen2p5_7b" in exp_dir:
            model = "qwen2.5-7b"
        else:
            model = "unknown"
        
        # Parse hyperparams from run_id
        L_match = re.search(r'__L([\d.p]+)_', run_id)
        eta_match = re.search(r'_eta([\d.p]+)_', run_id)
        ks_match = re.search(r'_ks([\d.p]+)(?:$|_)', run_id)
        
        L = L_match.group(1).replace('p', '.') if L_match else "?"
        eta = eta_match.group(1).replace('p', '.') if eta_match else "?"
        ks = ks_match.group(1).replace('p', '.') if ks_match else "?"
        
        # Read tokens from JSONL
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        tokens = obj.get("tokens_used", 0)
                        correct = obj.get("correct", False)
                        
                        key = (model, L, eta, ks)
                        data[key]["tokens"].append(tokens)
                        if correct:
                            data[key]["correct"].append(tokens)
                        else:
                            data[key]["incorrect"].append(tokens)
                    except:
                        pass
        except Exception as e:
            print(f"Error reading {jsonl_path}: {e}")
            continue

# Save as JSONL
output_path = "/private/zhenningshi/Steer-Memory-114/analysis/token_statistics_amc23.jsonl"

with open(output_path, 'w') as f:
    for (model, L, eta, ks) in sorted(data.keys()):
        record = {
            "model": model,
            "L": L,
            "eta": eta,
            "ks": ks,
            "tokens": data[(model, L, eta, ks)]["tokens"],
            "correct_tokens": data[(model, L, eta, ks)]["correct"],
            "incorrect_tokens": data[(model, L, eta, ks)]["incorrect"],
            "num_samples": len(data[(model, L, eta, ks)]["tokens"]),
            "num_correct": len(data[(model, L, eta, ks)]["correct"]),
            "num_incorrect": len(data[(model, L, eta, ks)]["incorrect"]),
        }
        f.write(json.dumps(record) + '\n')

print(f"Saved to {output_path}")
print(f"Total configurations: {len(data)}")
print(f"\nFirst few lines:")

with open(output_path, 'r') as f:
    for i, line in enumerate(f):
        if i < 3:
            record = json.loads(line)
            print(f"\nConfig {i+1}:")
            print(f"  Model: {record['model']}, L={record['L']}, eta={record['eta']}, ks={record['ks']}")
            print(f"  Total samples: {record['num_samples']} (correct: {record['num_correct']}, incorrect: {record['num_incorrect']})")
            print(f"  Token sample (first 5): {record['tokens'][:5]}")
        else:
            break