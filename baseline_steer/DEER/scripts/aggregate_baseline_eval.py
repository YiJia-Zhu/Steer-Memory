#!/usr/bin/env python
"""
Aggregate evaluation metrics for DEER runs across the baseline model/dataset grid.
Uses the same model/dataset specs as scripts/run_deer_baselines.sh (Steer-Memory layout)
and writes a CSV with accuracy/token stats so you don't have to check files one by one.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from transformers import AutoTokenizer

from utils.answer_extractor import extract_gold, extract_pred
from utils.data_loader import load_data
from utils.grader import check_is_correct
from utils.utils import load_jsonl

# Keep specs in sync with scripts/run_deer_baselines.sh (Steer-Memory subsets).
# Format: dataset, eval_split, max_eval (None means use full split).
DATASET_SPECS = [
    ("math500", "test", 400),
    ("aime_2024", "train", 20),
    ("amc23", "test", 30),
    ("aime25", "test", 20),
    ("arc-c", "validation", None),
    ("openbookqa", "validation", None),
]

# Model key, model path (basename determines output folder name).
STEER_ROOT = Path(os.environ.get("STEER_ROOT", "/private/zhenningshi/Steer-Memory-114"))
MODEL_SPECS = [
    ("ds_r1_qwen_1p5b", str(STEER_ROOT / "huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B")),
    ("qwen2p5_3b", str(STEER_ROOT / "huggingface_models/Qwen2.5-3B-Instruct")),
    ("ds_r1_qwen_7b", str(STEER_ROOT / "huggingface_models/DeepSeek-R1-Distill-Qwen-7B")),
    ("qwen2p5_7b", str(STEER_ROOT / "huggingface_models/Qwen2.5-7B-Instruct")),
]

DEFAULT_OUTPUT_ROOT = SCRIPT_ROOT / "outputs"
DEFAULT_DATASET_ROOT = STEER_ROOT / "datasets"


def read_jsonl_list(file_path: str) -> List[dict]:
    return list(load_jsonl(file_path))


def evaluate_outputs(
    data_name: str,
    examples: List[dict],
    file_outputs: List[dict],
    tokenizer,
    k: int = 1,
) -> Dict[str, Optional[float]]:
    example_by_idx = {ex["idx"]: ex for ex in examples} if examples else {}
    outputs_have_idx = all("idx" in fo for fo in file_outputs)

    correct_cnt = 0
    pass_at_k_list: List[float] = []
    evaluated_outputs: List[dict] = []
    skipped_without_gold = 0
    missing_example = 0

    for i, output in enumerate(file_outputs):
        example = None
        if outputs_have_idx and example_by_idx:
            example = example_by_idx.get(output.get("idx"))
            if example is None:
                missing_example += 1
        elif i < len(examples):
            example = examples[i]

        gold_field = None
        if example is not None:
            gold_field = example.get("answer", "")
        if gold_field in [None, ""]:
            gold_field = output.get("gold_answer")
        if gold_field in [None, ""]:
            skipped_without_gold += 1
            continue

        gt_ans = extract_gold(data_name, str(gold_field))
        if gt_ans is None:
            skipped_without_gold += 1
            continue

        generated_responses = output.get("generated_responses", [])
        generated_answers = [extract_pred(data_name, resp) for resp in generated_responses]
        is_correct_list = [
            check_is_correct(pred, gt_ans) if pred is not None else False
            for pred in generated_answers
        ]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1

        output["generated_answers"] = generated_answers
        output["gold_answer"] = gt_ans
        output["is_correct"] = is_correct
        output["answers_correctness"] = is_correct_list
        evaluated_outputs.append(output)

        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)

    total_eval = len(evaluated_outputs)

    response_lengths = []
    token_nums = []
    for data in evaluated_outputs:
        if not data.get("generated_responses"):
            continue
        response_lengths.append(len(data["generated_responses"][0].split()))
        token_nums.append(len(tokenizer(data["generated_responses"][0])["input_ids"]))

    avg_words = (sum(response_lengths) / total_eval) if total_eval and response_lengths else 0.0
    avg_tokens = (sum(token_nums) / total_eval) if total_eval and token_nums else 0.0

    metrics = {
        "total_outputs": len(file_outputs),
        "evaluated": total_eval,
        "correct": correct_cnt,
        "acc": (correct_cnt / total_eval) if total_eval else 0.0,
        "pass_at_k": (sum(pass_at_k_list) / len(pass_at_k_list)) if pass_at_k_list else None,
        "avg_words": avg_words,
        "avg_tokens": avg_tokens,
        "skipped_without_gold": skipped_without_gold,
        "missing_example": missing_example,
    }
    return metrics


def pick_generation_file(output_dir: str) -> Optional[str]:
    candidates = glob.glob(os.path.join(output_dir, "*.jsonl"))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root where generation files are stored",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Root of datasets (Steer-Memory layout)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(SCRIPT_ROOT / "outputs/baseline_eval.csv"),
        help="Where to save the aggregated CSV",
    )
    parser.add_argument("--k", type=int, default=1, help="k for pass@k")
    parser.add_argument("--verbose", action="store_true",default=True, help="Print progress for each model/dataset")
    args = parser.parse_args()

    csv_dir = os.path.dirname(args.csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    tokenizer_cache: Dict[str, any] = {}
    dataset_cache: Dict[Tuple[str, str, Optional[int], str], List[dict]] = {}
    rows: List[Dict[str, any]] = []

    for model_key, model_path in MODEL_SPECS:
        model_dir = os.path.basename(os.path.normpath(model_path))
        if model_path not in tokenizer_cache:
            if args.verbose:
                print(f"[load tokenizer] {model_key} from {model_path}")
            tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = tokenizer_cache[model_path]

        for dataset, eval_split, max_eval in DATASET_SPECS:
            cache_key = (dataset, eval_split, max_eval, args.dataset_root)
            if cache_key not in dataset_cache:
                if args.verbose:
                    print(f"[load data] dataset={dataset} split={eval_split} max_eval={max_eval}")
                examples = load_data(dataset, eval_split, args.dataset_root)
                if max_eval is not None:
                    examples = examples[:max_eval]
                dataset_cache[cache_key] = examples
            examples = dataset_cache[cache_key]

            output_dir = os.path.join(args.output_root, model_dir, dataset)
            generation_file = pick_generation_file(output_dir)
            if generation_file is None:
                if args.verbose:
                    print(f"[missing] model={model_key} dataset={dataset} split={eval_split} -> no jsonl in {output_dir}")
                rows.append(
                    {
                        "model_key": model_key,
                        "model_dir": model_dir,
                        "dataset": dataset,
                        "split": eval_split,
                        "max_eval": max_eval if max_eval is not None else "",
                        "generation_file": "",
                        "status": "missing",
                        "total_outputs": 0,
                        "evaluated": 0,
                        "correct": 0,
                        "acc": 0.0,
                        "pass_at_k": "",
                        "avg_words": "",
                        "avg_tokens": "",
                        "skipped_without_gold": "",
                        "missing_example": "",
                        "dataset_size": len(examples),
                    }
                )
                continue

            file_outputs = read_jsonl_list(generation_file)
            if args.verbose:
                rel_file = os.path.relpath(generation_file)
                print(f"[eval] model={model_key} dataset={dataset} split={eval_split} file={rel_file} n_out={len(file_outputs)} n_gold={len(examples)}")
            metrics = evaluate_outputs(dataset, examples, file_outputs, tokenizer, k=args.k)

            rows.append(
                {
                    "model_key": model_key,
                    "model_dir": model_dir,
                    "dataset": dataset,
                    "split": eval_split,
                    "max_eval": max_eval if max_eval is not None else "",
                    "generation_file": os.path.relpath(generation_file),
                    "status": "ok",
                    "total_outputs": metrics["total_outputs"],
                    "evaluated": metrics["evaluated"],
                    "correct": metrics["correct"],
                    "acc": round(metrics["acc"], 4) if metrics["evaluated"] else 0.0,
                    "pass_at_k": round(metrics["pass_at_k"], 4) if metrics["pass_at_k"] is not None else "",
                    "avg_words": round(metrics["avg_words"], 2) if metrics["evaluated"] else "",
                    "avg_tokens": round(metrics["avg_tokens"], 2) if metrics["evaluated"] else "",
                    "skipped_without_gold": metrics["skipped_without_gold"],
                    "missing_example": metrics["missing_example"],
                    "dataset_size": len(examples),
                }
            )

    fieldnames = [
        "model_key",
        "model_dir",
        "dataset",
        "split",
        "max_eval",
        "dataset_size",
        "generation_file",
        "status",
        "total_outputs",
        "evaluated",
        "correct",
        "acc",
        "pass_at_k",
        "avg_words",
        "avg_tokens",
        "skipped_without_gold",
        "missing_example",
    ]
    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved aggregated metrics for {len(rows)} runs to {args.csv_path}")


if __name__ == "__main__":
    main()
