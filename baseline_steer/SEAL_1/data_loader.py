"""
Data loader for various datasets used in SEAL baseline experiments.
Adapted from Steer-Memory-114 project.
"""

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read JSONL file and return list of dicts."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _read_parquet(path: str) -> list[dict[str, Any]]:
    """Read Parquet file and return list of dicts."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "需要读取本地 parquet，但未能导入 pyarrow。请安装：pip install pyarrow\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e
    table = pq.read_table(path)
    return table.to_pylist()


def load_dataset_seal(dataset_name: str, data_root: str, max_examples: int = None, offset: int = 0):
    """
    Load dataset for SEAL baseline.
    
    Args:
        dataset_name: Name of dataset (aime_2024, aime25, amc23, arc-c, math500, openbookqa)
        data_root: Root directory containing datasets
        max_examples: Maximum number of examples to load (None = load all)
        offset: Number of examples to skip from the beginning (for train/eval split)
        
    Returns:
        List of dicts with keys: question, answer, gt (ground truth)
    """
    dataset_name = dataset_name.lower().strip()
    data_root = Path(data_root)
    test_data = []
    
    if dataset_name in {"math500", "math-500"}:
        # Load MATH-500 dataset
        data_path = data_root / "HuggingFaceH4" / "MATH-500" / "test.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"未找到 MATH-500 数据文件: {data_path}")
        
        ds = _read_jsonl(str(data_path))
        for ex in ds:
            # Extract boxed answer from solution
            from answer_extractor import extract_box
            gt = extract_box(ex.get("solution", ""))
            test_data.append({
                "question": ex["problem"],
                "answer": ex.get("solution", ""),
                "gt": gt,
            })
            
    elif dataset_name in {"aime_2024", "aime2024"}:
        # Load AIME 2024 dataset
        data_path = data_root / "HuggingFaceH4" / "aime_2024" / "data" / "train-00000-of-00001.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"未找到 AIME_2024 数据文件: {data_path}")
        
        ds = _read_parquet(str(data_path))
        for ex in ds:
            test_data.append({
                "question": ex["problem"],
                "answer": str(ex.get("answer", "")),
                "gt": str(ex.get("answer", "")),
            })
            
    elif dataset_name in {"aime25", "aime_25", "aime_2025"}:
        # Load AIME 2025 dataset
        data_path = data_root / "math-ai" / "aime25" / "test.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"未找到 AIME25 数据文件: {data_path}")
        
        ds = _read_jsonl(str(data_path))
        for ex in ds:
            q = str(ex.get("problem", ex.get("question", "")))
            a = str(ex.get("answer", ""))
            test_data.append({
                "question": q,
                "answer": a,
                "gt": a,
            })
            
    elif dataset_name in {"amc23", "amc_23", "amc_2023"}:
        # Load AMC 2023 dataset
        data_path = data_root / "math-ai" / "amc23" / "test-00000-of-00001.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"未找到 AMC23 数据文件: {data_path}")
        
        ds = _read_parquet(str(data_path))
        for ex in ds:
            test_data.append({
                "question": str(ex.get("question", "")),
                "answer": str(ex.get("answer", "")),
                "gt": str(ex.get("answer", "")),
            })
            
    elif dataset_name in {"arc-c", "arc_challenge", "arc"}:
        # Load ARC-Challenge dataset
        # NOTE: Following Steer-Memory-114, we use validation split (299 samples) instead of test split (1172 samples)
        data_path = data_root / "allenai" / "ai2_arc" / "ARC-Challenge" / "validation-00000-of-00001.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"未找到 ARC-Challenge 数据文件: {data_path}")
        
        ds = _read_parquet(str(data_path))
        for ex in ds:
            q = ex["question"]
            choices = ex["choices"]
            # Format: {"label": [...], "text": [...]}
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            
            # Format question with choices
            q_with_choices = q + "\nChoices:\n"
            for lab, txt in zip(labels, texts):
                q_with_choices += f"{lab}. {txt}\n"
            
            ans = ex.get("answerKey", "")
            test_data.append({
                "question": q_with_choices.strip(),
                "answer": ans,
                "gt": ans,
            })
            
    elif dataset_name in {"openbookqa", "openbook_qa"}:
        # Load OpenBookQA dataset
        # NOTE: Following Steer-Memory-114, we use validation split (500 samples) instead of test split
        data_path = data_root / "allenai" / "openbookqa" / "main" / "validation-00000-of-00001.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"未找到 OpenBookQA 数据文件: {data_path}")
        
        ds = _read_parquet(str(data_path))
        for ex in ds:
            q = ex.get("question_stem", ex.get("question", ""))
            choices = ex["choices"]
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            
            # Format question with choices
            q_with_choices = q + "\nChoices:\n"
            for lab, txt in zip(labels, texts):
                q_with_choices += f"{lab}. {txt}\n"
            
            ans = ex.get("answerKey", "")
            test_data.append({
                "question": q_with_choices.strip(),
                "answer": ans,
                "gt": ans,
            })
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # Apply offset and truncation
    # This allows splitting data: first N for training/mining, rest for evaluation
    if offset > 0:
        test_data = test_data[offset:]
    
    if max_examples is not None:
        test_data = test_data[:max_examples]
    
    return test_data

