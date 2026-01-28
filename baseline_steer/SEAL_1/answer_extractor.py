"""
Answer extraction utilities for various datasets.
Adapted from Steer-Memory-114 project.
"""

import re
from typing import Optional

# Regex patterns adapted from Steer-Memory-114 for robust choice extraction
# - Hash-line matches are case-insensitive.
# - Plain letter matches are case-sensitive (avoid grabbing stray lowercase "a" in reasoning).
_RE_HASH_ANSWER_LETTER = re.compile(r"####\\s*([A-Ea-e])")
_RE_LAST_LETTER = re.compile(r"\b([A-E])\b")


def extract_box(pred_str: str) -> str:
    """
    Extract answer from LaTeX \\boxed{} notation.
    Used for MATH-style problems.
    """
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def extract_last_number(pred_str: str) -> Optional[str]:
    """
    Extract the last number from a string.
    Used for numerical answer extraction.
    """
    # Remove commas from numbers
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        return numbers[-1]
    return None


def extract_last_letter(pred_str: str) -> Optional[str]:
    """
    Extract the last letter choice (A-E) from a string.
    Used for multiple-choice questions. Supports both plain letters
    and hashed patterns like '#### Answer: C'.
    """
    m = _RE_HASH_ANSWER_LETTER.findall(pred_str)
    if m:
        return m[-1].upper()
    matches = _RE_LAST_LETTER.findall(pred_str)
    if matches:
        return matches[-1].upper()
    return None


def normalize_number(num_str: str) -> str:
    """
    Normalize number string by removing commas and extra whitespace.
    """
    if not num_str:
        return ""
    # Remove commas
    num_str = num_str.replace(",", "")
    # Remove whitespace
    num_str = num_str.strip()
    return num_str


def normalize_math_answer(ans_str: str) -> str:
    """
    Normalize mathematical answer string.
    """
    if not ans_str:
        return ""
    # Remove extra whitespace
    ans_str = ans_str.strip()
    # Remove LaTeX formatting
    ans_str = ans_str.replace("$", "").replace("\\", "")
    # Normalize commas in numbers
    ans_str = re.sub(r"(\d),(\d)", r"\1\2", ans_str)
    return ans_str


def extract_answer_by_dataset(dataset_name: str, text: str) -> Optional[str]:
    """
    Extract answer from generated text based on dataset type.
    
    Args:
        dataset_name: Name of the dataset
        text: Generated text from model
        
    Returns:
        Extracted answer string or None
    """
    dataset_name = dataset_name.lower().strip()
    
    # Math datasets: extract numbers or boxed answers
    if dataset_name in {"math500", "math-500"}:
        # Try boxed first
        boxed = extract_box(text)
        if boxed:
            return normalize_math_answer(boxed)
        # Fall back to last number
        num = extract_last_number(text)
        return normalize_number(num) if num else None
        
    elif dataset_name in {"aime_2024", "aime2024", "aime25", "aime_25", "aime_2025", "amc23", "amc_23", "amc_2023"}:
        # Extract last number
        num = extract_last_number(text)
        return normalize_number(num) if num else None
        
    # Multiple choice datasets: extract letter
    elif dataset_name in {"arc-c", "arc_challenge", "arc", "openbookqa", "openbook_qa"}:
        letter = extract_last_letter(text)
        return letter if letter else None
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def compare_answers(pred: Optional[str], gold: str, dataset_name: str) -> bool:
    """
    Compare predicted answer with ground truth.
    
    Args:
        pred: Predicted answer
        gold: Ground truth answer
        dataset_name: Name of the dataset
        
    Returns:
        True if answers match, False otherwise
    """
    if pred is None:
        return False
    
    dataset_name = dataset_name.lower().strip()
    
    # Normalize both answers
    pred = str(pred).strip()
    gold = str(gold).strip()
    
    # For multiple choice, do case-insensitive comparison
    if dataset_name in {"arc-c", "arc_challenge", "arc", "openbookqa", "openbook_qa"}:
        return pred.upper() == gold.upper()
    
    # For numerical answers
    if dataset_name in {"math500", "math-500", "aime_2024", "aime2024", "aime25", "aime_25", "aime_2025", "amc23", "amc_23", "amc_2023"}:
        # Try to compare as floats if possible
        try:
            pred_num = float(pred.replace(",", ""))
            gold_num = float(gold.replace(",", ""))
            return abs(pred_num - gold_num) < 1e-6
        except (ValueError, AttributeError):
            # Fall back to string comparison
            return normalize_number(pred) == normalize_number(gold)
    
    return pred == gold
