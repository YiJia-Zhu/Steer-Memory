import os
import json
import random
from pathlib import Path

from datasets import load_dataset, Dataset, concatenate_datasets
from utils.utils import load_jsonl, lower_keys

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


def _read_jsonl(path: Path):
    return list(load_jsonl(path))


def _read_parquet(path: Path):
    if pq is None:
        raise ImportError("pyarrow is required to read parquet files for local datasets.")
    table = pq.read_table(path)
    return table.to_pylist()


def _load_local_extended_dataset(data_name: str, split: str, data_dir: str):
    """
    Load locally cached datasets that follow the Steer-Memory layout.
    Returns None if the dataset is not handled here.
    """
    name = str(data_name).lower()
    root = Path(data_dir)

    def _default_idx(val, fallback):
        try:
            return int(val)
        except Exception:
            return fallback

    if name in {"math500", "math-500"}:
        path = root / "HuggingFaceH4" / "MATH-500" / f"{split}.jsonl"
        if not path.exists():
            return None
        rows = _read_jsonl(path)
        examples = []
        for i, ex in enumerate(rows):
            examples.append(
                {
                    "idx": _default_idx(ex.get("unique_id", ex.get("id", i)), i),
                    "problem": ex.get("problem", ""),
                    "answer": ex.get("solution", ex.get("answer", "")),
                    "subject": ex.get("subject"),
                    "level": ex.get("level"),
                }
            )
        return examples

    if name in {"aime_2024", "aime2024"}:
        path = root / "HuggingFaceH4" / "aime_2024" / "data" / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            return None
        rows = _read_parquet(path)
        examples = []
        for i, ex in enumerate(rows):
            examples.append(
                {
                    "idx": _default_idx(ex.get("id", i), i),
                    "problem": ex.get("problem", ""),
                    "answer": str(ex.get("answer", "")),
                    "year": ex.get("year"),
                }
            )
        return examples

    if name in {"aime25", "aime_25", "aime_2025"}:
        path = root / "math-ai" / "aime25" / f"{split}.jsonl"
        if not path.exists():
            return None
        rows = _read_jsonl(path)
        examples = []
        for i, ex in enumerate(rows):
            examples.append(
                {
                    "idx": _default_idx(ex.get("id", i), i),
                    "problem": ex.get("problem", ex.get("question", "")),
                    "answer": str(ex.get("answer", "")),
                }
            )
        return examples

    if name in {"amc23", "amc_23", "amc_2023"}:
        base_dir = root / "math-ai" / "amc23"
        primary = base_dir / f"{split}-00000-of-00001.parquet"
        path = primary
        if not path.exists() and base_dir.exists():
            candidates = sorted(base_dir.glob(f"{split}-*.parquet"))
            path = candidates[0] if candidates else path
        if not path.exists():
            return None
        rows = _read_parquet(path)
        examples = []
        for i, ex in enumerate(rows):
            examples.append(
                {
                    "idx": _default_idx(ex.get("id", i), i),
                    "problem": str(ex.get("question", "")),
                    "answer": str(ex.get("answer", "")),
                    "options": ex.get("options"),
                }
            )
        return examples

    if name in {"arc-c", "arc_challenge", "arc"}:
        base_dir = root / "allenai" / "ai2_arc" / "ARC-Challenge"
        path = base_dir / f"{split}-00000-of-00001.parquet"
        if not path.exists() and base_dir.exists():
            candidates = sorted(base_dir.glob(f"{split}-*.parquet"))
            path = candidates[0] if candidates else path
        if not path.exists():
            return None
        rows = _read_parquet(path)
        examples = []
        for i, ex in enumerate(rows):
            choices = ex.get("choices", {}) or {}
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            options = {lab: txt for lab, txt in zip(labels, texts)}
            examples.append(
                {
                    "idx": _default_idx(ex.get("id", i), i),
                    "problem": ex.get("question", ""),
                    "answer": ex.get("answerKey", ""),
                    "options": options,
                }
            )
        return examples

    if name in {"openbookqa", "openbook_qa"}:
        base_dir = root / "allenai" / "openbookqa" / "main"
        path = base_dir / f"{split}-00000-of-00001.parquet"
        if not path.exists() and base_dir.exists():
            candidates = sorted(base_dir.glob(f"{split}-*.parquet"))
            path = candidates[0] if candidates else path
        if not path.exists():
            return None
        rows = _read_parquet(path)
        examples = []
        for i, ex in enumerate(rows):
            choices = ex.get("choices", {}) or {}
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            options = {lab: txt for lab, txt in zip(labels, texts)}
            examples.append(
                {
                    "idx": _default_idx(ex.get("id", i), i),
                    "problem": ex.get("question_stem", ex.get("question", "")),
                    "answer": ex.get("answerKey", ""),
                    "options": options,
                }
            )
        return examples

    return None


def load_data(data_name, split, data_dir="./data"):
    """
    Load dataset samples. For math/ARC/openbook-style tasks we first try to load
    locally cached files following the Steer-Memory layout. Falls back to the
    original HF-based loader for legacy datasets.
    """
    data_name_lower = str(data_name).lower()

    extended = _load_local_extended_dataset(data_name_lower, split, data_dir)
    if extended is not None:
        examples = extended
    else:
        data_file = f"{data_dir}/{data_name}/{split}.jsonl"
        if os.path.exists(data_file):
            examples = list(load_jsonl(data_file))
        else:
            if data_name == "math":
                dataset = load_dataset("competition_math", split=split, name="main", cache_dir=f"{data_dir}/temp")
            elif data_name == "theorem-qa":
                dataset = load_dataset("wenhu/TheoremQA", split=split)
            elif data_name == "gsm8k":
                dataset = load_dataset(data_name, split=split)
            elif data_name == "gsm-hard":
                dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
            elif data_name == "svamp":
                dataset = load_dataset("ChilleD/SVAMP", split="train")
                dataset = concatenate_datasets([dataset, load_dataset("ChilleD/SVAMP", split="test")])
            elif data_name == "asdiv":
                dataset = load_dataset("EleutherAI/asdiv", split="validation")
                dataset = dataset.filter(lambda x: ";" not in x["answer"])
            elif data_name == "mawps":
                examples = []
                for sub_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                    sub_examples = list(load_jsonl(f"{data_dir}/mawps/{sub_name}.jsonl"))
                    for example in sub_examples:
                        example["type"] = sub_name
                    examples.extend(sub_examples)
                dataset = Dataset.from_list(examples)
            elif data_name == "finqa":
                dataset = load_dataset("dreamerdeo/finqa", split=split, name="main")
                dataset = dataset.select(random.sample(range(len(dataset)), 1000))
            elif data_name == "tabmwp":
                examples = []
                with open(f"{data_dir}/tabmwp/tabmwp_{split}.json", "r") as f:
                    data_dict = json.load(f)
                    examples.extend(data_dict.values())
                dataset = Dataset.from_list(examples)
                dataset = dataset.select(random.sample(range(len(dataset)), 1000))
            elif data_name == "bbh":
                examples = []
                for sub_name in [
                    "reasoning_about_colored_objects",
                    "penguins_in_a_table",
                    "date_understanding",
                    "repeat_copy_logic",
                    "object_counting",
                ]:
                    with open(f"{data_dir}/bbh/bbh/{sub_name}.json", "r") as f:
                        sub_examples = json.load(f)["examples"]
                        for example in sub_examples:
                            example["type"] = sub_name
                        examples.extend(sub_examples)
                dataset = Dataset.from_list(examples)
            else:
                raise NotImplementedError(data_name)

            examples = list(dataset)
            examples = [lower_keys(example) for example in examples]
            dataset = Dataset.from_list(examples)
            os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
            dataset.to_json(data_file)

    if not examples:
        return []

    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    examples = sorted(examples, key=lambda x: x["idx"])
    return examples
