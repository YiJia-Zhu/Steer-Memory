#!/usr/bin/env python3
"""
Summarize self baselines into one CSV.

Aggregates:
- outputs/<run_name>/<run_id>/tables/main_results_single.csv
- tokens_used_mean from eval/<method_tag>_T*/per_example.jsonl

Args:
  --out-cfg-dir   Path to generated configs (used to enumerate runs)
  --outputs-root  Root of outputs directory
  --baseline-id   Baseline ID (prefix of run_id)
  --summary-out   Output CSV path (default: outputs/_self_baselines/<baseline_id>/summary.csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _tokens_mean(per_example_path: Path) -> tuple[float | None, int]:
    toks: list[float] = []
    try:
        with per_example_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    tok = obj.get("tokens_used")
                    if tok is not None:
                        toks.append(float(tok))
                except Exception:
                    m = re.search(r'"tokens_used"\s*:\s*([0-9.]+)', s)
                    if m:
                        toks.append(float(m.group(1)))
    except FileNotFoundError:
        return None, 0
    return ((sum(toks) / len(toks)) if toks else None, len(toks))


def summarize(out_cfg_dir: Path, outputs_root: Path, baseline_id: str, summary_out: Path) -> int:
    rows: list[dict[str, Any]] = []

    for cfg_path in sorted(out_cfg_dir.glob("*.yaml")):
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        run_name = str(((raw.get("outputs") or {}).get("run_name") or "")).strip()
        if not run_name:
            continue
        eval_cfg = raw.get("eval") or {}
        methods = eval_cfg.get("methods") or []
        if not methods:
            continue
        baseline = str(methods[0]).strip().lower().replace("-", "_")
        run_id = f"{baseline_id}__{baseline}"

        model = (raw.get("model") or {}).get("name_or_path")
        task_cfg = raw.get("task") or {}
        dataset = task_cfg.get("dataset")
        split = task_cfg.get("eval_split")

        run_dir = outputs_root / run_name / run_id
        main_csv = run_dir / "tables" / "main_results_single.csv"
        if not main_csv.exists():
            # Skip incomplete runs
            continue

        with main_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            body = next(r, None)
        if not header or not body or len(header) < 4:
            continue

        # Columns: dataset, split, T_max, <method1>, ...
        t_max = _safe_int(body[2])
        for idx in range(3, len(header)):
            method_name = header[idx]
            acc = _safe_float(body[idx])
            # Pick per_example tokens for the matching eval tag
            method_tag = re.sub(r"[^a-z0-9]+", "_", method_name.lower()).strip("_")
            tag = f"{method_tag}_T{t_max}" if t_max is not None else method_tag
            per_example = run_dir / "eval" / tag / "per_example.jsonl"
            tok_mean, n = _tokens_mean(per_example)
            rows.append(
                {
                    "run_name": run_name,
                    "run_id": run_id,
                    "baseline": baseline,
                    "method": method_name,
                    "model": model,
                    "dataset": body[0] or dataset,
                    "split": body[1] or split,
                    "T_max": t_max,
                    "acc": acc,
                    "n": n,
                    "tokens_used_mean": tok_mean,
                }
            )

    if not rows:
        print("[summary] no completed runs found; summary not written")
        return 0

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "run_id",
        "baseline",
        "method",
        "model",
        "dataset",
        "split",
        "T_max",
        "acc",
        "n",
        "tokens_used_mean",
    ]
    with summary_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[summary] wrote {len(rows)} rows to {summary_out}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-cfg-dir", required=True, type=Path)
    ap.add_argument("--outputs-root", required=True, type=Path)
    ap.add_argument("--baseline-id", required=True, type=str)
    ap.add_argument("--summary-out", required=True, type=Path)
    args = ap.parse_args()

    summarize(
        out_cfg_dir=args.out_cfg_dir,
        outputs_root=args.outputs_root,
        baseline_id=args.baseline_id,
        summary_out=args.summary_out,
    )


if __name__ == "__main__":
    main()
