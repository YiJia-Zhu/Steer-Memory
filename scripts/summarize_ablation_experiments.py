#!/usr/bin/env python3
"""
Summarize ablation runs into one CSV.

Assumes run_name pattern: <run_name_prefix>_<model>_<dataset>_<ablation>
and the usual outputs layout with tables/main_results_single.csv.

Usage examples:
  python scripts/summarize_ablation_experiments.py --run-id 20260101_120000
  python scripts/summarize_ablation_experiments.py --run-name-prefix ablation --ablations "use_random_memory,no_probing"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _resolve_run_id(run_dir: Path, run_id_arg: str | None) -> str | None:
    if run_id_arg:
        return run_id_arg
    latest = run_dir / "LATEST"
    if latest.exists():
        rid = latest.read_text(encoding="utf-8").strip()
        if rid:
            return rid
    # Fallback: pick the newest timestamp-like directory.
    candidates = [p.name for p in run_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _read_main_row(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        row = next(r, None)
    if not header or not row:
        return None, header or []
    return dict(zip(header, row)), header


def _basename(s: str | None) -> str | None:
    if s is None:
        return None
    return s.replace("\\", "/").rstrip("/").split("/")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="run_id to use (default: LATEST under each run_name)")
    ap.add_argument("--run-name-prefix", default="ablation", help="run name prefix to match (default: ablation)")
    ap.add_argument("--ablations", default="", help="comma list to filter (default: all)")
    ap.add_argument("--outputs-root", default="outputs", help="outputs root directory")
    ap.add_argument("--out-csv", default=None, help="explicit output csv path")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root).expanduser()
    abl_filter = {a.strip() for a in args.ablations.split(",") if a.strip()} if args.ablations else None

    rows: list[dict[str, Any]] = []
    for run_dir in outputs_root.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if not name.startswith(str(args.run_name_prefix)):
            continue
        parts = name.split("_")
        if len(parts) < 2:
            continue
        ablation = parts[-1]
        if abl_filter is not None and ablation not in abl_filter:
            continue

        rid = _resolve_run_id(run_dir, args.run_id)
        if rid is None:
            continue
        job_dir = run_dir / rid
        main_path = job_dir / "tables" / "main_results_single.csv"
        cfg_path = job_dir / "config_resolved.json"
        row, header = _read_main_row(main_path)
        if row is None:
            continue
        cfg = _load_json(cfg_path) or {}
        model_name = _basename((cfg.get("model") or {}).get("name_or_path"))
        dataset = (cfg.get("task") or {}).get("dataset")
        split = (cfg.get("task") or {}).get("eval_split")
        T_max = row.get("T_max")
        acc_greedy = row.get("Greedy-CoT") or row.get("greedy")
        acc_esm = row.get("ESM") or row.get("esm")

        rows.append(
            {
                "run_name": name,
                "run_id": rid,
                "ablation": ablation,
                "model": model_name,
                "dataset": dataset,
                "split": split,
                "T_max": T_max,
                "acc_greedy": acc_greedy,
                "acc_esm": acc_esm,
            }
        )

    if not rows:
        print("no rows found", flush=True)
        return

    out_path = (
        Path(args.out_csv)
        if args.out_csv
        else (outputs_root / "_ablation" / (args.run_id or "latest") / "ablation_summary.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "run_id",
        "ablation",
        "model",
        "dataset",
        "split",
        "T_max",
        "acc_greedy",
        "acc_esm",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(out_path)


if __name__ == "__main__":
    main()
