#!/usr/bin/env python3
"""
Aggregate cross-dataset generalization runs into one CSV.

Expected inputs (per run):
- outputs/<run_name>/<run_id>/config_resolved.json
- outputs/<run_name>/<run_id>/tables/main_results_single.csv

Only runs that set eval.artifact_run_dir (i.e., reuse another dataset's memory)
are included in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _resolve_run_dir(outputs_root: Path, run_name: str, run_id: str | None) -> Path | None:
    run_root = outputs_root / run_name
    if not run_root.exists():
        return None

    if run_id:
        if run_id.lower() == "latest":
            latest = run_root / "LATEST"
            if latest.exists():
                rid = latest.read_text(encoding="utf-8").strip()
                if rid:
                    cand = run_root / rid
                    if cand.is_dir():
                        return cand
        return run_root / run_id

    latest = run_root / "LATEST"
    if latest.exists():
        rid = latest.read_text(encoding="utf-8").strip()
        if rid:
            cand = run_root / rid
            if cand.is_dir():
                return cand

    dirs = [p for p in run_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def _resolve_artifact_root(run_root: Path, artifact_run_dir: str | None) -> Path | None:
    if artifact_run_dir is None:
        return None
    s = str(artifact_run_dir).strip()
    if s == "":
        return None
    p = Path(s)
    if not p.is_absolute():
        p = run_root / p
    if p.is_dir():
        return p
    if p.name.lower() == "latest":
        latest = p.parent / "LATEST"
        if latest.exists():
            rid = latest.read_text(encoding="utf-8").strip()
            if rid:
                cand = p.parent / rid
                if cand.is_dir():
                    return cand
    return p if p.exists() else None


def _guess_mem_dataset_from_name(run_name: str) -> str | None:
    m = re.search(r"mem_([a-z0-9]+)_on_", run_name)
    if m:
        return m.group(1)
    return None


def _load_main_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return rows
            base_fields = {"dataset", "split", "T_max"}
            for row in reader:
                if not row:
                    continue
                data = {h: row[i] if i < len(row) else "" for i, h in enumerate(header)}
                for idx, col in enumerate(header):
                    if col in base_fields:
                        continue
                    acc_raw = row[idx] if idx < len(row) else ""
                    if acc_raw is None or str(acc_raw).strip() == "":
                        continue
                    try:
                        acc_val: float | None = float(acc_raw)
                    except Exception:
                        acc_val = None
                    rows.append(
                        {
                            "dataset": data.get("dataset"),
                            "split": data.get("split"),
                            "T_max": data.get("T_max"),
                            "method": col,
                            "acc": acc_val,
                        }
                    )
    except FileNotFoundError:
        return rows
    return rows


def _load_diagnostics(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load tokens/overhead diagnostics per method (greedy/esm)."""
    diag: dict[str, dict[str, Any]] = {}
    try:
        # Pick the first diagnostics_T*.csv if multiple exist.
        candidates = sorted(run_dir.glob("tables/diagnostics_T*.csv"))
        if not candidates:
            return diag
        df = pd.read_csv(candidates[0])
    except Exception:
        return diag

    for _, row in df.iterrows():
        method = str(row.get("method", "")).strip().lower()
        if not method:
            continue
        diag[method] = {
            "tokens_mean": row.get("tokens_mean"),
            "tokens_p50": row.get("tokens_p50"),
            "tokens_p90": row.get("tokens_p90"),
            "tokens_max": row.get("tokens_max"),
            "budget_mean": row.get("budget_mean"),
            "overhead_mean": row.get("overhead_mean"),
            "overhead_p90": row.get("overhead_p90"),
            "overhead_max": row.get("overhead_max"),
        }
    return diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cross-dataset generalization runs.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Outputs root directory.")
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="generalization",
        help="Only summarize run_names starting with this prefix.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id to load under each run_name (use 'latest' to follow LATEST).",
    )
    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Optional substring filter applied to run_name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the summary CSV (default: outputs/_generalization[/<run_id>]/generalization_summary.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.expanduser()
    if not outputs_root.is_dir():
        raise SystemExit(f"[error] outputs_root not found: {outputs_root}")

    run_names = []
    for p in outputs_root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(args.run_name_prefix):
            continue
        if args.contains and args.contains not in p.name:
            continue
        run_names.append(p.name)

    rows: list[dict[str, Any]] = []
    for run_name in sorted(run_names):
        run_dir = _resolve_run_dir(outputs_root, run_name, args.run_id)
        if run_dir is None:
            continue

        cfg = _load_json(run_dir / "config_resolved.json")
        if cfg is None:
            continue

        eval_cfg = cfg.get("eval") or {}
        artifact_root = _resolve_artifact_root(run_dir, eval_cfg.get("artifact_run_dir"))
        if artifact_root is None:
            # Offline-only; skip.
            continue

        artifact_cfg = _load_json(artifact_root / "config_resolved.json")
        mem_dataset = None
        if artifact_cfg:
            mem_dataset = (artifact_cfg.get("task") or {}).get("dataset") or None
        if mem_dataset is None:
            mem_dataset = _guess_mem_dataset_from_name(run_name)

        # Diagnostics (tokens/overhead per method).
        diag = _load_diagnostics(run_dir)
        # Probe tokens from config (same for all rows in this run).
        probe_tokens = None
        online_cfg = cfg.get("online") or {}
        if "probe_tokens" in online_cfg:
            try:
                probe_tokens = int(online_cfg.get("probe_tokens"))
            except Exception:
                probe_tokens = online_cfg.get("probe_tokens")

        model_name = Path((cfg.get("model") or {}).get("name_or_path", "")).name
        prompt_template = (cfg.get("prompt") or {}).get("template")
        offline_layers = (cfg.get("offline_mine") or {}).get("candidate_layers") or []
        offline_layer = offline_layers[0] if offline_layers else None
        online_k_scale = (cfg.get("online") or {}).get("k_scale")

        main_results = _load_main_results(run_dir / "tables" / "main_results_single.csv")
        for res in main_results:
            # Map main_results method headers to diagnostics' method keys.
            method_key = str(res.get("method", "")).strip().lower()
            diag_key = {
                "greedy-cot": "greedy",
                "greedy": "greedy",
                "esm": "esm",
            }.get(method_key, method_key)
            diag_row = diag.get(diag_key, {})

            rows.append(
                {
                    "run_name": run_name,
                    "run_id": run_dir.name,
                    "model": model_name,
                    "memory_run_name": artifact_root.parent.name if artifact_root else None,
                    "memory_run_id": artifact_root.name if artifact_root else None,
                    "memory_dataset": mem_dataset,
                    "target_dataset": res.get("dataset"),
                    "target_split": res.get("split"),
                    "T_max": res.get("T_max"),
                    "method": res.get("method"),
                    "acc": res.get("acc"),
                    "tokens_mean": diag_row.get("tokens_mean"),
                    "tokens_p50": diag_row.get("tokens_p50"),
                    "tokens_p90": diag_row.get("tokens_p90"),
                    "tokens_max": diag_row.get("tokens_max"),
                    "budget_mean": diag_row.get("budget_mean"),
                    "overhead_mean": diag_row.get("overhead_mean"),
                    "overhead_p90": diag_row.get("overhead_p90"),
                    "overhead_max": diag_row.get("overhead_max"),
                    "probe_tokens": probe_tokens,
                    "online_k_scale": online_k_scale,
                    "offline_candidate_layers": offline_layer,
                    "artifact_run_dir": str(artifact_root) if artifact_root else None,
                    "prompt_template": prompt_template,
                }
            )

    if args.output is None:
        out_root = outputs_root / "_generalization"
        if args.run_id:
            out_root = out_root / args.run_id
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "generalization_summary.csv"
    else:
        out_path = args.output.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "run_id",
        "model",
        "memory_run_name",
        "memory_run_id",
        "memory_dataset",
        "target_dataset",
        "target_split",
        "T_max",
        "method",
        "acc",
        "tokens_mean",
        "tokens_p50",
        "tokens_p90",
        "tokens_max",
        "budget_mean",
        "overhead_mean",
        "overhead_p90",
        "overhead_max",
        "probe_tokens",
        "online_k_scale",
        "offline_candidate_layers",
        "artifact_run_dir",
        "prompt_template",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[ok] saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
