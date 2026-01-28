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
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _percentile_int(xs: list[int], p: float) -> int | None:
    if not xs:
        return None
    xs_sorted = sorted(int(x) for x in xs)
    if len(xs_sorted) == 1:
        return int(xs_sorted[0])
    # Nearest-rank on [0, n-1]; dependency-free and sufficient for summaries.
    k = int(round((float(p) / 100.0) * float(len(xs_sorted) - 1)))
    k = max(0, min(int(k), len(xs_sorted) - 1))
    return int(xs_sorted[k])


def _safe_mean_int(xs: list[int]) -> float | None:
    if not xs:
        return None
    return float(sum(int(x) for x in xs)) / float(len(xs))


def _parse_finish_reason(raw: str) -> str | None:
    s = str(raw).strip()
    if s == "" or s == "null":
        return None
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        # Minimal unquoting; finish_reason is typically "stop"/"length"/"eos".
        s = s[1:-1]
        s = s.replace('\\"', '"').replace("\\\\", "\\")
    return s


_RE_TOKENS_USED = re.compile(r'"tokens_used"\s*:\s*(\d+)')
_RE_BUDGET_USED = re.compile(r'"budget_used"\s*:\s*(\d+)')
_RE_PROBE_USED = re.compile(r'"probe_tokens_used"\s*:\s*(\d+)')
_RE_FINISH_REASON = re.compile(r'"finish_reason"\s*:\s*(null|"(?:\\.|[^"])*")')
_RE_EVAL_METHOD_T = re.compile(r'^(?P<method>[A-Za-z0-9_-]+)_T(?P<T>\d+)$')


def _read_token_stats_from_per_example(per_example_path: Path, *, T_max: int | None) -> dict[str, Any]:
    """
    Extract token/budget usage stats from eval/*/per_example.jsonl.

    Best-effort and streaming: avoids json parsing (ESM rows can be very large due to `steps` and `text`).
    """
    toks: list[int] = []
    buds: list[int] = []
    probes: list[int] = []
    overhead: list[int] = []

    n = 0
    finish_seen = 0
    trunc = 0

    with per_example_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n += 1

            m_tok = _RE_TOKENS_USED.search(s)
            tok = int(m_tok.group(1)) if m_tok else 0
            toks.append(int(tok))

            m_fr = _RE_FINISH_REASON.search(s)
            fr = _parse_finish_reason(m_fr.group(1)) if m_fr else None
            if m_fr:
                finish_seen += 1

            is_trunc = False
            if fr is not None:
                is_trunc = str(fr).lower() == "length"
            elif T_max is not None:
                is_trunc = int(tok) >= int(T_max)
            trunc += int(is_trunc)

            m_b = _RE_BUDGET_USED.search(s)
            m_p = _RE_PROBE_USED.search(s)
            if m_b and m_p:
                b = int(m_b.group(1))
                p = int(m_p.group(1))
                buds.append(int(b))
                probes.append(int(p))
                overhead.append(max(0, int(b) - int(tok)))

    out: dict[str, Any] = {
        "n_per_example": int(n),
        "tokens_used_mean": _safe_mean_int(toks),
        "tokens_used_p50": _percentile_int(toks, 50),
        "tokens_used_p90": _percentile_int(toks, 90),
        "tokens_used_max": int(max(toks)) if toks else None,
        "trunc_rate": (float(trunc) / float(max(1, n))) if n > 0 else None,
        "finish_reason_seen_rate": (float(finish_seen) / float(max(1, n))) if n > 0 else None,
    }

    if len(buds) == n and n > 0:
        out.update(
            {
                "budget_used_mean": _safe_mean_int(buds),
                "budget_used_p90": _percentile_int(buds, 90),
                "budget_used_max": int(max(buds)) if buds else None,
                "probe_tokens_used_mean": _safe_mean_int(probes),
                "probe_tokens_used_p90": _percentile_int(probes, 90),
                "probe_tokens_used_max": int(max(probes)) if probes else None,
                "overhead_mean": _safe_mean_int(overhead),
                "overhead_p90": _percentile_int(overhead, 90),
                "overhead_max": int(max(overhead)) if overhead else None,
            }
        )
    return out


def _find_esm_per_example(run_dir: Path, T_max: int | None) -> Path | None:
    """
    Return per_example.jsonl under eval/*_T*/ that best matches the eval method.

    Prefers methods listed in config (when available), otherwise defaults to ESM/greedy.
    Tries to match T_max exactly; falls back to the closest (and then largest) T value.
    """
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        return None

    def _parse_eval_dir(d: Path) -> tuple[str | None, int | None]:
        m = _RE_EVAL_METHOD_T.match(d.name)
        if not m:
            return (None, None)
        method = m.group("method")
        try:
            t_val = int(m.group("T"))
        except Exception:
            t_val = None
        return (method, t_val)

    methods_pref: list[str] = []
    cfg = _load_json(run_dir / "config_resolved.json") or {}
    for m in (cfg.get("eval") or {}).get("methods") or []:
        methods_pref.append(str(m).lower())
    for m in ["esm", "greedy"]:
        if m not in methods_pref:
            methods_pref.append(m)

    def _method_rank(method: str | None) -> int:
        if method is None:
            return len(methods_pref)
        try:
            return methods_pref.index(method.lower())
        except ValueError:
            return len(methods_pref)

    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for p in eval_dir.iterdir():
        if not p.is_dir():
            continue
        method, t_val = _parse_eval_dir(p)
        per_example = p / "per_example.jsonl"
        if method is None or not per_example.exists():
            continue
        pref_rank = _method_rank(method)
        t_penalty = abs(int(t_val) - int(T_max)) if (T_max is not None and t_val is not None) else 0
        t_rank = -int(t_val) if t_val is not None else 0
        candidates.append(((pref_rank, t_penalty, t_rank), per_example))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


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


TOKEN_FIELDS: list[str] = [
    "n_per_example",
    "tokens_used_mean",
    "tokens_used_p50",
    "tokens_used_p90",
    "tokens_used_max",
    "budget_used_mean",
    "budget_used_p90",
    "budget_used_max",
    "probe_tokens_used_mean",
    "probe_tokens_used_p90",
    "probe_tokens_used_max",
    "overhead_mean",
    "overhead_p90",
    "overhead_max",
    "trunc_rate",
    "finish_reason_seen_rate",
]


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

        t_max_int = _safe_int(T_max)
        if t_max_int is None:
            t_max_int = _safe_int((cfg.get("decode") or {}).get("max_new_tokens"))

        per_example_path = _find_esm_per_example(job_dir, t_max_int)
        token_stats: dict[str, Any] = {}
        if per_example_path is not None and per_example_path.exists():
            token_stats = _read_token_stats_from_per_example(per_example_path, T_max=t_max_int)

        row_out: dict[str, Any] = {
            "run_name": name,
            "run_id": rid,
            "ablation": ablation,
            "model": model_name,
            "dataset": dataset,
            "split": split,
            "T_max": T_max,
            "acc_greedy": acc_greedy,
            "acc_esm": acc_esm,
            "eval_per_example_path": str(per_example_path) if per_example_path else None,
        }
        row_out.update(token_stats)
        for k in TOKEN_FIELDS:
            row_out.setdefault(k, None)
        rows.append(row_out)

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
        *TOKEN_FIELDS,
        "eval_per_example_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(out_path)


if __name__ == "__main__":
    main()
