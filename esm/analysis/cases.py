from __future__ import annotations

from pathlib import Path
from typing import Any

from esm.data.loaders import load_task_dataset
from esm.utils.io import read_jsonl, write_text


def _index_by_id(rows: list[dict]) -> dict[str, dict]:
    return {str(r["example_id"]): r for r in rows}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _segment_marker(text: str, start_char: int, seg: dict[str, Any]) -> str:
    parts = [f"m={_safe_int(seg.get('m'), 0)}"]
    if seg.get("layer") is not None:
        parts.append(f"layer={_safe_int(seg.get('layer'), 0)}")
    tool_name = seg.get("tool_name")
    if tool_name:
        parts.append(f"tool={tool_name}")
    marker = f"<<< injected {' '.join(parts)} >>>"
    prev_char = text[start_char - 1] if start_char > 0 and start_char - 1 < len(text) else ""
    next_char = text[start_char] if start_char < len(text) else ""
    prefix = "" if not prev_char or prev_char.isspace() else " "
    suffix = "" if not next_char or next_char.isspace() else " "
    return f"{prefix}{marker}{suffix}"


def render_output_with_injection_markers(row: dict[str, Any], *, max_chars: int | None = None) -> str:
    text = str(row.get("text", "") or "")
    segments = row.get("segments")
    if not isinstance(segments, list) or not segments:
        return text[:max_chars] if max_chars is not None else text

    parts: list[str] = []
    used = 0
    cursor = 0

    def append_piece(piece: str, *, allow_truncate: bool) -> bool:
        nonlocal used
        if piece == "":
            return False
        if max_chars is None:
            parts.append(piece)
            used += len(piece)
            return False
        remaining = int(max_chars) - int(used)
        if remaining <= 0:
            return True
        if len(piece) <= remaining:
            parts.append(piece)
            used += len(piece)
            return False
        if allow_truncate:
            parts.append(piece[:remaining])
            used += remaining
        return True

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = _safe_int(seg.get("start_char"), cursor)
        end = _safe_int(seg.get("end_char"), start)
        start = max(cursor, min(len(text), start))
        end = max(start, min(len(text), end))

        if start > cursor and append_piece(text[cursor:start], allow_truncate=True):
            return "".join(parts)

        if bool(seg.get("injected")):
            marker = _segment_marker(text, start, seg)
            if append_piece(marker, allow_truncate=False):
                return "".join(parts)

        if append_piece(text[start:end], allow_truncate=True):
            return "".join(parts)
        cursor = end

    if cursor < len(text):
        append_piece(text[cursor:], allow_truncate=True)

    return "".join(parts)


def write_case_markdown(
    *,
    run_dir: str | Path,
    dataset: str,
    split: str,
    max_examples: int | None,
    seed: int,
    data_root: str | None,
    T_max: int,
    greedy_tag: str,
    esm_tag: str,
    top_n: int = 5,
) -> str:
    run_dir = Path(run_dir)
    greedy_path = run_dir / "eval" / greedy_tag / "per_example.jsonl"
    esm_path = run_dir / "eval" / esm_tag / "per_example.jsonl"
    if not greedy_path.exists() or not esm_path.exists():
        return ""

    greedy = _index_by_id(read_jsonl(greedy_path))
    esm = _index_by_id(read_jsonl(esm_path))

    exs = load_task_dataset(dataset, split, max_examples, seed, data_root=data_root)
    q_by_id = {str(e.id): e.question for e in exs}

    improve = []
    regress = []
    for ex_id in esm.keys():
        if ex_id not in greedy:
            continue
        g = greedy[ex_id]
        e = esm[ex_id]
        if bool(e.get("correct")) and not bool(g.get("correct")):
            improve.append(ex_id)
        if not bool(e.get("correct")) and bool(g.get("correct")):
            regress.append(ex_id)

    improve = improve[:top_n]
    regress = regress[:top_n]

    lines = []
    lines.append(f"# Case studies (T_max={T_max})")
    lines.append("")
    lines.append("## ESM improves over Greedy")
    lines.append("")
    for idx, ex_id in enumerate(improve, 1):
        g = greedy[ex_id]
        e = esm[ex_id]
        q = q_by_id.get(ex_id, "")
        lines.append(f"### Improve-{idx}: id={ex_id}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(q)
        lines.append("")
        lines.append(f"**Gold**: {e.get('gold')}")
        lines.append("")
        lines.append(f"**Greedy pred**: {g.get('pred')}  | correct={g.get('correct')}")
        lines.append("")
        lines.append("**Greedy output**")
        lines.append("")
        lines.append("```")
        lines.append(str(g.get("text", ""))[:2000])
        lines.append("```")
        lines.append("")
        lines.append(f"**ESM pred**: {e.get('pred')}  | correct={e.get('correct')}")
        lines.append("")
        lines.append("**ESM output (inline injected markers)**")
        lines.append("")
        lines.append("```")
        lines.append(render_output_with_injection_markers(e, max_chars=2000))
        lines.append("```")
        lines.append("")

    lines.append("## ESM regresses vs Greedy")
    lines.append("")
    for idx, ex_id in enumerate(regress, 1):
        g = greedy[ex_id]
        e = esm[ex_id]
        q = q_by_id.get(ex_id, "")
        lines.append(f"### Regress-{idx}: id={ex_id}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(q)
        lines.append("")
        lines.append(f"**Gold**: {e.get('gold')}")
        lines.append("")
        lines.append(f"**Greedy pred**: {g.get('pred')}  | correct={g.get('correct')}")
        lines.append("")
        lines.append("```")
        lines.append(str(g.get("text", ""))[:2000])
        lines.append("```")
        lines.append("")
        lines.append(f"**ESM pred**: {e.get('pred')}  | correct={e.get('correct')}")
        lines.append("")
        lines.append("**ESM output (inline injected markers)**")
        lines.append("")
        lines.append("```")
        lines.append(render_output_with_injection_markers(e, max_chars=2000))
        lines.append("```")
        lines.append("")

    out_dir = run_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cases_T{T_max}.md"
    write_text(out_path, "\n".join(lines) + "\n")
    return str(out_path)


