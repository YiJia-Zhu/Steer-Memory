"""
Visualize hidden-state artifacts (mine/library/memory) with t-SNE / UMAP.

The script loads key vectors (and optional delta vectors) from a run directory,
reduces them to 2D, and saves stage-colored scatter plots. It can also attempt
per-example layer trajectories when multiple layers exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
import seaborn as sns

import matplotlib
import numpy as np
import torch
COLORS = sns.color_palette("Paired")
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
matplotlib.use("Agg")  # non-interactive backend for batch runs
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency
    TSNE = None
    StandardScaler = None

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    umap = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs" / "main_ds_r1_qwen_7b_math500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE/UMAP over hidden-state artifacts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Path to outputs/<run_name>/<run_id>. If omitted, use "
            "outputs/main_ds_r1_qwen_7b_math500/LATEST when available."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save figures (default: <run_dir>/figures/latent_states).",
    )
    parser.add_argument(
        "--stages",
        type=str,
        # default="mine,library,memory",
        default="memory",
        help="Comma-separated subset of {mine,library,memory} to plot (default: all).",
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        default=None,
        help="Optional cap per (stage, entry_type, source) to downsample for speed.",
    )
    parser.add_argument(
        "--include-delta",
        action="store_true",
        help="Also plot delta vectors (vector_path) in addition to hidden states (key_path).",
    )
    parser.add_argument("--perplexity", type=float, default=20.0, help="t-SNE perplexity.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reducers.")
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=50,
        help="UMAP n_neighbors (ignored if umap-learn is not installed).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.3,
        help="UMAP min_dist (ignored if umap-learn is not installed).",
    )
    parser.add_argument(
        "--axis-limit",
        type=float,
        default=None,
        help="If set, fix both x/y limits to [-axis_limit, axis_limit] for scatter plots.",
    )
    return parser.parse_args()


def resolve_run_dir(run_dir_arg: Path | None) -> Path:
    if run_dir_arg is not None:
        return run_dir_arg.resolve()

    latest = DEFAULT_RUN_ROOT / "LATEST"
    if latest.exists():
        return (DEFAULT_RUN_ROOT / latest.read_text().strip()).resolve()

    raise ValueError("No --run-dir provided and default run not found.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_artifact_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if raw_path.startswith("outputs/"):
        return (REPO_ROOT / raw_path).resolve()
    return (run_dir / raw_path).resolve()


def standardize(features: np.ndarray) -> np.ndarray:
    if StandardScaler is None:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        return (features - mean) / std
    return StandardScaler().fit_transform(features)


def compute_tsne(features: np.ndarray, seed: int, perplexity: float) -> np.ndarray | None:
    if TSNE is None:
        print("Skipping t-SNE: scikit-learn not installed.")
        return None
    max_perp = max(5.0, float(len(features) - 1))
    eff_perp = float(min(perplexity, max_perp))
    tsne = TSNE(
        n_components=2,
        perplexity=eff_perp,
        init="random",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features)


def compute_umap(
    features: np.ndarray, seed: int, n_neighbors: int, min_dist: float
) -> np.ndarray | None:
    if umap is None:
        print("Skipping UMAP: umap-learn not installed.")
        return None
    n_neighbors = int(min(n_neighbors, max(2, len(features) - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        random_state=seed,
    )
    return reducer.fit_transform(features)


def plot_embedding(
    coords: np.ndarray,
    meta: list[dict[str, Any]],
    out_path: Path,
    title: str,
    subtitle: str,
    pair_edges: list[tuple[int, int, str]] | None = None,
    line_alpha: float = 0.4,
    line_width: float = 0.8,
    axis_limit: float | None = None,
) -> None:
    if coords is None or len(coords) == 0:
        return

    emb_name = "t-SNE" if "tsne" in out_path.stem.lower() else "UMAP"
    stages = sorted({row["stage"] for row in meta})
    entry_types = sorted({row["entry_type"] for row in meta})
    stage_colors = {s: plt.cm.tab10(i % 10) for i, s in enumerate(stages)}
    def lighten(color, factor: float = 0.1):
        r, g, b = color
        return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)

    # right_color = lighten(COLORS[3])  # lighter green
    right_color = "#008200"  # lighter green
    # wrong_color = lighten(COLORS[5])  # lighter red
    wrong_color = "#FF0000"
    # Custom palette: green for Right/Correct, red for Wrong/Incorrect; fallback to stage color.
    entry_colors = {
        "Right": right_color,
        "Correct": right_color,
        "Wrong": wrong_color,
        "Incorrect": wrong_color,
    }
    markers = ["o", "^", "s", "P", "X", "D"]
    type_markers = {t: markers[i % len(markers)] for i, t in enumerate(entry_types)}

    fig, ax = plt.subplots(figsize=(6.65, 4.65))
    for stage in stages:
        for entry_type in entry_types:
            idx = [
                i
                for i, row in enumerate(meta)
                if row["stage"] == stage and row["entry_type"] == entry_type
            ]
            if not idx:
                continue
            pts = coords[idx]
            label_entry_type = entry_type.title()
            color = entry_colors.get(label_entry_type, stage_colors.get(stage, "#666666"))
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=80,
                c=[color],
                marker=type_markers[entry_type],
                alpha=0.6,
                linewidths=0,
                edgecolors="#666666",
                label=label_entry_type,
                zorder=3,
        )

    ax.set_xlabel(f"{emb_name} Dimension 1", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f"{emb_name} Dimension 2", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    if pair_edges:
        edge_color = "#666666"
        for w_idx, r_idx, stage in pair_edges:
            pts = coords[[w_idx, r_idx]]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=edge_color,
                alpha=line_alpha,
                linewidth=line_width,
                zorder=1,
            )
    # if axis_limit is not None:
        # ax.set_xlim(-30, 30)
    ax.set_ylim(-32, 32)
    ax.set_autoscale_on(False)
    legend = ax.legend(
        frameon=True,
        fontsize=20,
        handleheight=0.9,
        handlelength=1.0,
        handletextpad=0.1,
        columnspacing=0.4,
        markerscale=1.6,
        ncol=2,
        loc="upper left",
    )
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_layer_trajectories(
    coords: np.ndarray,
    meta: list[dict[str, Any]],
    out_path: Path,
    max_examples: int = 16,
) -> None:
    if coords is None or len(coords) == 0:
        return

    rows_with_examples = [
        (row, coords[i]) for i, row in enumerate(meta) if row.get("example_id") is not None
    ]
    by_example: dict[str, list[tuple[int, np.ndarray]]] = {}
    for row, pt in rows_with_examples:
        layer = row.get("layer")
        if layer is None:
            continue
        by_example.setdefault(str(row["example_id"]), []).append((int(layer), pt))

    valid = {ex: pts for ex, pts in by_example.items() if len({l for l, _ in pts}) > 1}
    if not valid:
        print("No multi-layer trajectories found; skipping trajectory plot.")
        return

    fig, ax = plt.subplots(figsize=(6.65, 4.65))
    for ex_i, (example_id, pts) in enumerate(valid.items()):
        if ex_i >= max_examples:
            break
        pts_sorted = sorted(pts, key=lambda x: x[0])
        xs = [p[1][0] for p in pts_sorted]
        ys = [p[1][1] for p in pts_sorted]
        layers = [p[0] for p in pts_sorted]
        ax.plot(xs, ys, "-o", linewidth=0.2, markersize=10, alpha=0.8, label=str(example_id))
        for x, y, layer in zip(xs, ys, layers):
            ax.text(x, y, str(layer), fontsize=20, ha="center", va="center")

    ax.set_title("Per-example layer trajectories")
    ax.set_xlabel("dim 1",fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("dim 2",fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    legend = ax.legend(frameon=True, fontsize=20, ncol=2)
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def load_tensor(path: Path) -> np.ndarray:
    """
    Load a torch-saved tensor/ndarray. Handles numpy-saved via torch.save under PyTorch 2.6+.
    """
    import torch.serialization

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        # Allowlist numpy reconstruct used when torch.save(np.ndarray) was called.
        with torch.serialization.safe_globals(["numpy._core.multiarray._reconstruct"]):
            obj = torch.load(str(path), map_location="cpu", weights_only=False)
    return np.asarray(obj, dtype=np.float32).reshape(-1)


def collect_records(
    rows: Iterable[dict[str, Any]],
    stage: str,
    run_dir: Path,
    limit_per_type: int | None,
    include_delta: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    taken: dict[tuple[str, str, str], int] = {}

    for row in rows:
        key_path = resolve_artifact_path(run_dir, str(row["key_path"]))
        if not key_path.exists():
            continue
        arr = load_tensor(key_path)

        entry_type = str(row.get("entry_type", "unknown"))
        key = (stage, entry_type, "key")
        if limit_per_type is None or taken.get(key, 0) < limit_per_type:
            states.append(
                {
                    "stage": stage,
                    "entry_type": entry_type,
                    "pair_id": row.get("pair_id"),
                    "layer": row.get("layer"),
                    "control_point_m": row.get("control_point_m"),
                    "example_id": row.get("example_id"),
                    "vector": arr,
                }
            )
            taken[key] = taken.get(key, 0) + 1

        if include_delta and row.get("vector_path"):
            vec_path = resolve_artifact_path(run_dir, str(row["vector_path"]))
            if not vec_path.exists():
                continue
            delta_arr = load_tensor(vec_path)
            d_key = (stage, "delta", "delta")
            if limit_per_type is None or taken.get(d_key, 0) < limit_per_type:
                deltas.append(
                    {
                        "stage": stage,
                        "entry_type": "delta",
                        "pair_id": row.get("pair_id"),
                        "layer": row.get("layer"),
                        "control_point_m": row.get("control_point_m"),
                        "example_id": row.get("example_id"),
                        "vector": delta_arr,
                    }
                )
                taken[d_key] = taken.get(d_key, 0) + 1

    return states, deltas


def flatten_vectors(records: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.stack([np.asarray(rec["vector"], dtype=np.float32) for rec in records], axis=0)


def build_pair_edges(meta: Sequence[dict[str, Any]]) -> list[tuple[int, int, str]]:
    """
    Return list of (wrong_idx, right_idx, stage) for rows sharing pair_id within the same stage.
    """
    by_stage: dict[str, dict[Any, dict[str, int]]] = {}
    for i, row in enumerate(meta):
        pid = row.get("pair_id")
        if pid is None:
            continue
        stage = str(row["stage"])
        entry_type = str(row.get("entry_type", ""))
        stage_map = by_stage.setdefault(stage, {})
        pair = stage_map.setdefault(pid, {})
        if entry_type == "wrong":
            pair["wrong"] = i
        elif entry_type == "right":
            pair["right"] = i

    edges: list[tuple[int, int, str]] = []
    for stage, pairs in by_stage.items():
        for pair in pairs.values():
            if "wrong" in pair and "right" in pair:
                edges.append((pair["wrong"], pair["right"], stage))
    return edges


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    out_dir = (args.output_dir or (run_dir / "figures" / "latent_states")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_stages = {"mine", "library", "memory"}
    stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    stages = [s for s in stages if s in allowed_stages]
    if not stages:
        raise ValueError(f"Invalid --stages; expected subset of {sorted(allowed_stages)}")

    stage_path_map = {
        "mine": run_dir / "mine" / "candidates.jsonl",
        "library": run_dir / "library" / "library.jsonl",
        "memory": run_dir / "memory" / "entries.jsonl",
    }
    stage_files = {s: stage_path_map[s] for s in stages}

    pair_to_example: dict[Any, Any] = {}
    if "mine" in stage_files:
        if not stage_files["mine"].exists():
            raise FileNotFoundError(f"Missing mine output: {stage_files['mine']}")
        mine_rows = load_jsonl(stage_files["mine"])
        pair_to_example = {
            row["pair_id"]: row.get("example_id") for row in mine_rows if row.get("pair_id") is not None
        }

    all_state_records: list[dict[str, Any]] = []
    all_delta_records: list[dict[str, Any]] = []

    for stage, jsonl_path in stage_files.items():
        if not jsonl_path.exists():
            print(f"Skipping stage {stage}: {jsonl_path} not found.")
            continue
        rows = load_jsonl(jsonl_path)
        if stage != "mine":
            for row in rows:
                if not row.get("example_id") and row.get("pair_id") in pair_to_example:
                    row["example_id"] = pair_to_example[row["pair_id"]]
        states, deltas = collect_records(
            rows=rows,
            stage=stage,
            run_dir=run_dir,
            limit_per_type=args.limit_per_type,
            include_delta=args.include_delta,
        )
        all_state_records.extend(states)
        all_delta_records.extend(deltas)
        print(f"Loaded stage {stage}: {len(states)} states, {len(deltas)} deltas.")

    if not all_state_records:
        raise RuntimeError("No hidden-state vectors loaded; nothing to plot.")

    state_matrix = flatten_vectors(all_state_records)
    state_matrix = standardize(state_matrix)
    pair_edges = build_pair_edges(all_state_records)

    tsne_coords = compute_tsne(state_matrix, seed=args.seed, perplexity=args.perplexity)
    if tsne_coords is not None:
        plot_embedding(
            coords=tsne_coords,
            meta=all_state_records,
            out_path=out_dir / "tsne_states.pdf",
            title="Hidden states (key_path)",
            subtitle=f"stages={sorted({r['stage'] for r in all_state_records})}",
            pair_edges=pair_edges,
            axis_limit=args.axis_limit,
        )

    umap_coords = compute_umap(
        state_matrix,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    if umap_coords is not None:
        plot_embedding(
            coords=umap_coords,
            meta=all_state_records,
            out_path=out_dir / "umap_states.pdf",
            title="Hidden states (key_path)",
            subtitle=f"stages={sorted({r['stage'] for r in all_state_records})}",
            pair_edges=pair_edges,
            axis_limit=args.axis_limit,
        )

    if len(all_delta_records) > 1:
        delta_matrix = flatten_vectors(all_delta_records)
        delta_matrix = standardize(delta_matrix)

        tsne_delta = compute_tsne(delta_matrix, seed=args.seed, perplexity=args.perplexity)
        if tsne_delta is not None:
            plot_embedding(
                coords=tsne_delta,
                meta=all_delta_records,
                out_path=out_dir / "tsne_deltas.pdf",
                title="Delta vectors (vector_path)",
                subtitle=f"stages={sorted({r['stage'] for r in all_delta_records})}",
                axis_limit=args.axis_limit,
            )

        umap_delta = compute_umap(
            delta_matrix,
            seed=args.seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        if umap_delta is not None:
            plot_embedding(
                coords=umap_delta,
                meta=all_delta_records,
                out_path=out_dir / "umap_deltas.pdf",
                title="Delta vectors (vector_path)",
                subtitle=f"stages={sorted({r['stage'] for r in all_delta_records})}",
                axis_limit=args.axis_limit,
            )

    # Layer trajectories only make sense when multiple layers exist; pick mine stage to avoid duplicates.
    mine_records = [rec for rec in all_state_records if rec["stage"] == "mine"]
    if len(mine_records) > 1 and len({r.get("layer") for r in mine_records}) > 1:
        mine_matrix = flatten_vectors(mine_records)
        mine_matrix = standardize(mine_matrix)
        coords = compute_tsne(mine_matrix, seed=args.seed, perplexity=args.perplexity)
        if coords is not None:
            plot_layer_trajectories(
                coords=coords,
                meta=mine_records,
                out_path=out_dir / "tsne_layer_trajectories.pdf",
            )
    else:
        print("Single-layer data detected; skipping layer trajectory plot.")


if __name__ == "__main__":
    main()
