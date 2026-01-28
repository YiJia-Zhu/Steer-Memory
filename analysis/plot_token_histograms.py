from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
TOKEN_PATH = ANALYSIS_DIR / "token.jsonl"
FIG_DIR = ANALYSIS_DIR / "token_figs"

# Mapping from model key in token.jsonl to the corresponding main run directory.
MODEL_RUN_ROOTS: Dict[str, Path] = {
    "qwen-1.5b": REPO_ROOT / "outputs" / "main_ds_r1_qwen_1p5b_amc23",
    "qwen-7b": REPO_ROOT / "outputs" / "main_ds_r1_qwen_7b_amc23",
    "qwen2.5-3b": REPO_ROOT / "outputs" / "main_qwen2p5_3b_amc23",
    "qwen2.5-7b": REPO_ROOT / "outputs" / "main_qwen2p5_7b_amc23",
}

# Align colors with latent_tsne_umap.py: red for vanilla (wrong) and green for STIR (right).
PALETTE = {"greedy": "#FF0000", "esm": "#008200"}
DISPLAY_NAMES = {"greedy": "Vanilla", "esm": "STIR"}
HIST_ALPHA = 0.5
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 16
FIT_LINE_ALPHA = 0.65
_WARNED_GMM = False


def _slugify(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    pattern = re.compile(r"^\d{8}_\d{6}$")
    for child in sorted(root.iterdir()):
        if child.is_dir() and pattern.match(child.name):
            yield child


def resolve_run_dir(model: str) -> Path:
    root = MODEL_RUN_ROOTS[model]
    latest = root / "LATEST"
    if latest.exists():
        run_id = latest.read_text().strip()
        candidate = root / run_id
        if candidate.exists():
            return candidate
    for run_dir in reversed(list(_iter_run_dirs(root))):
        return run_dir
    raise FileNotFoundError(f"No run directory found under {root}")


def read_token_stats() -> Dict[Tuple[str, str], dict]:
    stats: Dict[Tuple[str, str], dict] = {}
    with TOKEN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["model"], row["tier"])
            stats[key] = row
    return stats


def read_greedy_tokens(model: str) -> Tuple[List[int], int]:
    run_dir = resolve_run_dir(model)
    greedy_dir = run_dir / "eval"
    greedy_paths = list(greedy_dir.glob("greedy_T*/per_example.jsonl"))
    if not greedy_paths:
        raise FileNotFoundError(f"No greedy per_example.jsonl found under {greedy_dir}")
    greedy_path = sorted(greedy_paths)[-1]
    m = re.search(r"_T(\d+)", greedy_path.as_posix())
    t_max = int(m.group(1)) if m else 0

    tokens: List[int] = []
    with greedy_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tokens.append(int(obj.get("tokens_used", 0) or 0))
    return tokens, t_max


def _skewness(data: List[int]) -> float:
    if not data:
        return 0.0
    arr = np.asarray(data, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    if std == 0.0:
        return 0.0
    centered = arr - mean
    m3 = float(np.mean(centered ** 3))
    return m3 / (std ** 3)


def _normal_curve(x: np.ndarray, data: List[int], scale: float) -> np.ndarray:
    if not data:
        return np.zeros_like(x)
    mu = float(np.mean(data))
    sigma = float(np.std(data))
    if sigma == 0:
        return np.zeros_like(x)
    coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
    return coeff * np.exp(-0.5 * ((x - mu) / sigma) ** 2) * scale


def _half_normal_curve(x: np.ndarray, data: List[int], scale: float, loc: float = 0.0) -> np.ndarray:
    """
    Fit a simple half-normal (abs of Gaussian) over non-negative support.
    loc anchors the left edge (defaults to 0); data below loc is ignored.
    """
    if not data:
        return np.zeros_like(x)
    loc = float(loc)
    shifted = np.asarray([d - loc for d in data if d >= loc], dtype=float)
    if len(shifted) == 0:
        return np.zeros_like(x)
    sigma = float(np.sqrt(np.mean(shifted ** 2)))
    if sigma == 0.0:
        return np.zeros_like(x)
    x_shifted = np.asarray(x, dtype=float) - loc
    x_shifted = np.where(x_shifted < 0.0, np.nan, x_shifted)
    coeff = np.sqrt(2.0 / np.pi) / sigma
    curve = coeff * np.exp(-0.5 * (x_shifted / sigma) ** 2)
    curve = np.where(np.isnan(curve), 0.0, curve)
    return curve * scale


def _select_fit_mode(mode: str, data: List[int]) -> str:
    """
    Choose fit mode. In 'auto', prefer half-normal when data is non-negative and
    visibly right-skewed (skewness > 0.5), otherwise fall back to normal.
    """
    mode = mode.lower()
    if mode != "auto":
        return mode
    if not data or min(data) < 0:
        return "normal"
    return "half" if _skewness(data) > 0.5 else "normal"


def _gaussian_mixture_curve(
    x: np.ndarray, data: List[int], scale: float, n_components: int
) -> np.ndarray:
    """
    Fit a 1D Gaussian Mixture and return density scaled to histogram counts.
    """
    global _WARNED_GMM
    if not data or n_components < 1:
        return np.zeros_like(x)
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        if not _WARNED_GMM:
            print("Skipping GMM fit: scikit-learn not installed.")
            _WARNED_GMM = True
        return np.zeros_like(x)

    n_components = min(int(max(1, n_components)), len(data))
    arr = np.asarray(data, dtype=float).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    try:
        gmm.fit(arr)
    except Exception:
        return np.zeros_like(x)
    dens = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    return dens * scale


def plot_histogram(
    model: str,
    tier: str,
    esm_tokens: List[int],
    greedy_tokens: List[int],
    t_max: int,
    bins: int,
    fit_normal: bool,
    fit_mode: str,
    gmm_components: int,
) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    combined = esm_tokens + greedy_tokens

    if combined:
        lo, hi = min(combined), max(combined)
        lo = min(0, lo)  # anchor at 0 to show empty mass near the origin
        if lo == hi:
            lo, hi = 0, hi + 1
        bin_edges = np.linspace(lo, hi, bins + 1)
    else:
        bin_edges = bins

    # 拟合数据：去掉达到 T_max 的样本，并且去掉最右侧 bin 内的样本（通常是截断峰）
    fit_greedy = greedy_tokens
    fit_esm = esm_tokens
    if isinstance(bin_edges, np.ndarray) and len(bin_edges) >= 2:
        right_bin_start = float(bin_edges[-2])
        fit_greedy = [t for t in fit_greedy if t < right_bin_start]
        fit_esm = [t for t in fit_esm if t < right_bin_start]
    if t_max:
        fit_greedy = [t for t in fit_greedy if t < t_max]
        fit_esm = [t for t in fit_esm if t < t_max]
    if not fit_greedy:
        fit_greedy = greedy_tokens
    if not fit_esm:
        fit_esm = esm_tokens

    plt.figure(figsize=(6.65, 4.65))
    colors = PALETTE
    g_counts, _, _ = plt.hist(
        greedy_tokens,
        bins=bin_edges,
        alpha=HIST_ALPHA,
        label="Vanilla",
        color=colors["greedy"],
    )
    e_counts, _, _ = plt.hist(
        esm_tokens,
        bins=bin_edges,
        alpha=HIST_ALPHA,
        label="STIR",
        color=colors["esm"],
    )

    if fit_normal and combined and isinstance(bin_edges, np.ndarray):
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        x_grid = np.linspace(bin_edges[0], bin_edges[-1], 200)
        for label, data in (("greedy", fit_greedy), ("esm", fit_esm)):
            if not data:
                continue
            display_label = DISPLAY_NAMES.get(label, label)
            series_mode = _select_fit_mode(fit_mode, data)
            modes = ["normal", "half"] if series_mode == "both" else [series_mode]
            for dist in modes:
                scale = len(data) * bin_width
                if dist == "half":
                    loc = max(0.0, float(min(data)))
                    curve = _half_normal_curve(x_grid, data, scale, loc=loc)
                    curve_label = (
                        f"{display_label} Half-normal Fit"
                        if len(modes) > 1
                        else f"{display_label} Fit"
                    )
                elif dist == "gmm":
                    curve = _gaussian_mixture_curve(x_grid, data, scale, n_components=gmm_components)
                    curve_label = f"{display_label} GMM({gmm_components}) Fit"
                else:
                    curve = _normal_curve(x_grid, data, scale)
                    curve_label = (
                        f"{display_label} Normal Fit"
                        if len(modes) > 1
                        else f"{display_label} Fit"
                    )
                if np.any(curve):
                    plt.plot(
                        x_grid,
                        curve,
                        color=colors[label],
                        linewidth=1.6,
                        alpha=FIT_LINE_ALPHA,
                        label=curve_label,
                    )

    ax = plt.gca()
    ax.set_xlabel("Generated Tokens", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Count", fontsize=LABEL_FONTSIZE)
    # plt.title(f"{model} ({tier}) token usage")
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.25)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=1,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        handleheight=0.9,
        handlelength=1.0,
        handletextpad=0.1,
        columnspacing=0.4,
        markerscale=1.2,
    )
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()

    fname = FIG_DIR / f"tokens_hist_amc_{_slugify(model)}_{tier}.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AMC token histograms for ESM vs greedy.")
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins (smaller width -> larger bins count).")
    parser.add_argument(
        "--fit-normal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay normal fit curves.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["auto", "normal", "half", "both", "gmm"],
        default="normal",
        help="Which distribution to overlay when fitting (auto picks half-normal for right-skewed non-negative data).",
    )
    parser.add_argument(
        "--gmm-components",
        type=int,
        default=2,
        help="Number of components for GMM fit when --fit-mode gmm.",
    )
    args = parser.parse_args()

    stats = read_token_stats()
    saved: list[Path] = []

    for (model, tier), row in sorted(stats.items()):
        esm_tokens = [int(x) for x in row.get("tokens", [])]
        greedy_tokens, t_max = read_greedy_tokens(model)
        saved.append(
            plot_histogram(
                model=model,
                tier=tier,
                esm_tokens=esm_tokens,
                greedy_tokens=greedy_tokens,
                t_max=t_max,
                bins=max(5, int(args.bins)),
                fit_normal=bool(args.fit_normal),
                fit_mode=str(args.fit_mode),
                gmm_components=int(args.gmm_components),
            )
        )

    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
