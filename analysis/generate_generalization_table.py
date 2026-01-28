"""Generate heatmaps for cross-dataset generalization.

Output: analysis/generalization_heatmap.pdf
- Left: accuracy heatmap (numbers = absolute accuracy %, color = drop vs diagonal)
- Middle: token heatmap (numbers = absolute tokens, color = increase vs diagonal)
- Right: one shared colorbar (green = diagonal or better, yellow = worse)
"""

from pathlib import Path
from typing import Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "generation.csv"
FIGURE_PATH = REPO_ROOT / "generalization_heatmap.pdf"

def _soft_colormap(
    name: str, blend: float = 0.25, hi_clip: float = 0.7
) -> mcolors.LinearSegmentedColormap:
    """Return a softened colormap by blending toward white and clipping the bright tail."""
    base = colormaps[name]
    colors = base(np.linspace(0, hi_clip, 256))  # avoid very bright tail
    colors[:, :3] = colors[:, :3] * (1 - blend) + blend  # blend RGB toward white
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_soft", colors)


# Colormap: reverse YlGn so low penalty = green, high penalty = muted yellow-green.
HEATMAP_CMAP = _soft_colormap("YlGn_r", blend=0.25, hi_clip=0.7)

TARGET_ORDER = ["aime_2024", "aime25", "amc23", "arc-c", "math500", "openbookqa"]
DISPLAY_LABELS = {
    "aime_2024": "AIME 24",
    "aime25": "AIME 25",
    "amc23": "AMC 23",
    "arc-c": "ARC-C",
    "math500": "MATH500",
    "openbookqa": "OBQA",
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the two-level headers coming from the CSV export."""
    level0: list[str] = []
    previous = ""
    for raw in df.columns.get_level_values(0):
        name = raw.strip()
        if name.startswith("Unnamed"):
            name = previous
        if name == "列标签target dataset":
            name = "memory_dataset"
        level0.append(name)
        previous = name

    level1: list[str] = []
    for raw in df.columns.get_level_values(1):
        name = raw.replace("求和项:", "").strip()
        if name == "行标签memory dataset":
            name = ""
        level1.append(name)

    df.columns = pd.MultiIndex.from_arrays([level0, level1])
    df.columns = pd.MultiIndex.from_tuples(
        [(name.strip(), metric) for name, metric in df.columns]
    )
    return df


def load_generation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load accuracy and token matrices indexed by memory dataset."""
    raw = pd.read_csv(CSV_PATH, header=[0, 1], encoding="utf-8-sig")
    raw = _clean_columns(raw)
    raw = raw.set_index(("memory_dataset", ""))
    raw.index = raw.index.str.strip()

    acc = raw.xs("acc", level=1, axis=1)
    tokens = raw.xs("tokens_mean", level=1, axis=1)

    acc = acc.apply(pd.to_numeric, errors="coerce")
    tokens = tokens.apply(pd.to_numeric, errors="coerce")

    acc = acc.reindex(index=TARGET_ORDER, columns=TARGET_ORDER)
    tokens = tokens.reindex(index=TARGET_ORDER, columns=TARGET_ORDER)
    return acc, tokens


def compute_penalties(acc: pd.DataFrame, tokens: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute penalty matrices (>=0) relative to each target diagonal."""
    acc_diag = pd.Series({ds: acc.loc[ds, ds] for ds in TARGET_ORDER})
    tok_diag = pd.Series({ds: tokens.loc[ds, ds] for ds in TARGET_ORDER})

    # Accuracy penalty: relative drop vs diagonal (%). Improvements -> 0.
    acc_base = acc_diag.replace(0, np.nan)
    acc_penalty = ((acc_base - acc).div(acc_base)).clip(lower=0) * 100.0
    acc_penalty = acc_penalty.fillna(0)

    # Token penalty: percent increase over diagonal; reductions -> 0.
    tok_base = tok_diag.replace(0, np.nan)
    tok_penalty = ((tokens - tok_base) / tok_base).clip(lower=0) * 100.0
    tok_penalty = tok_penalty.fillna(0)

    return acc_penalty, tok_penalty


def normalize_for_colors(acc_penalty: pd.DataFrame, tok_penalty: pd.DataFrame):
    """Create a shared Normalize over both matrices for a single colorbar."""
    combined = np.concatenate([acc_penalty.values.ravel(), tok_penalty.values.ravel()])
    combined = combined[~np.isnan(combined)]
    if combined.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)
    vmax = np.percentile(combined, 98)
    if vmax <= 1e-6:
        vmax = 1.0
    return mcolors.Normalize(vmin=0, vmax=vmax)


def plot_heatmaps(
    acc: pd.DataFrame,
    tokens: pd.DataFrame,
    acc_penalty: pd.DataFrame,
    tok_penalty: pd.DataFrame,
    norm: mcolors.Normalize,
) -> None:
    fig = plt.figure(figsize=(12.5, 2), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.1])
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_tok = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    # ax_cbar.set_title("Worse", fontsize=9, pad=6)

    def draw(ax, values, penalty, title, formatter):
        im = ax.imshow(penalty.values, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")
        for i, row in enumerate(values.index):
            for j, col in enumerate(values.columns):
                ax.text(
                    j,
                    i,
                    formatter(values.iloc[i, j]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
        ax.set_xticks(range(len(values.columns)))
        ax.set_xticklabels([DISPLAY_LABELS.get(c, c) for c in values.columns], rotation=0, ha="center", fontsize=10)
        ax.set_yticks(range(len(values.index)))
        ax.set_yticklabels([DISPLAY_LABELS.get(r, r) for r in values.index], fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Source dataset", fontsize=11)
        ax.set_xlabel("Target dataset", fontsize=11)
        return im

    acc_img = draw(
        ax_acc,
        acc,
        acc_penalty,
        "Accuracy",
        lambda v: f"{v * 100:.1f}",
    )
    draw(
        ax_tok,
        tokens,
        tok_penalty,
        "Avg. Tokens",
        lambda v: f"{int(round(v)):,}",
    )

    cbar = fig.colorbar(acc_img, cax=ax_cbar, orientation="vertical")
    cbar.set_ticks([])
    cbar.set_label("", fontsize=9)
    ax_cbar.text(
        1.15,
        0.5,
        "Worse →",
        transform=ax_cbar.transAxes,
        ha="left",
        va="center",
        rotation=90,
        fontsize=10,
    )

    fig.subplots_adjust(wspace=0.5, right=0.985)
    cb_pos = ax_cbar.get_position()
    ax_cbar.set_position(
        [cb_pos.x0 - 0.08, cb_pos.y0, cb_pos.width*0.8, cb_pos.height]
    )
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    acc, tokens = load_generation()
    acc_penalty, tok_penalty = compute_penalties(acc, tokens)
    norm = normalize_for_colors(acc_penalty, tok_penalty)
    plot_heatmaps(acc, tokens, acc_penalty, tok_penalty, norm)
    print(f"Saved heatmap to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
