import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import pandas as pd
from pathlib import Path
from matplotlib.ticker import FixedLocator

# config = {
#     "font.family": "serif",
#     "font.size": 11,
#     "mathtext.fontset": "stix",
#     "font.serif": ["Times New Roman"],
# }
mpl.rc("pdf", fonttype=42)
FONTSIZE = 20
ALLWIDTH = 1.5
Marker = ["o", "v", "8", "s", "p", "^", "<", ">", "*", "h", "H", "D", "d", "P", "X"]
HATCH = ["+", "x", "/", "o", "|", "\\", "-", "O", ".", "*"]
COLORS = sns.color_palette("Paired")
# rcParams.update(config)

        # "acc": (80, 95),
        # "acc_ticks": [80, 85, 90, 95],
# Per-figure y-lims for easy tweaking
YLIMS = {
    "math500_param_by_k": {
        "tokens": (2100, 3600),
        "tokens_ticks": [2100, 2600, 3100, 3600],
        "acc": (77, 92),
        "acc_ticks": [77, 82, 87, 92],
    },
    "math500_param_by_layer": {
        "tokens": (2000, 3500),
        "tokens_ticks": [2000, 2500, 3000, 3500],
        "acc": (80, 95),
        "acc_ticks": [80, 85, 90, 95],
    },
    "amc23_param_by_k": {
        "tokens": (4600, 5800),
        "tokens_ticks": [4600, 5000, 5400, 5800],
        "acc": (80, 92),
        "acc_ticks": [80, 84, 88, 92],
    },
    "amc23_param_by_layer": {
        "tokens": (4000, 5800),
        "tokens_ticks": [4000, 4600, 5200, 5800],
        "acc": (80, 95),
        "acc_ticks": [80, 85, 90, 95],
    },
}


def plot_bar_line(
    df: pd.DataFrame,
    dataset: str,
    varying_col: str,
    fixed_col: str,
    fixed_val: float,
    xlabel: str,
    outfile: Path,
    color_idx: int,
    hatch_idx: int,
    line_color_idx: int,
    tokens_ylim=None,
    acc_ylim=None,
    tokens_ticks=None,
    acc_ticks=None,
):
    subset = df[(df["dataset"] == dataset) & (df[fixed_col] == fixed_val)].copy()
    if subset.empty:
        return

    x_vals = sorted(subset[varying_col].unique())
    x = np.arange(len(x_vals))
    tokens = [subset[subset[varying_col] == v]["tokens_used_mean"].mean() for v in x_vals]
    acc = [subset[subset[varying_col] == v]["acc"].mean() * 100 for v in x_vals]

    fig, ax1 = plt.subplots(figsize=(6.65, 4.65))

    bar = ax1.bar(
        x,
        tokens,
        width=0.45,
        color="white",
        ec=COLORS[color_idx],
        hatch=HATCH[hatch_idx] * 2,
        linewidth=ALLWIDTH,
        label="Avg. Tokens",
    )
    ax1.set_ylabel("Tokens", fontsize=FONTSIZE)
    ax1.set_xlabel(xlabel, fontsize=FONTSIZE-2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v:.1f}" if isinstance(v, (int, float)) else f"{v}" for v in x_vals], rotation=0)
    ax1.tick_params(labelsize=FONTSIZE-2)
    ax1.grid(linestyle=":", axis="y")

    if tokens_ylim:
        ax1.set_ylim(tokens_ylim)
    if tokens_ticks:
        ax1.yaxis.set_major_locator(FixedLocator(tokens_ticks))
    else:
        y_min, y_max = min(tokens), max(tokens)
        ax1.set_ylim(y_min * 0.92 if y_min > 0 else y_min - 50, y_max * 1.08 + 1)

    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        acc,
        color=COLORS[line_color_idx],
        marker="o",
        linestyle="-",
        linewidth=1.33,
        markersize=5,
        markeredgewidth=1.33,
        label="Accuracy (%)",
    )
    ax2.set_ylabel("Accuracy (%)", fontsize=FONTSIZE)
    ax2.tick_params(labelsize=FONTSIZE-2)
    if acc_ylim:
        ax2.set_ylim(acc_ylim)
    if acc_ticks:
        ax2.yaxis.set_major_locator(FixedLocator(acc_ticks))
    else:
        ax2.set_ylim(min(acc) - 2, max(acc) + 2)

    handles = [bar, line[0]]
    labels = ["Avg. Tokens", "Accuracy"]
    fig.legend(
        handles,
        labels,
        fontsize=FONTSIZE,
        loc="upper left",
        ncol=2,
        handleheight=0.7,
        handlelength=1.2,
        handletextpad=0.2,
        columnspacing=1,
        frameon=True,
        bbox_to_anchor=(0.2, 0.96),
    )
    plt.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(outfile) as pp:
        plt.savefig(pp, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv("plot_param.csv", encoding="utf-8-sig")

    color_map = {
        "math500": (1, 1, 3),  # bar edge idx, hatch idx, line color idx
        "amc23": (9, 5, 11),
    }

    out_dir = Path("param_figs")

    for dataset in ["math500", "amc23"]:
        bar_idx, hatch_idx, line_idx = color_map.get(dataset, (1, 1, 3))

        cfg = YLIMS.get(f"{dataset}_param_by_k", {})
        plot_bar_line(
            df,
            dataset=dataset,
            varying_col="online_k_scale",
            fixed_col="offline_candidate_layers",
            fixed_val=0.8,
            xlabel="Strength",
            outfile=out_dir / f"{dataset}_param_by_k.pdf",
            color_idx=bar_idx,
            hatch_idx=hatch_idx,
            line_color_idx=line_idx,
            tokens_ylim=cfg.get("tokens"),
            acc_ylim=cfg.get("acc"),
            tokens_ticks=cfg.get("tokens_ticks"),
            acc_ticks=cfg.get("acc_ticks"),
        )

        cfg = YLIMS.get(f"{dataset}_param_by_layer", {})
        plot_bar_line(
            df,
            dataset=dataset,
            varying_col="offline_candidate_layers",
            fixed_col="online_k_scale",
            fixed_val=1.0,
            xlabel="Layer",
            outfile=out_dir / f"{dataset}_param_by_layer.pdf",
            color_idx=bar_idx,
            hatch_idx=hatch_idx,
            line_color_idx=line_idx,
            tokens_ylim=cfg.get("tokens"),
            acc_ylim=cfg.get("acc"),
            tokens_ticks=cfg.get("tokens_ticks"),
            acc_ticks=cfg.get("acc_ticks"),
        )


if __name__ == "__main__":
    main()
