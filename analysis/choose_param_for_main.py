"""Select two parameter tiers (high / med) per model-dataset pair.

Inputs:
- param.csv: sweep results for offline_candidate_layers & online_k_scale
- baseline.csv: greedy baseline containing tokens_used_mean

Outputs:
- chosen_params_for_main.csv: two rows per (models, dataset) with tier=high|med
"""

import argparse
from pathlib import Path

import pandas as pd


BASELINE_ACC_COL = "Greedy-CoT"
ACC_TOL = 0.001  # 0.1% tolerance for "about the same" accuracy
MIN_MED_SAVING = 0.10  # 10% token saving threshold vs baseline
NUMERIC_COLUMNS = [
    "acc",
    "tokens_used_mean",
    "probe_tokens_used_mean",
    "offline_candidate_layers",
    "online_k_scale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick high/med params for main runs.")
    repo_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--param-path",
        type=Path,
        default=repo_dir / "param.csv",
        help="CSV with sweep results (default: analysis/param.csv).",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=repo_dir / "baseline.csv",
        help="CSV with baseline tokens_used_mean (default: analysis/baseline.csv).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=repo_dir / "chosen_params_for_main.csv",
        help="Where to write the selected params.",
    )
    return parser.parse_args()


def load_param_table(param_path: Path, baseline_path: Path) -> pd.DataFrame:
    param = pd.read_csv(param_path)
    for col in NUMERIC_COLUMNS:
        if col in param.columns:
            param[col] = pd.to_numeric(param[col], errors="coerce")

    baseline = pd.read_csv(baseline_path)[
        ["models", "dataset", "tokens_used_mean", BASELINE_ACC_COL]
    ]
    baseline = baseline.rename(
        columns={
            "tokens_used_mean": "baseline_tokens_used_mean",
            BASELINE_ACC_COL: "baseline_acc",
        }
    )
    baseline["baseline_acc"] = pd.to_numeric(baseline["baseline_acc"], errors="coerce")

    merged = param.merge(baseline, on=["models", "dataset"], how="left")
    if merged["baseline_tokens_used_mean"].isna().any():
        missing = (
            merged[merged["baseline_tokens_used_mean"].isna()][["models", "dataset"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"Missing baseline tokens for: {missing}")

    if merged["baseline_acc"].isna().any():
        missing = (
            merged[merged["baseline_acc"].isna()][["models", "dataset"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"Missing baseline acc for: {missing}")

    merged["token_ratio_vs_baseline"] = (
        merged["tokens_used_mean"] / merged["baseline_tokens_used_mean"]
    )
    merged["token_saving_pct"] = 1.0 - merged["token_ratio_vs_baseline"]
    return merged


def select_high_row(group: pd.DataFrame) -> pd.Series:
    """Highest acc; tie-break by lower token ratio."""
    high_acc = group["acc"].max()
    high_rows = group[group["acc"] == high_acc]
    return high_rows.sort_values("token_ratio_vs_baseline").iloc[0]


def select_med_row(
    group: pd.DataFrame,
    high_row: pd.Series,
) -> pd.Series:
    """
    Med: choose minimum tokens; if saving vs baseline <10%, instead choose
    the minimum-token row whose acc is within ACC_TOL of baseline_acc (if any).
    """
    med_row = group.sort_values(
        ["token_ratio_vs_baseline", "acc"], ascending=[True, False]
    ).iloc[0]

    med_saving = 1.0 - float(med_row["token_ratio_vs_baseline"])
    if med_saving < MIN_MED_SAVING:
        near_baseline = group[group["acc"] >= group["baseline_acc"].iloc[0] - ACC_TOL]
        if not near_baseline.empty:
            return near_baseline.sort_values(
                ["token_ratio_vs_baseline", "acc"], ascending=[True, False]
            ).iloc[0]

    return med_row


def to_selection_row(row: pd.Series, tier: str, high_acc: float) -> dict:
    return {
        "models": row["models"],
        "dataset": row["dataset"],
        "tier": tier,
        "offline_candidate_layers": row["offline_candidate_layers"],
        "online_k_scale": row["online_k_scale"],
        "acc": row["acc"],
        "baseline_acc": row["baseline_acc"],
        "acc_increase_from_baseline": row["acc"] - row["baseline_acc"],
        "acc_gap_vs_high": row["acc"] - high_acc,
        "tokens_used_mean": row["tokens_used_mean"],
        "baseline_tokens_used_mean": row["baseline_tokens_used_mean"],
        "token_ratio_vs_baseline": row["token_ratio_vs_baseline"],
        "token_saving_pct": row["token_saving_pct"],
        "token_drop_rate_pct": row["token_saving_pct"] * 100.0,
        "probe_tokens_used_mean": row.get("probe_tokens_used_mean"),
    }


def main() -> None:
    args = parse_args()
    table = load_param_table(args.param_path, args.baseline_path)

    selections = []
    for (model, dataset), group in table.groupby(["models", "dataset"]):
        high_row = select_high_row(group)
        med_row = select_med_row(
            group=group,
            high_row=high_row,
        )
        selections.append(to_selection_row(high_row, "high", high_row["acc"]))
        selections.append(to_selection_row(med_row, "med", high_row["acc"]))

    output = pd.DataFrame(selections).sort_values(
        ["models", "dataset", "tier"]
    ).reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)

    pair_count = len(output) // 2
    print(f"Saved {len(output)} rows ({pair_count} model/dataset pairs) to {args.output_path}")


if __name__ == "__main__":
    main()
