import json
from pathlib import Path

import pandas as pd


DEFAULT_TARGET_COLUMNS = ["req_0", "req_1", "req_2", "req_3"]
INPUT_CSV = Path("datasets/dataset15000_generated.csv")
OUTPUT_CSV = Path("datasets/dataset15000_balanced_joint.csv")
REPORT_JSON = Path("datasets/dataset15000_balanced_joint_report.json")
TARGET_COLUMNS = DEFAULT_TARGET_COLUMNS
BALANCE_DATASET = True
RANDOM_SEED = 42


def validate_target_columns(df: pd.DataFrame, target_columns: list[str]) -> None:
    missing = [column for column in target_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")


def joint_labels(df: pd.DataFrame, target_columns: list[str]) -> pd.Series:
    validate_target_columns(df, target_columns)
    return df[target_columns].astype(bool).astype(int).astype(str).agg("".join, axis=1)


def compute_joint_counts(df: pd.DataFrame, target_columns: list[str]) -> pd.Series:
    return joint_labels(df, target_columns).value_counts().sort_index()


def choose_target_count(counts: pd.Series) -> int:
    if counts.empty:
        raise ValueError("Cannot choose a target count from an empty distribution.")
    return max(1, int(counts.median()))


def list_all_combinations(width: int) -> list[str]:
    return [format(index, f"0{width}b") for index in range(2**width)]


def build_report(
    original_df: pd.DataFrame,
    target_columns: list[str],
    balanced_df: pd.DataFrame | None = None,
    target_count: int | None = None,
    seed: int | None = None,
) -> dict:
    before_counts = compute_joint_counts(original_df, target_columns)
    all_combinations = list_all_combinations(len(target_columns))
    report = {
        "target_columns": target_columns,
        "strategy": "joint_median_cap_undersampling",
        "seed": seed,
        "target_count": target_count,
        "rows_before": int(len(original_df)),
        "present_combinations_before": int(len(before_counts)),
        "missing_combinations_before": [
            combo for combo in all_combinations if combo not in before_counts.index
        ],
        "counts_before": {combo: int(before_counts.get(combo, 0)) for combo in all_combinations},
    }

    if balanced_df is not None:
        after_counts = compute_joint_counts(balanced_df, target_columns)
        report.update(
            {
                "rows_after": int(len(balanced_df)),
                "present_combinations_after": int(len(after_counts)),
                "missing_combinations_after": [
                    combo for combo in all_combinations if combo not in after_counts.index
                ],
                "counts_after": {combo: int(after_counts.get(combo, 0)) for combo in all_combinations},
            }
        )

    return report


def rebalance_joint_distribution(
    df: pd.DataFrame,
    target_columns: list[str],
    target_count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    labels = joint_labels(df, target_columns)
    selected_indices = []

    for combination, count in labels.value_counts().sort_index().items():
        group_indices = labels[labels == combination].index.to_series()
        if count > target_count:
            sampled_indices = group_indices.sample(n=target_count, random_state=seed, replace=False)
            selected_indices.extend(sampled_indices.tolist())
        else:
            selected_indices.extend(group_indices.tolist())

    balanced_df = df.loc[sorted(selected_indices)].reset_index(drop=True)
    metadata = {
        "target_count": int(target_count),
        "rows_before": int(len(df)),
        "rows_after": int(len(balanced_df)),
        "dropped_rows": int(len(df) - len(balanced_df)),
    }
    return balanced_df, metadata


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    validate_target_columns(df, TARGET_COLUMNS)

    balanced_df = None
    target_count = choose_target_count(compute_joint_counts(df, TARGET_COLUMNS))

    if BALANCE_DATASET:
        balanced_df, _ = rebalance_joint_distribution(
            df,
            target_columns=TARGET_COLUMNS,
            target_count=target_count,
            seed=RANDOM_SEED,
        )
        balanced_df.to_csv(OUTPUT_CSV, index=False)

    report = build_report(
        original_df=df,
        target_columns=TARGET_COLUMNS,
        balanced_df=balanced_df,
        target_count=target_count,
        seed=RANDOM_SEED,
    )

    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
