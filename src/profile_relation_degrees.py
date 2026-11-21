"""Report temporal in/out-degree summaries for the assignment's FK relations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data import ROW_TIME_COLUMN, as_utc_naive_timestamp
from relations import FOREIGN_KEYS, PRIMARY_KEYS


def _summary(degrees: pd.Series) -> str:
    quantiles = degrees.quantile([0.50, 0.90, 0.95, 0.99]).round(1)
    return (
        f"nodes={len(degrees):,}, zero={degrees.eq(0).mean():.1%}, "
        f"p50/p90/p95/p99/max="
        f"{quantiles.iloc[0]:g}/{quantiles.iloc[1]:g}/"
        f"{quantiles.iloc[2]:g}/{quantiles.iloc[3]:g}/{degrees.max():g}"
    )


def profile_relation_degrees(
    data_dir: Path, cutoff: str | pd.Timestamp
) -> list[str]:
    """Summarize neighbors visible strictly before cutoff, without writing files."""
    cutoff_utc = as_utc_naive_timestamp(cutoff)
    required_columns = {
        table_name: {PRIMARY_KEYS[table_name], ROW_TIME_COLUMN[table_name]}
        for table_name in PRIMARY_KEYS
    }
    for source_table, foreign_key, _, _ in FOREIGN_KEYS:
        required_columns[source_table].add(foreign_key)
    required_columns["Users"].add("is_cohort")

    tables: dict[str, pd.DataFrame] = {}
    for table_name, columns in required_columns.items():
        path = data_dir / f"{table_name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing table: {path}")
        table = pd.read_parquet(path, columns=sorted(columns))
        row_time = ROW_TIME_COLUMN[table_name]
        times = pd.to_datetime(table[row_time], errors="coerce", utc=True).dt.tz_localize(None)
        available = times.notna() & times.lt(cutoff_utc)
        tables[table_name] = table.loc[available].copy()

    available_ids = {
        table_name: set(
            pd.to_numeric(table[PRIMARY_KEYS[table_name]], errors="coerce")
            .dropna()
            .astype("int64")
        )
        for table_name, table in tables.items()
    }

    lines = [
        f"Cutoff UTC: {cutoff_utc}",
        "Forward FK direction has at most one neighbor per source row.",
        "Reverse direction reports the number of source rows per target node.",
    ]
    cohort_user_ids = set(
        pd.to_numeric(
            tables["Users"].loc[
                tables["Users"]["is_cohort"].fillna(False), "user_id"
            ],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
    )

    for source_table, foreign_key, target_table, relation_name in FOREIGN_KEYS:
        source_rows = tables[source_table]
        target_rows = tables[target_table]
        source_pk = PRIMARY_KEYS[source_table]
        target_pk = PRIMARY_KEYS[target_table]

        foreign_key_values = pd.to_numeric(
            source_rows[foreign_key], errors="coerce"
        )
        source_has_edge = foreign_key_values.isin(available_ids[target_table])
        values = foreign_key_values.loc[source_has_edge].astype("int64")
        counts = values.value_counts()

        lines.append(
            f"{source_table} -[{relation_name}]-> {target_table}: "
            f"edges={len(values):,}, source_rows={len(source_rows):,}, "
            f"source_rows_with_edge={int(source_has_edge.sum()):,}, "
            f"source_fanout_max={1 if len(values) else 0}"
        )

        target_ids = pd.to_numeric(target_rows[target_pk], errors="coerce").dropna()
        target_ids = target_ids.astype("int64")
        reverse_degrees = counts.reindex(target_ids, fill_value=0).reset_index(drop=True)
        lines.append(
            f"  reverse {target_table} <- {source_table}: "
            f"{_summary(reverse_degrees)}"
        )
        if target_table == "Users":
            cohort_degrees = counts.reindex(sorted(cohort_user_ids), fill_value=0)
            lines.append(f"  cohort users only: {_summary(cohort_degrees)}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect per-relation neighbor counts before choosing fanout caps."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("C:/Users/User/Desktop/assignment2/data/db"),
    )
    parser.add_argument(
        "--cutoff",
        required=True,
        help="Prediction time in UTC; rows at or after it are excluded.",
    )
    args = parser.parse_args()

    for line in profile_relation_degrees(args.data_dir, args.cutoff):
        print(line)


if __name__ == "__main__":
    main()
