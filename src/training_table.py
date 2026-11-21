"""Build historical supervision rows for the assignment's temporal task."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from data import ROW_TIME_COLUMN, as_utc_naive_timestamp
from labels import ACTIVITY_SOURCES, LABEL_COLUMNS
from temporal_split import split_training_table


TRAINING_TABLE_COLUMNS = ("UserID", "Time", *LABEL_COLUMNS)


def _to_utc_naive(values: pd.Series) -> pd.Series:
    """Match the assignment's UTC timestamps, which are stored without a zone."""
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)


def _load_cohort_users(data_dir: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    users = pd.read_parquet(
        data_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort", "first_seen_at"],
    )
    users = users.loc[users["is_cohort"].fillna(False)].copy()
    users["first_seen_at"] = _to_utc_naive(users["first_seen_at"])
    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce")
    users["github_user_id"] = pd.to_numeric(users["github_user_id"], errors="coerce")
    users = users.dropna(subset=["user_id", "github_user_id"])
    internal_to_github = dict(
        zip(users["user_id"].astype("int64"), users["github_user_id"].astype("int64"))
    )
    return users, internal_to_github


def _load_activity_rows(
    data_dir: Path, internal_to_github: dict[int, int]
) -> dict[str, pd.DataFrame]:
    """Load each activity source once and map actors to stable GitHub user IDs."""
    activity_rows: dict[str, pd.DataFrame] = {}

    for label_name, (table_name, actor_column, activity_time_column) in ACTIVITY_SOURCES.items():
        row_time_column = ROW_TIME_COLUMN[table_name]
        selected_columns = list(
            dict.fromkeys((actor_column, activity_time_column, row_time_column))
        )
        if label_name == "open_issue":
            selected_columns.append("is_pull_request")

        table = pd.read_parquet(data_dir / f"{table_name}.parquet", columns=selected_columns)
        if label_name == "open_issue":
            # A GitHub PR also has an Issues row; it is not an "open issue" label.
            table = table.loc[table["is_pull_request"].eq(False)]

        actor_ids = pd.to_numeric(table[actor_column], errors="coerce")
        activity_rows[label_name] = pd.DataFrame(
            {
                "UserID": actor_ids.map(internal_to_github),
                "activity_time": _to_utc_naive(table[activity_time_column]),
                "row_time": _to_utc_naive(table[row_time_column]),
            }
        ).dropna(subset=["UserID", "activity_time", "row_time"])
        activity_rows[label_name]["UserID"] = activity_rows[label_name]["UserID"].astype(
            "int64"
        )

    return activity_rows


def validate_label_coverage(
    cutoff_times: Iterable[str | pd.Timestamp],
    *,
    data_through: str | pd.Timestamp,
    prediction_days: int,
    gap_days: int = 0,
) -> None:
    """Reject samples whose full future label window is not in this data release.

    ``data_through`` is the release's known coverage boundary, not the latest
    event timestamp in a table (a quiet period can have no events).
    """
    if prediction_days <= 0 or gap_days < 0:
        raise ValueError("prediction_days must be positive and gap_days non-negative")
    observed_through = as_utc_naive_timestamp(data_through)
    duration = pd.Timedelta(days=gap_days + prediction_days)
    for value in cutoff_times:
        cutoff = as_utc_naive_timestamp(value)
        if cutoff + duration > observed_through:
            raise ValueError(
                "Cannot create a complete label window: "
                f"{cutoff} + {gap_days + prediction_days} days ends after data_through "
                f"({observed_through}). Choose an earlier prediction time or "
                "a newer data release."
            )


def build_training_table(
    data_dir: Path,
    cutoff_times: Iterable[str | pd.Timestamp],
    *,
    data_through: str | pd.Timestamp,
    history_days: int = 90,
    prediction_days: int = 7,
    min_history_activities: int = 5,
    gap_days: int = 0,
    include_inactive_users: bool = False,
    labels_available_through: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build rows of (GitHub user, prediction time, next-window labels).

    At each cutoff, eligible targets are cohort users with at least
    ``min_history_activities`` activity records during the preceding
    ``history_days``. For eligibility, an activity must itself have been
    available in its source table before the cutoff. This mirrors the official
    test-user rule while preventing later-discovered rows from selecting a
    historical target.

    With include_inactive_users, all cohort users first seen before each cutoff
    are included, regardless of their recent activity.
    Future-window labels start ``gap_days`` after the cutoff and span
    ``prediction_days``. The cutoff controls the input graph;
    these labels are supervision only and must not be added to that graph.
    Each row is one supervised example, not a claim that the model was trained
    online at that row's time. Training examples may jointly update the model,
    provided their target windows end before the later validation period.
    """
    if history_days <= 0 or prediction_days <= 0 or min_history_activities <= 0:
        raise ValueError("history_days, prediction_days, and minimum activity count must be positive")

    cutoffs = [as_utc_naive_timestamp(value) for value in cutoff_times]
    if not cutoffs:
        raise ValueError("At least one historical cutoff is required")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoff_times must not contain duplicates")
    validate_label_coverage(
        cutoffs,
        data_through=data_through,
        prediction_days=prediction_days,
        gap_days=gap_days,
    )

    cohort_users, internal_to_github = _load_cohort_users(data_dir)
    activity_rows = _load_activity_rows(data_dir, internal_to_github)
    if not include_inactive_users:
        all_activities = pd.concat(activity_rows.values(), ignore_index=True)
    history_window = pd.Timedelta(days=history_days)
    gap = pd.Timedelta(days=gap_days)
    prediction_window = pd.Timedelta(days=prediction_days)
    available_through = (
        as_utc_naive_timestamp(labels_available_through)
        if labels_available_through is not None else None
    )
    training_rows: list[pd.DataFrame] = []

    for cutoff in sorted(cutoffs):
        cutoff_users = cohort_users.loc[cohort_users["first_seen_at"].lt(cutoff), "github_user_id"]
        cutoff_user_ids = pd.Index(cutoff_users.astype("int64").unique())
        eligible_ids = cutoff_user_ids
        if not include_inactive_users:

            in_history = (
                all_activities["row_time"].lt(cutoff)
                & all_activities["activity_time"].ge(cutoff - history_window)
                & all_activities["activity_time"].lt(cutoff)
                & all_activities["UserID"].isin(cutoff_user_ids)
            )
            history_counts = all_activities.loc[in_history, "UserID"].value_counts()
            eligible_ids = history_counts.loc[history_counts.ge(min_history_activities)].index

        labels = pd.DataFrame(
            0,
            index=pd.Index(eligible_ids, name="UserID"),
            columns=LABEL_COLUMNS,
            dtype="int8",
        )
        start = cutoff + gap
        end = start + prediction_window
        for label_name, events in activity_rows.items():
            in_target_window = events["activity_time"].ge(start) & events["activity_time"].lt(end)
            if available_through is not None:
                in_target_window &= events["row_time"].lt(available_through)
            active_ids = pd.Index(events.loc[in_target_window, "UserID"].unique())
            matching_ids = labels.index.intersection(active_ids)
            labels.loc[matching_ids, label_name] = 1

        sample = labels.reset_index()
        sample.insert(1, "Time", cutoff)
        training_rows.append(sample)

    result = pd.concat(training_rows, ignore_index=True)
    return result.loc[:, TRAINING_TABLE_COLUMNS].sort_values(
        ["Time", "UserID"], ignore_index=True
    )


def build_temporal_splits(
    data_dir: Path,
    cutoff_times: Iterable[str | pd.Timestamp],
    *,
    data_through: str | pd.Timestamp,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    history_days: int = 90,
    prediction_days: int = 7,
    min_history_activities: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and split labels in memory; no intermediate file is required.

    Returns ``(train, validation, excluded)``. The caller can pass the first
    two DataFrames directly to a graph training loop. The excluded rows are
    returned for inspection of boundary cases.
    """
    all_rows = build_training_table(
        data_dir,
        cutoff_times,
        data_through=data_through,
        history_days=history_days,
        prediction_days=prediction_days,
        min_history_activities=min_history_activities,
    )
    return split_training_table(
        all_rows,
        validation_start=validation_start,
        validation_end=validation_end,
        prediction_days=prediction_days,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a historical RDL training table from the assignment Parquet data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("C:/Users/User/Desktop/assignment2/data/db"),
        help="Folder containing the assignment Parquet files.",
    )
    parser.add_argument(
        "--cutoff",
        action="append",
        required=True,
        help="Historical prediction time in UTC; repeat for multiple times.",
    )
    parser.add_argument(
        "--data-through",
        required=True,
        help="UTC coverage boundary of this downloaded data release.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .parquet or .csv path. The caller chooses the historical cutoffs.",
    )
    args = parser.parse_args()

    training_table = build_training_table(
        args.data_dir,
        args.cutoff,
        data_through=args.data_through,
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        training_table.to_parquet(args.output, index=False)
    elif args.output.suffix.lower() == ".csv":
        training_table.to_csv(args.output, index=False)
    else:
        raise ValueError("--output must end in .parquet or .csv")

    print(f"Training rows: {len(training_table):,}")
    print(f"Prediction times: {training_table['Time'].nunique():,}")
    print(f"Unique users: {training_table['UserID'].nunique():,}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
