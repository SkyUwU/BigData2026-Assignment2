"""Read the relational tables available at a historical prediction time."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_policy import feature_columns


# Each table has its own row-time column. These follow data/README.md.
# These are structural columns needed for node identity, joins, time filtering,
# and target selection. Feature columns are added only by feature_policy.py.
STRUCTURAL_COLUMNS: dict[str, tuple[str, ...]] = {
    # is_cohort identifies which users have complete activity / are prediction targets.
    "Users": ("user_id", "github_user_id", "is_cohort", "first_seen_at"),
    "Repositories": ("repo_id", "github_repo_id", "first_seen_at"),
    "PullRequests": ("pr_id", "repo_id", "author_id", "first_seen_at", "created_at"),
    "Issues": (
        "issue_id",
        "repo_id",
        "author_id",
        "pull_request_id",
        "first_seen_at",
        "created_at",
        "assignee_id",
    ),
    "IssueComments": (
        "comment_id",
        "issue_id",
        "repo_id",
        "author_id",
        "created_at",
    ),
    "PRReviews": (
        "review_id",
        "pr_id",
        "repo_id",
        "reviewer_id",
        "first_seen_at",
        "submitted_at",
    ),
    "PRReviewComments": (
        "comment_id",
        "pr_id",
        "repo_id",
        "author_id",
        "pull_request_review_id",
        "in_reply_to_id",
        "created_at",
    ),
    "Pushes": ("push_id", "user_id", "repo_id", "created_at"),
}

ROW_TIME_COLUMN = {
    "Users": "first_seen_at",
    "Repositories": "first_seen_at",
    "PullRequests": "first_seen_at",
    "Issues": "first_seen_at",
    "IssueComments": "created_at",
    "PRReviews": "first_seen_at",
    "PRReviewComments": "created_at",
    "Pushes": "created_at",
}


def as_utc_naive_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Convert an input time to UTC, matching the dataset's timezone-naive times."""
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def load_snapshot(data_dir: Path, cutoff: str | pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Load only relation/key columns for rows available strictly before cutoff.

    This deliberately does not calculate activity-count features. It keeps the
    keys needed to construct a heterogeneous graph and drops rows whose row-time
    says they were not yet present at the prediction time.
    """
    cutoff_utc = as_utc_naive_timestamp(cutoff)
    tables: dict[str, pd.DataFrame] = {}

    for table_name, columns in STRUCTURAL_COLUMNS.items():
        table_path = data_dir / f"{table_name}.parquet"
        if not table_path.is_file():
            raise FileNotFoundError(f"Missing table: {table_path}")

        selected_columns = tuple(dict.fromkeys((*columns, *feature_columns(table_name))))
        table = pd.read_parquet(table_path, columns=list(selected_columns))
        row_time_column = ROW_TIME_COLUMN[table_name]
        available = table[row_time_column].notna() & (table[row_time_column] < cutoff_utc)
        # Keep the source row index after filtering; explicit PK columns remain
        # the authoritative values for joins and graph construction.
        tables[table_name] = table.loc[available]

    return tables
