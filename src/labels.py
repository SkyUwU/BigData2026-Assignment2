"""Build six binary activity labels for a future time window."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from data import as_utc_naive_timestamp


LABEL_COLUMNS = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)

# Activity time is the time the action happened, not necessarily the table's
# row-availability time. This mapping follows evaluation/README.md and SCHEMA.md.
ACTIVITY_SOURCES = {
    "open_pr": ("PullRequests", "author_id", "created_at"),
    "open_issue": ("Issues", "author_id", "created_at"),
    "issue_comment": ("IssueComments", "author_id", "created_at"),
    "pr_review": ("PRReviews", "reviewer_id", "submitted_at"),
    "pr_review_comment": ("PRReviewComments", "author_id", "created_at"),
    "push": ("Pushes", "user_id", "created_at"),
}


def build_activity_labels(
    data_dir: Path,
    target_user_ids: Iterable[int],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return one six-column 0/1 label row per GitHub user ID.

    The label window is half-open: start <= activity_time < end. These future
    events are used only as training answers; they must never enter the input
    snapshot at the corresponding prediction time.
    """
    start_utc = as_utc_naive_timestamp(start)
    end_utc = as_utc_naive_timestamp(end)
    if start_utc >= end_utc:
        raise ValueError("start must be earlier than end")

    target_ids = list(dict.fromkeys(int(user_id) for user_id in target_user_ids))
    labels = pd.DataFrame(
        0,
        index=pd.Index(target_ids, name="UserID"),
        columns=LABEL_COLUMNS,
        dtype="int8",
    )

    users = pd.read_parquet(
        data_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"],
    )
    cohort_users = users.loc[users["is_cohort"].fillna(False)]
    internal_to_github_id = dict(
        zip(cohort_users["user_id"].astype("int64"), cohort_users["github_user_id"].astype("int64"))
    )

    for label_name, (table_name, actor_column, time_column) in ACTIVITY_SOURCES.items():
        columns = [actor_column, time_column]
        if label_name == "open_issue":
            columns.append("is_pull_request")

        events = pd.read_parquet(data_dir / f"{table_name}.parquet", columns=columns)
        if label_name == "open_issue":
            events = events.loc[events["is_pull_request"].eq(False)]

        # Stored timestamps are UTC without a timezone; utc=True also safely
        # handles timezone-aware Parquet values if the format changes later.
        event_times = pd.to_datetime(events[time_column], errors="coerce", utc=True)
        event_times = event_times.dt.tz_localize(None)
        in_window = event_times.ge(start_utc) & event_times.lt(end_utc)
        actors = events.loc[in_window, actor_column].dropna().map(internal_to_github_id)
        active_ids = pd.Index(actors.dropna().astype("int64").unique())
        matching_targets = labels.index.intersection(active_ids)
        labels.loc[matching_targets, label_name] = 1

    return labels.reset_index()
