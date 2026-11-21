"""Point-in-time activity summaries for target-user model features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import ROW_TIME_COLUMN
from labels import ACTIVITY_SOURCES, LABEL_COLUMNS


ACTIVITY_WINDOWS_DAYS = (7, 30, 90)
WEEKLY_ACTIVITY_FEATURE_COLUMNS = tuple(f"{label}_count_7d" for label in LABEL_COLUMNS)
USER_ACTIVITY_FEATURE_COLUMNS = tuple(
    column
    for label in LABEL_COLUMNS
    for column in (
        *(f"{label}_count_{days}d" for days in ACTIVITY_WINDOWS_DAYS),
        f"{label}_days_since_last",
        f"{label}_has_history",
    )
)


@dataclass(frozen=True)
class _ActorEvents:
    row_times_ns: np.ndarray
    activity_times_ns: np.ndarray


class UserActivityHistory:
    """Query per-label event counts using only rows available before a cutoff.

    Actor IDs are the internal Users.parquet keys. Each source event is subject
    to both its row-availability time and its actual activity time, so a row
    first discovered after a cutoff cannot leak into that sample's features.
    """

    def __init__(self, tables: dict[str, pd.DataFrame], *, weekly_only: bool = False):
        self.weekly_only = weekly_only
        self.columns = (
            WEEKLY_ACTIVITY_FEATURE_COLUMNS if weekly_only else USER_ACTIVITY_FEATURE_COLUMNS
        )
        self.events: dict[str, dict[int, _ActorEvents]] = {}
        for label, (table_name, actor_column, activity_column) in ACTIVITY_SOURCES.items():
            table = tables[table_name]
            row_time_column = ROW_TIME_COLUMN[table_name]
            actor = pd.to_numeric(table[actor_column], errors="coerce")
            row_time = pd.to_datetime(table[row_time_column], errors="coerce", utc=True)
            activity_time = pd.to_datetime(table[activity_column], errors="coerce", utc=True)
            valid = actor.notna() & row_time.notna() & activity_time.notna()
            if label == "open_issue":
                valid &= table["is_pull_request"].eq(False)
            rows = pd.DataFrame({
                "actor": actor.loc[valid].astype("int64").to_numpy(),
                "row_time": row_time.loc[valid].dt.tz_localize(None).to_numpy(),
                "activity_time": activity_time.loc[valid].dt.tz_localize(None).to_numpy(),
            })
            by_actor: dict[int, _ActorEvents] = {}
            for actor_id, group in rows.groupby("actor", sort=False):
                group = group.sort_values("row_time", kind="stable")
                by_actor[int(actor_id)] = _ActorEvents(
                    row_times_ns=group["row_time"].astype("datetime64[ns]").astype("int64").to_numpy(),
                    activity_times_ns=group["activity_time"].astype("datetime64[ns]").astype("int64").to_numpy(),
                )
            self.events[label] = by_actor

    def features(self, internal_user_id: int, cutoff: pd.Timestamp) -> pd.Series:
        """Return cutoff-safe numeric activity features for one target user."""
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tz is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        cutoff_ns = int(cutoff.value)
        output: dict[str, float] = {}
        for label in LABEL_COLUMNS:
            events = self.events[label].get(int(internal_user_id))
            if events is None:
                latest_ns = None
                available_activity = np.empty(0, dtype="int64")
            else:
                n_available = int(np.searchsorted(events.row_times_ns, cutoff_ns, side="left"))
                available_activity = events.activity_times_ns[:n_available]
                available_activity = available_activity[available_activity < cutoff_ns]
                latest_ns = int(available_activity.max()) if available_activity.size else None
            for days in ((7,) if self.weekly_only else ACTIVITY_WINDOWS_DAYS):
                lower = cutoff_ns - days * 86_400_000_000_000
                output[f"{label}_count_{days}d"] = float(np.count_nonzero(available_activity >= lower))
            if self.weekly_only:
                continue
            output[f"{label}_has_history"] = float(latest_ns is not None)
            output[f"{label}_days_since_last"] = (
                float((cutoff_ns - latest_ns) / 86_400_000_000_000)
                if latest_ns is not None else float("nan")
            )
        return pd.Series(output, index=self.columns, dtype="float64")


def user_activity_feature_frame(
    history: UserActivityHistory,
    user_ids: list[int],
    cutoffs: list[pd.Timestamp],
) -> pd.DataFrame:
    """Build a frame for target training rows, preserving the provided order."""
    if len(user_ids) != len(cutoffs):
        raise ValueError("user_ids and cutoffs must have equal lengths")
    values = np.empty((len(user_ids), len(history.columns)), dtype="float64")
    for index, (user_id, cutoff) in enumerate(zip(user_ids, cutoffs)):
        values[index] = history.features(user_id, cutoff).to_numpy()
    return pd.DataFrame(values, columns=history.columns)
