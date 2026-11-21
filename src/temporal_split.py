"""Split historical prediction examples without crossing target windows."""

from __future__ import annotations

import pandas as pd

from data import as_utc_naive_timestamp
from labels import LABEL_COLUMNS


def split_training_table(
    table: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    prediction_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological train, validation, and excluded rows.

    A training label window must finish no later than ``validation_start``.
    A validation label window must fit completely inside
    ``[validation_start, validation_end)``. Rows whose label windows cross a
    boundary are excluded, so one future event cannot be a training answer and
    a validation answer at the same time.

    ``table`` can contain many users at each prediction time. The function
    keeps all six labels and does not change the user or label values.
    """
    required_columns = {"UserID", "Time", *LABEL_COLUMNS}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training table is missing required columns: {missing}")
    if prediction_days <= 0:
        raise ValueError("prediction_days must be positive")

    start = as_utc_naive_timestamp(validation_start)
    end = as_utc_naive_timestamp(validation_end)
    if start >= end:
        raise ValueError("validation_start must be earlier than validation_end")

    times = pd.to_datetime(table["Time"], errors="coerce", utc=True).dt.tz_localize(None)
    if times.isna().any():
        raise ValueError("Training table contains an invalid Time value")
    if table.duplicated(["UserID", "Time"]).any():
        raise ValueError("Training table contains duplicate (UserID, Time) rows")

    target_ends = times + pd.Timedelta(days=prediction_days)
    train_mask = target_ends.le(start)
    validation_mask = times.ge(start) & target_ends.le(end)
    excluded_mask = ~(train_mask | validation_mask)

    # Return independent copies so later edits to one split cannot affect another.
    return (
        table.loc[train_mask].copy(),
        table.loc[validation_mask].copy(),
        table.loc[excluded_mask].copy(),
    )
