"""Compare two CPU classifiers on historical issue-comment labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from activity_features import UserActivityHistory, user_activity_feature_frame
from data import ROW_TIME_COLUMN, as_utc_naive_timestamp
from labels import ACTIVITY_SOURCES
from training_table import build_temporal_splits


def repo_counts(events, cutoff, actor_ids):
    visible = events.loc[
        events["row_time"].lt(cutoff) & events["activity_time"].lt(cutoff)
        & events["activity_time"].ge(cutoff - pd.Timedelta(days=30))
    ]
    visits = visible.groupby(["actor", "repo"]).size().rename("visits").reset_index()
    favorite = (
        visits.sort_values(["visits", "repo"], ascending=[False, True])
        .drop_duplicates("actor").set_index("actor")["repo"].reindex(actor_ids)
    )
    issues = visible.loc[
        visible["open_issue"] & visible["activity_time"].ge(cutoff - pd.Timedelta(days=7))
    ]
    totals = favorite.map(issues.groupby("repo").size()).fillna(0)
    own = issues.groupby(["actor", "repo"]).size().reindex(
        pd.MultiIndex.from_arrays([actor_ids, favorite]), fill_value=0
    )
    return (totals.to_numpy() - own.to_numpy()) * np.where(favorite.notna(), 1.0, np.nan)


def inputs(data_dir, rows):
    tables, events = {}, []
    for label, (table_name, actor, activity_time) in ACTIVITY_SOURCES.items():
        row_time = ROW_TIME_COLUMN[table_name]
        columns = list(dict.fromkeys([actor, activity_time, row_time, "repo_id"]))
        if label == "open_issue":
            columns.append("is_pull_request")
        table = pd.read_parquet(data_dir / f"{table_name}.parquet", columns=columns)
        tables[table_name] = table
        if label == "open_issue":
            table = table.loc[table["is_pull_request"].eq(False)]
        events.append(pd.DataFrame({
            "actor": table[actor],
            "repo": table["repo_id"],
            "row_time": pd.to_datetime(table[row_time], utc=True).dt.tz_localize(None),
            "activity_time": pd.to_datetime(table[activity_time], utc=True).dt.tz_localize(None),
            "open_issue": label == "open_issue",
        }).dropna())
    users = pd.read_parquet(data_dir / "Users.parquet", columns=["github_user_id", "user_id"])
    ids = rows["UserID"].map(users.set_index("github_user_id")["user_id"])
    history = user_activity_feature_frame(
        UserActivityHistory(tables), ids.tolist(), rows["Time"].tolist()
    )
    events = pd.concat(events, ignore_index=True)
    counts = np.full(len(rows), np.nan)
    for cutoff, positions in rows.groupby("Time").indices.items():
        counts[positions] = repo_counts(events, cutoff, ids.iloc[positions].to_numpy())
    return history.to_numpy(), counts


def predict(train, labels, validation):
    """Fit a small L2-regularized logistic model with Newton updates."""
    train, validation = np.log1p(train), np.log1p(validation)
    means = np.nan_to_num(np.nanmean(train, axis=0))
    scales = np.nan_to_num(np.nanstd(train, axis=0), nan=1)
    scales[scales < 1e-8] = 1

    def design(values):
        return np.column_stack([
            np.ones(len(values)),
            np.nan_to_num((values - means) / scales),
            np.isnan(values),
        ])

    x = design(train)
    coefficients = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1]) * 0.001
    penalty[0, 0] = 0

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

    for _ in range(25):
        p = sigmoid(x @ coefficients)
        gradient = x.T @ (p - labels) / len(x) + penalty @ coefficients
        hessian = (x.T * (p * (1 - p))) @ x / len(x) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.norm(step) < 1e-7:
            break
    return sigmoid(design(validation) @ coefficients)


def metrics(labels, scores, threshold):
    predicted = scores >= threshold
    tp = int(labels[predicted].sum())
    return {
        "f1": 2 * tp / max(1, int(labels.sum() + predicted.sum())),
        "precision": tp / max(1, int(predicted.sum())),
        "recall": tp / max(1, int(labels.sum())),
    }


def average_precision(labels, scores):
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    tp = labels.cumsum()[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / max(1, tp[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-through", required=True)
    parser.add_argument("--train-start", default="2026-04-28T16:00:00Z")
    parser.add_argument("--validation-start", default="2026-10-06T16:00:00Z")
    parser.add_argument("--validation-end", default="2026-10-27T16:00:00Z")
    args = parser.parse_args()
    start, split, end = map(as_utc_naive_timestamp, (
        args.train_start, args.validation_start, args.validation_end
    ))
    cutoffs = pd.date_range(start, end, freq="7D", inclusive="left")
    train, validation, excluded = build_temporal_splits(
        args.data_dir, cutoffs, data_through=args.data_through,
        validation_start=split, validation_end=end,
    )
    weeks = sorted(validation["Time"].unique())
    if train.empty or len(weeks) < 2 or not excluded.empty:
        raise ValueError("Expected a nonempty training split and at least two complete validation weeks")
    rows = pd.concat([train, validation], ignore_index=True)
    history, repo = inputs(args.data_dir, rows)
    target = rows["issue_comment"].to_numpy()
    n = len(train)
    if len(np.unique(target[:n])) < 2:
        raise ValueError("Training needs both issue-comment classes")
    report = {"train_rows": n, "validation_rows": len(validation), "weeks": {}}
    for name, features in [
        ("history", history), ("history_plus_repo", np.column_stack([history, repo])),
    ]:
        scores = predict(features[:n], target[:n], features[n:])
        for week in weeks[1:]:
            earlier = validation["Time"].lt(week).to_numpy()
            current = validation["Time"].eq(week).to_numpy()
            truth = target[n:]
            threshold = max(np.arange(0.01, 1.0, 0.01), key=lambda t: metrics(
                truth[earlier], scores[earlier], t
            )["f1"])
            report["weeks"].setdefault(str(week), {})[name] = {
                "ap": average_precision(truth[current], scores[current]),
                "threshold": float(threshold),
                **metrics(truth[current], scores[current], threshold),
                "repo_coverage": float(np.isfinite(repo[n:][current]).mean()),
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
