"""Materialize one sampled subgraph's allowlisted row features for a model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data import ROW_TIME_COLUMN, as_utc_naive_timestamp
from activity_features import USER_ACTIVITY_FEATURE_COLUMNS, UserActivityHistory
from feature_policy import feature_columns
from temporal_subgraph import EdgeType, TemporalRelationalGraph, UserSubgraph


TIME_DELTA_DAYS_COLUMN = "time_delta_days"


@dataclass(frozen=True)
class SubgraphModelInput:
    """Raw, type-separated features aligned with one sampled subgraph.

    Feature frames are in the same order as ``subgraph.node_indices``. Raw
    source columns follow ``feature_policy.py``; target-user activity summaries
    are derived from cutoff-visible source events. ``time_delta_days`` is
    derived from each row's availability time and this sample's cutoff.
    Categorical encoding and tensor conversion are handled later.
    """

    user_id: int
    cutoff: pd.Timestamp
    node_features: dict[str, pd.DataFrame]
    edge_index: dict[EdgeType, tuple[tuple[int, ...], tuple[int, ...]]]
    seed_node_index: int


def materialize_subgraph_input(
    graph: TemporalRelationalGraph,
    subgraph: UserSubgraph,
    activity_history: UserActivityHistory | None = None,
) -> SubgraphModelInput:
    """Select safe row attributes for the nodes in a time-filtered subgraph."""
    cutoff = as_utc_naive_timestamp(subgraph.cutoff)
    node_features: dict[str, pd.DataFrame] = {}

    for node_type, indices in subgraph.node_indices.items():
        if node_type not in graph.tables:
            raise ValueError(f"Subgraph contains unknown node type: {node_type}")
        table = graph.tables[node_type]
        if any(index < 0 or index >= len(table) for index in indices):
            raise ValueError(f"Subgraph contains an invalid {node_type} row index")

        columns = feature_columns(node_type)
        missing_columns = set(columns).difference(table.columns)
        if missing_columns:
            raise ValueError(
                f"{node_type} is missing allowlisted features: "
                + ", ".join(sorted(missing_columns))
            )

        selected_rows = table.iloc[list(indices)]
        if indices:
            row_times = pd.to_datetime(
                selected_rows[ROW_TIME_COLUMN[node_type]], errors="coerce", utc=True
            ).dt.tz_localize(None)
            visible = row_times.lt(cutoff)
            if node_type == "Users" and subgraph.user_id in graph.cohort_ids:
                if not visible.iloc[subgraph.seed_node_index]:
                    row_times = row_times.mask(~visible)
                    visible.iloc[subgraph.seed_node_index] = True
            if not visible.all():
                raise ValueError(
                    f"Subgraph contains {node_type} rows not available before cutoff"
                )

        # Reset the row index so feature row 0 aligns with local node index 0.
        features = selected_rows.loc[:, list(columns)].reset_index(drop=True)
        # Give the model the elapsed time from when each row became available
        # to this sample's prediction time. This is sample-specific, unlike a
        # single timestamp feature computed once for the whole graph.
        features[TIME_DELTA_DAYS_COLUMN] = (
            (cutoff - row_times).dt.total_seconds().to_numpy() / (24 * 60 * 60)
            if indices
            else pd.Series(dtype="float64")
        )
        if node_type == "Users":
            # These cutoff-specific features are defined for the prediction
            # target only. Neighbor users retain missing values, so the model
            # cannot mistake an uncomputed neighbor summary for zero activity.
            columns = activity_history.columns if activity_history is not None else USER_ACTIVITY_FEATURE_COLUMNS
            for column in columns:
                features[column] = float("nan")
            if activity_history is not None:
                seed_global_index = indices[subgraph.seed_node_index]
                internal_user_id = int(table.iloc[seed_global_index]["user_id"])
                seed_features = (
                    activity_history.features(internal_user_id, cutoff)
                    if pd.notna(row_times.iloc[subgraph.seed_node_index])
                    else dict.fromkeys(columns, 0.0)
                )
                for column, value in seed_features.items():
                    features.at[subgraph.seed_node_index, column] = value
        node_features[node_type] = features

    if not node_features.get("Users", pd.DataFrame()).shape[0]:
        raise ValueError("Subgraph input must contain its target user")
    if not 0 <= subgraph.seed_node_index < len(node_features["Users"]):
        raise ValueError("Seed node index is outside the Users feature rows")

    return SubgraphModelInput(
        user_id=int(subgraph.user_id),
        cutoff=cutoff,
        node_features=node_features,
        edge_index=subgraph.edge_index,
        seed_node_index=subgraph.seed_node_index,
    )
