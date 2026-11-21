"""Profile two-hop subgraph sizes without loading text/features or writing data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd

from data import ROW_TIME_COLUMN, STRUCTURAL_COLUMNS, as_utc_naive_timestamp
from relations import FOREIGN_KEYS, PRIMARY_KEYS
from sampling_config import default_fanout_by_relation
from temporal_subgraph import EdgeType
from training_table import _load_activity_rows, _load_cohort_users


@dataclass(frozen=True)
class CSRAdjacency:
    offsets: np.ndarray
    neighbors: np.ndarray

    @classmethod
    def from_edges(
        cls, sources: np.ndarray, targets: np.ndarray, num_sources: int
    ) -> "CSRAdjacency":
        counts = np.bincount(sources, minlength=num_sources)
        offsets = np.empty(num_sources + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        order = np.argsort(sources, kind="stable")
        return cls(offsets=offsets, neighbors=targets[order])

    def for_node(self, node_index: int) -> np.ndarray:
        return self.neighbors[self.offsets[node_index] : self.offsets[node_index + 1]]


class SubgraphSizeProfiler:
    """Build compact structural indexes, then reuse per-sample time filtering."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables
        self.node_times: dict[str, np.ndarray] = {}
        self.node_ids: dict[str, pd.Index] = {}
        for table_name, table in tables.items():
            pk = PRIMARY_KEYS[table_name]
            if table[pk].duplicated().any():
                raise ValueError(f"{table_name}.{pk} contains duplicate IDs")
            self.node_ids[table_name] = pd.Index(table[pk])
            self.node_times[table_name] = pd.to_datetime(
                table[ROW_TIME_COLUMN[table_name]], errors="coerce", utc=True
            ).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")

        self.adjacencies: dict[EdgeType, CSRAdjacency] = {}
        for source_table, foreign_key, target_table, relation_name in FOREIGN_KEYS:
            source_rows = tables[source_table]
            target_positions = self.node_ids[target_table].get_indexer(
                source_rows[foreign_key]
            )
            valid = target_positions >= 0
            source_positions = np.flatnonzero(valid).astype(np.int64, copy=False)
            target_positions = target_positions[valid].astype(np.int64, copy=False)
            self.adjacencies[(source_table, relation_name, target_table)] = (
                CSRAdjacency.from_edges(
                    source_positions, target_positions, len(source_rows)
                )
            )
            self.adjacencies[(target_table, f"rev_{relation_name}", source_table)] = (
                CSRAdjacency.from_edges(
                    target_positions,
                    source_positions,
                    len(tables[target_table]),
                )
            )

    @classmethod
    def from_data_dir(
        cls, data_dir: Path, *, data_through: str | pd.Timestamp
    ) -> "SubgraphSizeProfiler":
        """Read structural columns available by an explicit release boundary."""
        through = as_utc_naive_timestamp(data_through)
        required = {
            name: set(columns)
            for name, columns in STRUCTURAL_COLUMNS.items()
        }
        # Keep this diagnostic lean: keys/times only, plus stable cohort IDs.
        for name in required:
            required[name] = {PRIMARY_KEYS[name], ROW_TIME_COLUMN[name]}
        for source, foreign_key, _, _ in FOREIGN_KEYS:
            required[source].add(foreign_key)
        required["Users"].update({"github_user_id", "is_cohort"})

        tables: dict[str, pd.DataFrame] = {}
        for table_name, columns in required.items():
            path = data_dir / f"{table_name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing table: {path}")
            table = pd.read_parquet(path, columns=sorted(columns))
            times = pd.to_datetime(
                table[ROW_TIME_COLUMN[table_name]], errors="coerce", utc=True
            ).dt.tz_localize(None)
            tables[table_name] = table.loc[times.lt(through)].copy()
        return cls(tables)

    def user_positions(self, github_user_ids: list[int]) -> list[int]:
        users = self.tables["Users"]
        lookup = pd.Index(users["github_user_id"])
        positions = lookup.get_indexer(github_user_ids)
        return [int(position) for position in positions]

    def extract_counts(
        self,
        user_position: int,
        *,
        cutoff: str | pd.Timestamp,
        fanout_by_relation: dict[EdgeType, tuple[int | None, ...]],
        random_seed: int,
    ) -> tuple[int, int]:
        cutoff_time = np.datetime64(as_utc_naive_timestamp(cutoff), "ns")
        if not self.node_times["Users"][user_position] < cutoff_time:
            raise ValueError("Sample user was not available before cutoff")
        is_cohort = self.tables["Users"].iloc[user_position]["is_cohort"]
        if pd.isna(is_cohort) or not bool(is_cohort):
            raise ValueError("Sample user is not in the cohort")
        selected: dict[str, set[int]] = {table: set() for table in self.tables}
        selected["Users"].add(user_position)
        frontier: dict[str, set[int]] = {"Users": {user_position}}
        sampled_edges: dict[EdgeType, set[tuple[int, int]]] = {
            edge_type: set() for edge_type in self.adjacencies
        }
        rng = random.Random(random_seed)

        for hop in range(2):
            next_frontier: dict[str, set[int]] = {}
            for edge_type, adjacency in self.adjacencies.items():
                source_type, _, target_type = edge_type
                target_times = self.node_times[target_type]
                fanout = fanout_by_relation[edge_type][hop]
                for source_index in sorted(frontier.get(source_type, ())):
                    candidates = adjacency.for_node(source_index)
                    visible = candidates[target_times[candidates] < cutoff_time]
                    if fanout is not None and len(visible) > fanout:
                        visible = np.asarray(
                            rng.sample(visible.tolist(), fanout), dtype=np.int64
                        )
                    for target_index_value in visible:
                        target_index = int(target_index_value)
                        sampled_edges[edge_type].add((source_index, target_index))
                        if target_index not in selected[target_type]:
                            selected[target_type].add(target_index)
                            next_frontier.setdefault(target_type, set()).add(target_index)
            frontier = next_frontier
            if not frontier:
                break

        node_count = sum(map(len, selected.values()))
        edge_count = sum(map(len, sampled_edges.values()))
        return node_count, edge_count


def _candidate_fanouts() -> dict[
    str, dict[EdgeType, tuple[int | None, int | None]]
]:
    baseline = default_fanout_by_relation()
    smaller = {
        edge_type: ((1, 1) if not relation.startswith("rev_") else (5, 2))
        for edge_type in baseline
        for relation in (edge_type[1],)
    }
    unrestricted = {edge_type: (None, None) for edge_type in baseline}
    return {
        "unrestricted": unrestricted,
        "baseline_10_5": baseline,
        "smaller_5_2": smaller,
    }


def _format_summary(values: list[int]) -> str:
    series = pd.Series(values, dtype="int64")
    quantiles = series.quantile([0.50, 0.90, 0.95, 0.99])
    return (
        f"n={len(values):,} mean={series.mean():.1f} "
        f"p50/p90/p95/p99/max="
        f"{quantiles.iloc[0]:.1f}/{quantiles.iloc[1]:.1f}/"
        f"{quantiles.iloc[2]:.1f}/{quantiles.iloc[3]:.1f}/{series.max()}"
    )


def _load_historical_eligibility_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the cohort and activity times needed for historical eligibility."""
    cohort_users, internal_to_github = _load_cohort_users(data_dir)
    activity_rows = _load_activity_rows(data_dir, internal_to_github)
    all_activities = pd.concat(activity_rows.values(), ignore_index=True)
    return cohort_users, all_activities


def _eligible_users_at_cutoff(
    cohort_users: pd.DataFrame, all_activities: pd.DataFrame, cutoff: pd.Timestamp
) -> list[int]:
    """Apply historical eligibility using only rows known before cutoff."""
    cohort_ids = pd.Index(
        cohort_users.loc[cohort_users["first_seen_at"].lt(cutoff), "github_user_id"]
        .astype("int64")
        .unique()
    )
    in_history = (
        all_activities["row_time"].lt(cutoff)
        & all_activities["activity_time"].ge(cutoff - pd.Timedelta(days=90))
        & all_activities["activity_time"].lt(cutoff)
        & all_activities["UserID"].isin(cohort_ids)
    )
    counts = all_activities.loc[in_history, "UserID"].value_counts()
    return counts.loc[counts.ge(5)].index.astype("int64").tolist()


def profile_subgraph_sizes(
    *,
    data_dir: Path,
    cutoffs: list[str],
    data_through: str,
    sample_size: int,
    random_seed: int,
) -> list[str]:
    """Compare two fanout candidates on a reproducible user/cutoff sample."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    through = as_utc_naive_timestamp(data_through)
    cutoff_times = [as_utc_naive_timestamp(value) for value in cutoffs]
    if any(cutoff >= through for cutoff in cutoff_times):
        raise ValueError("Every sample cutoff must be earlier than data_through")

    profiler = SubgraphSizeProfiler.from_data_dir(data_dir, data_through=through)
    cohort_users, historical_activities = _load_historical_eligibility_data(data_dir)
    lines = [
        f"Rows loaded strictly before data_through={through} UTC (structural columns only).",
        f"Historical eligibility: cohort users with at least 5 activities "
        f"available in the preceding 90 days; maximum samples per cutoff: "
        f"{sample_size:,}.",
    ]
    for cutoff, cutoff_time in zip(cutoffs, cutoff_times):
        eligible_ids = _eligible_users_at_cutoff(
            cohort_users, historical_activities, cutoff_time
        )
        eligible_positions = profiler.user_positions(eligible_ids)
        valid = [
            (github_id, position)
            for github_id, position in zip(eligible_ids, eligible_positions)
            if position >= 0
            and profiler.node_times["Users"][position]
            < np.datetime64(cutoff_time, "ns")
        ]
        if not valid:
            lines.append(f"{cutoff}: no historically eligible users")
            continue
        rng = random.Random(random_seed)
        samples = rng.sample(valid, min(sample_size, len(valid)))
        positions = [position for _, position in samples]
        for candidate_name, fanouts in _candidate_fanouts().items():
            node_counts: list[int] = []
            edge_counts: list[int] = []
            for sample_index, position in enumerate(positions):
                nodes, edges = profiler.extract_counts(
                    position,
                    cutoff=cutoff_time,
                    fanout_by_relation=fanouts,
                    random_seed=random_seed + sample_index,
                )
                node_counts.append(nodes)
                edge_counts.append(edges)
            lines.append(
                f"{cutoff} eligible_users={len(valid):,} {candidate_name}: "
                f"nodes [{_format_summary(node_counts)}]; "
                f"edges [{_format_summary(edge_counts)}]"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sampled subgraph sizes without using labels."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("C:/Users/User/Desktop/assignment2/data/db"),
    )
    parser.add_argument("--cutoff", action="append", required=True)
    parser.add_argument(
        "--data-through",
        required=True,
        help="Known UTC release coverage boundary; do not infer from last event.",
    )
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for line in profile_subgraph_sizes(
        data_dir=args.data_dir,
        cutoffs=args.cutoff,
        data_through=args.data_through,
        sample_size=args.sample_size,
        random_seed=args.seed,
    ):
        print(line)


if __name__ == "__main__":
    main()
