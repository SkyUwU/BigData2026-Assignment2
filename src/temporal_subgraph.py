"""Build one temporal relational graph and extract per-sample K-hop subgraphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Mapping, Sequence

import pandas as pd

from data import (
    ROW_TIME_COLUMN,
    STRUCTURAL_COLUMNS,
    as_utc_naive_timestamp,
)
from feature_policy import feature_columns
from relations import PRIMARY_KEYS, build_relation_edges


EdgeType = tuple[str, str, str]
REPO_CONTEXT_ARCHITECTURE = "relgnn_repo_context_v1"
REPO_CONTEXT_EDGES = (
    ("Issues", "repo_context", "Repositories"),
    ("IssueComments", "repo_context", "Repositories"),
)
REPO_CONTEXT_FANOUT = 2
REPO_CONTEXT_DAYS = 7


@dataclass(frozen=True)
class UserSubgraph:
    """A local graph around one user, using row positions from the global graph."""

    user_id: int
    node_indices: dict[str, tuple[int, ...]]
    edge_index: dict[EdgeType, tuple[tuple[int, ...], tuple[int, ...]]]
    seed_node_index: int
    cutoff: pd.Timestamp


class TemporalRelationalGraph:
    """One all-time graph; each sample gets a time-filtered K-hop subgraph."""

    def __init__(self, tables: dict[str, pd.DataFrame], *, cohort_ids: pd.Index | None = None):
        self.cohort_ids = frozenset(cohort_ids) if cohort_ids is not None else frozenset()
        if cohort_ids is not None:
            users = tables["Users"]
            missing = cohort_ids.difference(users["github_user_id"])
            first_unused = int(users["user_id"].max()) + 1 if len(users) else 0
            placeholders = pd.DataFrame({
                "user_id": range(first_unused, first_unused + len(missing)),
                "github_user_id": missing,
                "is_cohort": True,
                ROW_TIME_COLUMN["Users"]: pd.NaT,
            })
            if len(placeholders):
                tables = {**tables, "Users": pd.concat([users, placeholders], ignore_index=True)}
        self.tables = tables
        self._validate_tables()
        self._node_times = {
            table_name: pd.to_datetime(
                table[ROW_TIME_COLUMN[table_name]], errors="coerce", utc=True
            ).dt.tz_localize(None)
            for table_name, table in tables.items()
        }
        self.edges = build_relation_edges(tables)
        self._adjacency = self._build_adjacency()

    @classmethod
    def from_data_dir(
        cls, data_dir: Path, *, cohort_ids: pd.Index | None = None
    ) -> "TemporalRelationalGraph":
        """Load and index all available rows once; sampling applies time limits."""
        tables: dict[str, pd.DataFrame] = {}
        for table_name, columns in STRUCTURAL_COLUMNS.items():
            table_path = data_dir / f"{table_name}.parquet"
            if not table_path.is_file():
                raise FileNotFoundError(f"Missing table: {table_path}")
            selected_columns = tuple(dict.fromkeys((*columns, *feature_columns(table_name))))
            tables[table_name] = pd.read_parquet(
                table_path, columns=list(selected_columns)
            )
        return cls(tables, cohort_ids=cohort_ids)

    def _validate_tables(self) -> None:
        missing = set(PRIMARY_KEYS).difference(self.tables)
        if missing:
            raise ValueError(f"Graph is missing tables: {', '.join(sorted(missing))}")
        required_user_columns = {"github_user_id", "is_cohort"}
        missing_user_columns = required_user_columns.difference(
            self.tables["Users"].columns
        )
        if missing_user_columns:
            raise ValueError(
                "Users is missing required columns: "
                + ", ".join(sorted(missing_user_columns))
            )

        for table_name, table in self.tables.items():
            time_column = ROW_TIME_COLUMN[table_name]
            if time_column not in table:
                raise ValueError(f"{table_name} is missing row-time column {time_column!r}")
            times = pd.to_datetime(table[time_column], errors="coerce", utc=True)
            invalid_times = table[time_column].notna() & times.isna()
            if invalid_times.any():
                raise ValueError(f"{table_name} contains an invalid row time")

    def _build_adjacency(self) -> dict[EdgeType, dict[int, tuple[int, ...]]]:
        adjacency: dict[EdgeType, dict[int, list[int]]] = {}
        for edge_type, (sources, targets) in self.edges.items():
            neighbors: dict[int, list[int]] = {}
            for source, target in zip(sources, targets):
                neighbors.setdefault(source, []).append(target)
            adjacency[edge_type] = neighbors
        return {
            edge_type: {source: tuple(targets) for source, targets in neighbors.items()}
            for edge_type, neighbors in adjacency.items()
        }

    def extract_user_subgraph(
        self,
        github_user_id: int,
        *,
        num_hops: int,
        cutoff: str | pd.Timestamp,
        fanout_by_relation: Mapping[EdgeType, int | Sequence[int]] | None = None,
        random_seed: int = 0,
        sampling_strategy: str = "uniform",
        repo_context: bool = False,
    ) -> UserSubgraph:
        """Extract a time-safe K-hop subgraph around a cohort user.

        All relation directions are traversable, including the reverse edges
        added by ``build_relation_edges``. If a relation has a fanout, it is
        applied per frontier node and hop after filtering neighbors by time.
        Relations omitted from ``fanout_by_relation`` keep all visible neighbors.
        Edges in the result are only those actually traversed during sampling.
        """
        if num_hops < 0:
            raise ValueError("num_hops must be non-negative")
        if repo_context and num_hops != 2:
            raise ValueError("Repo context extends a two-hop base subgraph")
        if sampling_strategy not in {"uniform", "latest"}:
            raise ValueError("sampling_strategy must be 'uniform' or 'latest'")
        fanout_by_relation = fanout_by_relation or {}
        unknown_relations = set(fanout_by_relation).difference(self.edges)
        if unknown_relations:
            raise ValueError(f"Unknown fanout edge types: {sorted(unknown_relations)}")
        fanouts_by_hop: dict[EdgeType, tuple[int, ...]] = {}
        for edge_type, fanout in fanout_by_relation.items():
            if isinstance(fanout, int) and not isinstance(fanout, bool):
                values = (fanout,) * num_hops
            elif isinstance(fanout, Sequence) and not isinstance(fanout, (str, bytes)):
                values = tuple(fanout)
                if len(values) != num_hops:
                    raise ValueError(
                        f"Fanout sequence for {edge_type} must have "
                        f"{num_hops} values, one per hop"
                    )
            else:
                raise ValueError("fanout values must be integers or integer sequences")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in values
            ):
                raise ValueError("fanout values must be non-negative integers")
            fanouts_by_hop[edge_type] = values
        sample_cutoff = as_utc_naive_timestamp(cutoff)
        rng = random.Random(random_seed)

        users = self.tables["Users"]
        matches = users.index[users["github_user_id"].eq(github_user_id)].tolist()
        if len(matches) != 1:
            raise ValueError(
                f"Expected one user with github_user_id={github_user_id}; found {len(matches)}"
            )

        # Edge indices refer to row positions, not the DataFrame's index labels.
        seed_row_position = users.index.get_loc(matches[0])
        is_cohort = users.iloc[seed_row_position]["is_cohort"]
        if github_user_id not in self.cohort_ids and (pd.isna(is_cohort) or not bool(is_cohort)):
            raise ValueError(
                f"User github_user_id={github_user_id} is not a cohort user"
            )
        seed_time = self._node_times["Users"].iloc[seed_row_position]
        seed_visible = pd.notna(seed_time) and seed_time < sample_cutoff
        if not seed_visible and github_user_id not in self.cohort_ids:
            raise ValueError(
                f"User github_user_id={github_user_id} was not available before cutoff"
            )
        selected: dict[str, set[int]] = {name: set() for name in self.tables}
        selected["Users"].add(seed_row_position)
        frontier: dict[str, set[int]] = {"Users": {seed_row_position}} if seed_visible else {}
        sampled_edges: dict[EdgeType, set[tuple[int, int]]] = {
            edge_type: set() for edge_type in self.edges
        }

        for hop_index in range(num_hops):
            next_frontier: dict[str, set[int]] = {}
            for edge_type, neighbors_by_source in self._adjacency.items():
                source_type, _, target_type = edge_type
                for source_index in sorted(frontier.get(source_type, ())):
                    visible_neighbors = [
                        target_index
                        for target_index in neighbors_by_source.get(source_index, ())
                        if pd.notna(
                            self._node_times[target_type].iloc[target_index]
                        )
                        and self._node_times[target_type].iloc[target_index]
                        < sample_cutoff
                    ]
                    relation_fanouts = fanouts_by_hop.get(edge_type)
                    fanout = (
                        relation_fanouts[hop_index]
                        if relation_fanouts is not None
                        else None
                    )
                    if fanout is not None and len(visible_neighbors) > fanout:
                        if sampling_strategy == "latest":
                            visible_neighbors = sorted(
                                visible_neighbors,
                                key=lambda index: (
                                    self._node_times[target_type].iloc[index], index
                                ),
                                reverse=True,
                            )[:fanout]
                        else:
                            visible_neighbors = rng.sample(visible_neighbors, fanout)

                    for target_index in visible_neighbors:
                        sampled_edges[edge_type].add((source_index, target_index))
                        target_time = self._node_times[target_type].iloc[target_index]
                        if (
                            pd.notna(target_time)
                            and target_time < sample_cutoff
                            and target_index not in selected[target_type]
                        ):
                            selected[target_type].add(target_index)
                            next_frontier.setdefault(target_type, set()).add(target_index)
            frontier = next_frontier
            if not frontier:
                break

        if repo_context:
            lower = sample_cutoff - pd.Timedelta(days=REPO_CONTEXT_DAYS)
            actor_id = users.iloc[seed_row_position]["user_id"]
            for edge in REPO_CONTEXT_EDGES:
                event_type = edge[0]
                neighbors = self._adjacency[("Repositories", "rev_in_repo", event_type)]
                events = self.tables[event_type]
                sampled_edges[edge] = set()
                for repo in selected["Repositories"]:
                    candidates = []
                    for position in neighbors.get(repo, ()):
                        row_time = self._node_times[event_type].iat[position]
                        if pd.isna(row_time) or not lower <= row_time < sample_cutoff:
                            continue
                        event = events.iloc[position]
                        activity_time = event["created_at"]
                        author = event["author_id"]
                        if pd.isna(author) or author == actor_id or pd.isna(activity_time):
                            continue
                        activity_time = as_utc_naive_timestamp(activity_time)
                        if not lower <= activity_time < sample_cutoff:
                            continue
                        if event_type == "Issues" and (
                            pd.isna(event["is_pull_request"]) or event["is_pull_request"]
                        ):
                            continue
                        candidates.append((activity_time, position))
                    for _, position in sorted(candidates, reverse=True)[:REPO_CONTEXT_FANOUT]:
                        selected[event_type].add(position)
                        sampled_edges[edge].add((position, repo))

        ordered_nodes = {
            node_type: tuple(sorted(indices)) for node_type, indices in selected.items()
        }
        local_index = {
            node_type: {old: new for new, old in enumerate(indices)}
            for node_type, indices in ordered_nodes.items()
        }

        subgraph_edges: dict[EdgeType, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for edge_type, edges in sampled_edges.items():
            source_type, _, target_type = edge_type
            ordered_edges = sorted(edges)
            subgraph_edges[edge_type] = (
                tuple(local_index[source_type][source] for source, _ in ordered_edges),
                tuple(local_index[target_type][target] for _, target in ordered_edges),
            )

        return UserSubgraph(
            user_id=int(github_user_id),
            node_indices=ordered_nodes,
            edge_index=subgraph_edges,
            seed_node_index=local_index["Users"][seed_row_position],
            cutoff=sample_cutoff,
        )
