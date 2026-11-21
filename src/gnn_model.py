"""A small native-PyTorch heterogeneous GNN for sample-local subgraphs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from feature_encoding import EncodedSubgraphInput, TrainFeatureEncoder
from relations import FOREIGN_KEYS
from temporal_subgraph import EdgeType, REPO_CONTEXT_ARCHITECTURE, REPO_CONTEXT_EDGES


@dataclass(frozen=True)
class TensorGraphBatch:
    """Disjoint union of independently sampled user subgraphs."""

    numeric: dict[str, torch.Tensor]
    categorical: dict[str, dict[str, torch.Tensor]]
    edge_index: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]]
    seed_nodes: torch.Tensor
    labels: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "TensorGraphBatch":
        return TensorGraphBatch(
            numeric={key: value.to(device) for key, value in self.numeric.items()},
            categorical={
                node_type: {name: value.to(device) for name, value in fields.items()}
                for node_type, fields in self.categorical.items()
            },
            edge_index={
                key: (source.to(device), target.to(device))
                for key, (source, target) in self.edge_index.items()
            },
            seed_nodes=self.seed_nodes.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
        )


def collate_subgraphs(
    inputs: list[EncodedSubgraphInput],
    labels: list[list[int]] | None = None,
) -> TensorGraphBatch:
    """Pack encoded subgraphs with node-type-specific offsets."""
    if not inputs:
        raise ValueError("Cannot collate an empty subgraph batch")
    if labels is not None and len(labels) != len(inputs):
        raise ValueError("The number of label rows must match the subgraph batch")

    node_types = list(inputs[0].node_features)
    if any(list(item.node_features) != node_types for item in inputs):
        raise ValueError("All batch inputs must contain node types in the same order")

    numeric: dict[str, torch.Tensor] = {}
    categorical: dict[str, dict[str, torch.Tensor]] = {}
    offsets: dict[str, list[int]] = {node_type: [] for node_type in node_types}
    totals = {node_type: 0 for node_type in node_types}
    seeds: list[int] = []

    for item in inputs:
        seeds.append(totals["Users"] + item.seed_node_index)
        for node_type in node_types:
            offsets[node_type].append(totals[node_type])
            features = item.node_features[node_type]
            totals[node_type] += len(features.numeric)

    for node_type in node_types:
        numeric[node_type] = torch.from_numpy(
            pd.concat(
                [item.node_features[node_type].numeric_with_missing for item in inputs],
                ignore_index=True,
            ).to_numpy(dtype="float32", copy=True)
        )
        columns = list(inputs[0].node_features[node_type].categorical.columns)
        categorical[node_type] = {
            column: torch.from_numpy(
                pd.concat(
                    [item.node_features[node_type].categorical[column] for item in inputs],
                    ignore_index=True,
                ).to_numpy(dtype="int64", copy=True)
            )
            for column in columns
        }

    all_edge_types = list(inputs[0].edge_index)
    edge_index: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]] = {}
    for edge_type in all_edge_types:
        source_type, _, target_type = edge_type
        source_parts: list[int] = []
        target_parts: list[int] = []
        for item_index, item in enumerate(inputs):
            source, target = item.edge_index[edge_type]
            source_offset = offsets[source_type][item_index]
            target_offset = offsets[target_type][item_index]
            source_parts.extend(source_offset + value for value in source)
            target_parts.extend(target_offset + value for value in target)
        edge_index[edge_type] = (
            torch.tensor(source_parts, dtype=torch.long),
            torch.tensor(target_parts, dtype=torch.long),
        )

    label_tensor = (
        torch.tensor(labels, dtype=torch.float32) if labels is not None else None
    )
    return TensorGraphBatch(
        numeric=numeric,
        categorical=categorical,
        edge_index=edge_index,
        seed_nodes=torch.tensor(seeds, dtype=torch.long),
        labels=label_tensor,
    )


def _relation_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


class RelationalMessageLayer(nn.Module):
    """Relation-specific messages with per-relation mean aggregation."""

    def __init__(self, node_types: tuple[str, ...], edge_types: tuple[EdgeType, ...], hidden_size: int):
        super().__init__()
        self.self_linear = nn.ModuleDict(
            {node_type: nn.Linear(hidden_size, hidden_size) for node_type in node_types}
        )
        self.relation_linear = nn.ModuleDict(
            {_relation_key(edge_type): nn.Linear(hidden_size, hidden_size, bias=False)
             for edge_type in edge_types}
        )
        self.norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_size) for node_type in node_types}
        )

    def forward(
        self,
        hidden: dict[str, torch.Tensor],
        edge_index: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        relation_sum = {
            node_type: torch.zeros_like(values) for node_type, values in hidden.items()
        }
        relation_count = {
            node_type: values.new_zeros((len(values), 1))
            for node_type, values in hidden.items()
        }
        for edge_type, (sources, targets) in edge_index.items():
            if sources.numel() == 0:
                continue
            source_type, _, target_type = edge_type
            messages = self.relation_linear[_relation_key(edge_type)](
                hidden[source_type][sources]
            )
            aggregate = torch.zeros_like(hidden[target_type])
            aggregate.index_add_(0, targets, messages)
            degree = hidden[target_type].new_zeros((len(hidden[target_type]), 1))
            degree.index_add_(0, targets, messages.new_ones((len(targets), 1)))
            active = degree.gt(0)
            relation_sum[target_type] += aggregate / degree.clamp_min(1.0)
            relation_count[target_type] += active.to(relation_count[target_type].dtype)

        result = {}
        for node_type, values in hidden.items():
            neighbor_mean = relation_sum[node_type] / relation_count[node_type].clamp_min(1.0)
            result[node_type] = F.relu(
                self.norm[node_type](self.self_linear[node_type](values) + neighbor_mean)
            )
        return result


class RelationalActivityGNN(nn.Module):
    """Encode row features, pass typed messages twice, and classify the seed user."""

    def __init__(
        self,
        encoder: TrainFeatureEncoder,
        edge_types: tuple[EdgeType, ...],
        *,
        hidden_size: int = 64,
        dropout: float = 0.2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or not 0 <= dropout < 1 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive and dropout must be in [0, 1)")
        self.node_types = tuple(encoder.numeric_columns)
        self.categorical_columns = encoder.categorical_columns
        self.dropout = dropout
        self.input_projection = nn.ModuleDict()
        self.category_embeddings = nn.ModuleDict()
        self._category_keys: dict[tuple[str, str], str] = {}
        for node_type in self.node_types:
            input_width = 2 * len(encoder.numeric_columns[node_type])
            for column in encoder.categorical_columns[node_type]:
                key = f"{node_type}__{column}"
                self._category_keys[(node_type, column)] = key
                cardinality = max(2, len(encoder.category_ids[(node_type, column)]) + 2)
                embedding_width = min(16, max(4, hidden_size // 8))
                self.category_embeddings[key] = nn.Embedding(cardinality, embedding_width)
                input_width += embedding_width
            self.input_projection[node_type] = nn.Linear(input_width, hidden_size)

        self.layers = nn.ModuleList(
            [
                RelationalMessageLayer(self.node_types, edge_types, hidden_size)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 6),
        )

    def forward(self, batch: TensorGraphBatch) -> torch.Tensor:
        hidden: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            parts = [batch.numeric[node_type].float()]
            for column in self.categorical_columns[node_type]:
                key = self._category_keys[(node_type, column)]
                parts.append(self.category_embeddings[key](batch.categorical[node_type][column]))
            hidden[node_type] = F.relu(self.input_projection[node_type](torch.cat(parts, dim=1)))

        for layer in self.layers:
            hidden = layer(hidden, batch.edge_index)
            hidden = {key: F.dropout(value, p=self.dropout, training=self.training)
                      for key, value in hidden.items()}
        return self.classifier(hidden["Users"][batch.seed_nodes])


@dataclass(frozen=True)
class AtomicRoute:
    """One schema-derived RelGNN route over a sampled relational subgraph."""

    key: str
    source_type: str
    source_edge: EdgeType
    middle_type: str | None
    destination_type: str
    destination_edge: EdgeType


def build_atomic_routes(edge_types: tuple[EdgeType, ...]) -> tuple[AtomicRoute, ...]:
    """Derive RelGNN routes from FK roles, retaining distinct FK columns.

    A single-FK table gives two directed dim-dim routes. A table with multiple
    FKs gives every ordered pair of distinct FK roles, matching the authors'
    atomic_routes.py construction. Self-referential FKs remain distinct roles.
    """
    available = set(edge_types)
    by_fact: dict[str, list[tuple[str, str, str, str]]] = {}
    for fact_type, _column, referenced_type, relation in FOREIGN_KEYS:
        by_fact.setdefault(fact_type, []).append(
            (fact_type, _column, referenced_type, relation)
        )

    routes: list[AtomicRoute] = []
    for fact_type, foreign_keys in by_fact.items():
        if len(foreign_keys) == 1:
            _fact, _column, referenced_type, relation = foreign_keys[0]
            forward = (fact_type, relation, referenced_type)
            reverse = (referenced_type, f"rev_{relation}", fact_type)
            if forward in available:
                routes.append(
                    AtomicRoute(
                        key=f"dim_dim__{fact_type}__{relation}__{referenced_type}",
                        source_type=fact_type,
                        source_edge=forward,
                        middle_type=None,
                        destination_type=referenced_type,
                        destination_edge=forward,
                    )
                )
            if reverse in available:
                routes.append(
                    AtomicRoute(
                        key=f"dim_dim__{referenced_type}__rev_{relation}__{fact_type}",
                        source_type=referenced_type,
                        source_edge=reverse,
                        middle_type=None,
                        destination_type=fact_type,
                        destination_edge=reverse,
                    )
                )
            continue

        for source_fk in foreign_keys:
            source_type = source_fk[2]
            source_relation = source_fk[3]
            source_edge = (source_type, f"rev_{source_relation}", fact_type)
            for destination_fk in foreign_keys:
                if source_fk[3] == destination_fk[3]:
                    continue
                destination_type = destination_fk[2]
                destination_relation = destination_fk[3]
                destination_edge = (fact_type, destination_relation, destination_type)
                if source_edge not in available or destination_edge not in available:
                    continue
                routes.append(
                    AtomicRoute(
                        key=(
                            f"dim_fact_dim__{source_type}__{source_relation}__"
                            f"{fact_type}__{destination_relation}__{destination_type}"
                        ),
                        source_type=source_type,
                        source_edge=source_edge,
                        middle_type=fact_type,
                        destination_type=destination_type,
                        destination_edge=destination_edge,
                    )
                )
    return tuple(routes)


class AtomicRouteAttention(nn.Module):
    """Route-specific composite fusion and destination-query attention."""

    def __init__(self, hidden_size: int, num_heads: int, *, composite: bool) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.composite = composite
        if composite:
            self.source_to_middle = nn.Linear(hidden_size, hidden_size, bias=False)
            self.middle_self = nn.Linear(hidden_size, hidden_size, bias=True)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.destination = nn.Linear(hidden_size, hidden_size, bias=False)

    def fuse_middle(
        self,
        middle: torch.Tensor,
        source: torch.Tensor,
        source_indices: torch.Tensor,
        middle_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GraphSAGE-style FUSE, equivalent to the paper's learned linear sum."""
        assert self.composite
        source_count = middle.new_zeros((len(middle), 1))
        source_sum = middle.new_zeros((len(middle), self.hidden_size))
        if len(source_indices):
            source_sum.index_add_(
                0,
                middle_indices,
                self.source_to_middle(source[source_indices]),
            )
            source_count.index_add_(
                0,
                middle_indices,
                middle.new_ones((len(middle_indices), 1)),
            )
        fused = self.middle_self(middle) + source_sum / source_count.clamp_min(1.0)
        return fused, source_count.squeeze(1).gt(0)

    def forward(
        self,
        destination: torch.Tensor,
        candidate: torch.Tensor,
        candidate_indices: torch.Tensor,
        destination_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Attend per destination; nodes without route neighbors receive zero."""
        node_count = len(destination)
        if not len(candidate_indices):
            return destination.new_zeros((node_count, self.hidden_size))

        query = self.query(destination).view(node_count, self.num_heads, self.head_size)
        key = self.key(candidate[candidate_indices]).view(-1, self.num_heads, self.head_size)
        value = self.value(candidate[candidate_indices]).view(-1, self.num_heads, self.head_size)
        scores = (query[destination_indices] * key).sum(dim=-1) / (self.head_size ** 0.5)

        # Stable segmented softmax over candidate rows belonging to each target.
        max_scores = scores.new_full((node_count, self.num_heads), -torch.inf)
        max_scores.scatter_reduce_(
            0,
            destination_indices[:, None].expand(-1, self.num_heads),
            scores,
            reduce="amax",
            include_self=True,
        )
        weights = torch.exp(scores - max_scores[destination_indices])
        normalizers = scores.new_zeros((node_count, self.num_heads))
        normalizers.index_add_(0, destination_indices, weights)
        weights = weights / normalizers[destination_indices].clamp_min(1e-12)

        aggregated = value.new_zeros((node_count, self.num_heads, self.head_size))
        aggregated.index_add_(
            0,
            destination_indices,
            weights.unsqueeze(-1) * value,
        )
        combined = aggregated.reshape(node_count, self.hidden_size)
        # A route is active per destination, not per mini-batch. Otherwise a
        # neighbor in an unrelated subgraph would enable this self projection.
        active = normalizers.sum(dim=1).gt(0).unsqueeze(1)
        return torch.where(active, self.destination(destination) + combined, torch.zeros_like(combined))


class RelGNNMessageLayer(nn.Module):
    """RelGNN composite message passing over schema-derived atomic routes."""

    def __init__(
        self,
        node_types: tuple[str, ...],
        routes: tuple[AtomicRoute, ...],
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.routes = routes
        self.route_modules = nn.ModuleDict(
            {
                route.key: AtomicRouteAttention(
                    hidden_size, num_heads, composite=route.middle_type is not None
                )
                for route in routes
            }
        )
        self.norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_size) for node_type in node_types}
        )

    @staticmethod
    def _directed_edges(
        edge_type: EdgeType,
        edge_index: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read sampled FK links in the requested direction without new links."""
        source_type, relation, destination_type = edge_type
        inverse_relation = (
            relation[len("rev_") :]
            if relation.startswith("rev_")
            else f"rev_{relation}"
        )
        inverse_type = (destination_type, inverse_relation, source_type)
        direct = edge_index.get(edge_type)
        inverse = edge_index.get(inverse_type)
        template = direct or inverse
        empty = (
            template[0].new_empty((0,))
            if template is not None
            else torch.empty(0, dtype=torch.long)
        )
        direct_source, direct_destination = direct if direct is not None else (empty, empty)
        inverse_source, inverse_destination = inverse if inverse is not None else (empty, empty)
        sources = torch.cat((direct_source, inverse_destination))
        destinations = torch.cat((direct_destination, inverse_source))
        if not sources.numel():
            return sources, destinations
        # Sampling may have traversed the same FK in both directions. Keep one
        # candidate per row-pair so attention does not count that link twice.
        pairs = torch.unique(torch.stack((sources, destinations), dim=1), dim=0)
        return pairs[:, 0], pairs[:, 1]

    def forward(
        self,
        hidden: dict[str, torch.Tensor],
        edge_index: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        route_sum = {node_type: torch.zeros_like(value) for node_type, value in hidden.items()}
        route_seen = {
            node_type: torch.zeros(len(value), dtype=torch.bool, device=value.device)
            for node_type, value in hidden.items()
        }
        oriented_edges: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]] = {}

        def get_edges(edge_type: EdgeType) -> tuple[torch.Tensor, torch.Tensor]:
            if edge_type not in oriented_edges:
                oriented_edges[edge_type] = self._directed_edges(edge_type, edge_index)
            return oriented_edges[edge_type]

        for route in self.routes:
            module = self.route_modules[route.key]
            destination = hidden[route.destination_type]
            if route.middle_type is None:
                candidate_indices, destination_indices = get_edges(route.source_edge)
                candidate = hidden[route.source_type]
            else:
                source_indices, middle_source_indices = get_edges(route.source_edge)
                middle = hidden[route.middle_type]
                candidate, has_source = module.fuse_middle(
                    middle,
                    hidden[route.source_type],
                    source_indices,
                    middle_source_indices,
                )
                middle_target_indices, destination_indices = get_edges(route.destination_edge)
                valid = has_source[middle_target_indices]
                candidate_indices = middle_target_indices[valid]
                destination_indices = destination_indices[valid]
            if not len(candidate_indices):
                continue
            route_sum[route.destination_type] += module(
                destination,
                candidate,
                candidate_indices,
                destination_indices,
            )
            route_seen[route.destination_type][destination_indices] = True

        result = {}
        for node_type, value in hidden.items():
            # Preserve the original representation only for individual nodes
            # with no valid routes, regardless of other graphs in the batch.
            updated = torch.where(route_seen[node_type].unsqueeze(1), route_sum[node_type], value)
            result[node_type] = F.relu(self.norm[node_type](updated))
        return result


class RelGNNActivityGNN(nn.Module):
    """One or two composite layers over the same sampled two-hop graph."""

    architecture = "relgnn_atomic_routes_v1"

    def __init__(
        self,
        encoder: TrainFeatureEncoder,
        edge_types: tuple[EdgeType, ...],
        *,
        hidden_size: int = 64,
        dropout: float = 0.2,
        num_layers: int = 1,
        attention_heads: int = 4,
        repo_context: bool = False,
    ) -> None:
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError(
                "RelGNN supports one or two composite layers on the fixed two-hop graph."
            )
        if repo_context and num_layers != 1:
            raise ValueError("Repo-context experiments retain exactly one composite layer")
        if hidden_size <= 0 or not 0 <= dropout < 1:
            raise ValueError("hidden_size must be positive and dropout must be in [0, 1)")
        self.node_types = tuple(encoder.numeric_columns)
        self.categorical_columns = encoder.categorical_columns
        self.dropout = dropout
        self.input_projection = nn.ModuleDict()
        self.category_embeddings = nn.ModuleDict()
        self._category_keys: dict[tuple[str, str], str] = {}
        for node_type in self.node_types:
            input_width = 2 * len(encoder.numeric_columns[node_type])
            for column in encoder.categorical_columns[node_type]:
                key = f"{node_type}__{column}"
                self._category_keys[(node_type, column)] = key
                cardinality = max(2, len(encoder.category_ids[(node_type, column)]) + 2)
                embedding_width = min(16, max(4, hidden_size // 8))
                self.category_embeddings[key] = nn.Embedding(cardinality, embedding_width)
                input_width += embedding_width
            self.input_projection[node_type] = nn.Linear(input_width, hidden_size)

        routes = build_atomic_routes(edge_types)
        if not routes:
            raise ValueError("No atomic routes can be built from the supplied FK edge types")
        self.routes = routes
        self.layers = nn.ModuleList(
            [RelGNNMessageLayer(self.node_types, routes, hidden_size, attention_heads)
             for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 6),
        )
        self.repo_context = nn.ModuleDict({
            edge[0]: AtomicRouteAttention(hidden_size, attention_heads, composite=False)
            for edge in REPO_CONTEXT_EDGES
        } if repo_context else {})

    def forward(self, batch: TensorGraphBatch) -> torch.Tensor:
        hidden: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            parts = [batch.numeric[node_type].float()]
            for column in self.categorical_columns[node_type]:
                key = self._category_keys[(node_type, column)]
                parts.append(self.category_embeddings[key](batch.categorical[node_type][column]))
            hidden[node_type] = F.relu(self.input_projection[node_type](torch.cat(parts, dim=1)))

        if self.repo_context:
            repos = hidden["Repositories"]
            context = torch.zeros_like(repos)
            for edge in REPO_CONTEXT_EDGES:
                source, target = batch.edge_index[edge]
                context = context + self.repo_context[edge[0]](
                    repos, hidden[edge[0]], source, target
                )
            hidden["Repositories"] = repos + context

        for layer in self.layers:
            hidden = layer(hidden, batch.edge_index)
            hidden = {
                key: F.dropout(value, p=self.dropout, training=self.training)
                for key, value in hidden.items()
            }
        return self.classifier(hidden["Users"][batch.seed_nodes])


def build_activity_model(
    encoder: TrainFeatureEncoder,
    edge_types: tuple[EdgeType, ...],
    *,
    architecture: str,
    hidden_size: int,
    dropout: float,
    num_layers: int,
    attention_heads: int = 4,
) -> nn.Module:
    """Build a checkpoint-compatible model; old checkpoints default to baseline."""
    if architecture in {RelGNNActivityGNN.architecture, REPO_CONTEXT_ARCHITECTURE}:
        return RelGNNActivityGNN(
            encoder,
            edge_types,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            attention_heads=attention_heads,
            repo_context=architecture == REPO_CONTEXT_ARCHITECTURE,
        )
    if architecture == "hetero_mean_baseline":
        return RelationalActivityGNN(
            encoder,
            edge_types,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")
