"""Generate model inputs and labels from per-example temporal subgraphs."""

from __future__ import annotations

import pandas as pd

from activity_features import UserActivityHistory
from feature_encoding import TrainFeatureEncoder
from gnn_model import collate_subgraphs
from labels import LABEL_COLUMNS
from model_input import materialize_subgraph_input
from temporal_subgraph import TemporalRelationalGraph


def example_inputs(
    graph: TemporalRelationalGraph,
    encoder: TrainFeatureEncoder,
    rows: pd.DataFrame,
    *,
    fanout: dict,
    seed: int,
    epoch: int,
    training: bool,
    oov_dropout: float,
    sampling_strategy: str = "uniform",
    num_hops: int = 2,
    repo_context: bool = False,
    activity_history: UserActivityHistory | None = None,
):
    for index, row in enumerate(rows.itertuples(index=False)):
        subgraph = graph.extract_user_subgraph(
            int(row.UserID),
            num_hops=num_hops,
            cutoff=row.Time,
            fanout_by_relation=fanout,
            random_seed=seed + epoch * 1_000_003 + index,
            sampling_strategy=sampling_strategy,
            repo_context=repo_context,
        )
        raw_input = materialize_subgraph_input(graph, subgraph, activity_history)
        encoded = encoder.transform(raw_input)
        if training and oov_dropout:
            encoded = encoder.apply_training_oov_dropout(
                encoded,
                probability=oov_dropout,
                random_seed=seed + epoch * 2_000_003 + index,
            )
        label = (
            [int(getattr(row, label_name)) for label_name in LABEL_COLUMNS]
            if set(LABEL_COLUMNS).issubset(rows.columns)
            else None
        )
        yield encoded, label


def batch_iterator(examples, batch_size: int):
    batch_inputs, batch_labels = [], []
    for model_input, label in examples:
        batch_inputs.append(model_input)
        batch_labels.append(label)
        if len(batch_inputs) == batch_size:
            yield collate_subgraphs(
                batch_inputs, None if batch_labels[0] is None else batch_labels
            )
            batch_inputs, batch_labels = [], []
    if batch_inputs:
        yield collate_subgraphs(
            batch_inputs, None if batch_labels[0] is None else batch_labels
        )
