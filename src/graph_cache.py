"""Disk-backed, cutoff-specific feature caches for repeated GNN epochs."""

from __future__ import annotations

import json
from pathlib import Path
import random

import pandas as pd
import torch

from activity_features import (
    ACTIVITY_WINDOWS_DAYS,
    UserActivityHistory,
)
from feature_encoding import TrainFeatureEncoder
from sample_examples import batch_iterator, example_inputs
from temporal_subgraph import EdgeType, TemporalRelationalGraph


CACHE_VERSION = 2


def write_split_cache(
    graph: TemporalRelationalGraph,
    encoder: TrainFeatureEncoder,
    rows: pd.DataFrame,
    *,
    cache_dir: Path,
    split_name: str,
    fanout: dict[EdgeType, tuple[int, int]],
    seed: int,
    training: bool,
    oov_dropout: float,
    sampling_strategy: str = "uniform",
    num_hops: int = 2,
    repo_context: bool = False,
    chunk_size: int = 512,
    activity_history: UserActivityHistory | None = None,
) -> dict:
    """Sample each example once and write bounded-size chunks to disk."""
    split_dir = cache_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=False)
    files = []
    chunk = []
    node_counts: dict[str, int] = {}
    category_counts: dict[str, dict[str, int]] = {}
    numeric_missing: dict[str, dict[str, int]] = {}
    total_examples = 0
    for model_input, label in example_inputs(
        graph,
        encoder,
        rows,
        fanout=fanout,
        seed=seed,
        epoch=0,
        training=training,
        oov_dropout=oov_dropout,
        sampling_strategy=sampling_strategy,
        num_hops=num_hops,
        repo_context=repo_context,
        activity_history=activity_history,
    ):
        chunk.append((model_input, label))
        total_examples += 1
        for node_type, values in model_input.node_features.items():
            node_counts[node_type] = node_counts.get(node_type, 0) + len(values.numeric)
            missing_map = numeric_missing.setdefault(node_type, {})
            for column in values.numeric_missing:
                missing_map[column] = missing_map.get(column, 0) + int(values.numeric_missing[column].sum())
            category_map = category_counts.setdefault(node_type, {})
            for column in values.categorical:
                field = f"{column}__missing"
                category_map[field] = category_map.get(field, 0) + int(values.categorical[column].eq(1).sum())
                field = f"{column}__oov"
                category_map[field] = category_map.get(field, 0) + int(values.categorical[column].eq(0).sum())
        if len(chunk) >= chunk_size:
            file_path = split_dir / f"chunk-{len(files):05d}.pt"
            torch.save(chunk, file_path)
            files.append(file_path.name)
            chunk = []
        if total_examples % 10_000 == 0:
            print(f"[{split_name}] cached {total_examples:,}/{len(rows):,} samples", flush=True)
    if chunk:
        file_path = split_dir / f"chunk-{len(files):05d}.pt"
        torch.save(chunk, file_path)
        files.append(file_path.name)
    if total_examples != len(rows):
        raise ValueError(f"Cached {total_examples} samples, expected {len(rows)}")
    manifest = {
        "version": CACHE_VERSION,
        "split": split_name,
        "sampling_strategy": sampling_strategy,
        "target_user_activity_features": (
            {
                "columns": list(activity_history.columns),
                "count_windows_days": [7] if activity_history.weekly_only else list(ACTIVITY_WINDOWS_DAYS),
                "time_rule": "row_time < cutoff and activity_time < cutoff",
            }
            if activity_history is not None
            else None
        ),
        "num_hops": num_hops,
        "repo_context": repo_context,
        "fanout_by_relation": {"__".join(edge): list(values) for edge, values in fanout.items()},
        "examples": total_examples,
        "chunk_size": chunk_size,
        "files": files,
        "cutoffs": sorted(str(value) for value in rows["Time"].unique()),
        "training_oov_dropout_enabled_for_epoch_batches": bool(training),
        "node_counts": node_counts,
        "numeric_missing_counts": numeric_missing,
        "categorical_missing_and_oov_counts": category_counts,
        "categorical_missing_and_oov_rates": {
            node_type: {
                field: count / node_counts[node_type]
                for field, count in fields.items()
            }
            for node_type, fields in category_counts.items()
            if node_counts[node_type]
        },
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def cached_batches(
    split_dir: Path,
    encoder: TrainFeatureEncoder,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    training: bool,
    oov_dropout: float,
):
    """Yield disjoint mini-batches from a completed split cache."""
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") not in {1, CACHE_VERSION}:
        raise ValueError(f"Unsupported graph cache version in {split_dir}")
    file_names = list(manifest["files"])
    rng = random.Random(seed + epoch * 1_000_003)
    if training:
        rng.shuffle(file_names)
    def examples():
        dropout_index = 0
        for file_name in file_names:
            path = split_dir / file_name
            stored = torch.load(path, map_location="cpu", weights_only=False)
            indices = list(range(len(stored)))
            if training:
                rng.shuffle(indices)
            for index in indices:
                model_input, label = stored[index]
                if training and oov_dropout:
                    model_input = encoder.apply_training_oov_dropout(
                        model_input,
                        probability=oov_dropout,
                        random_seed=seed + epoch * 2_000_003 + dropout_index,
                    )
                dropout_index += 1
                yield model_input, label
    yield from batch_iterator(examples(), batch_size)
