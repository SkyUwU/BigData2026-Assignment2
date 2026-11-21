"""Predict official test users, or run frozen-checkpoint 'calibrate' and 'predict' commands."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from activity_features import UserActivityHistory
from feature_encoding import TrainFeatureEncoder
from gnn_model import build_activity_model
from graph_cache import cached_batches, write_split_cache
from labels import LABEL_COLUMNS
from sampling_config import default_fanout_by_relation
from temporal_subgraph import REPO_CONTEXT_ARCHITECTURE, TemporalRelationalGraph
from training_table import build_training_table
from train_model import _encoder_fit_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--test-cutoff", required=True)
    parser.add_argument("--data-through", required=True)
    parser.add_argument("--train-start")
    parser.add_argument("--validation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-users", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite prediction file: {args.output.resolve()}")
    model_output = args.output.with_suffix(".model.pt")
    if model_output.exists():
        raise FileExistsError(f"Refusing to overwrite model file: {model_output.resolve()}")
    cache_dir = (args.cache_dir or args.output.with_name(args.output.stem + ".cache")).resolve()
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing graph cache: {cache_dir}")
    checkpoint_path = args.validation_checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["labels"] != list(LABEL_COLUMNS):
        raise ValueError("Validation checkpoint label order does not match the task")
    config = checkpoint["config"]
    batch_size = args.batch_size or int(config["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_through = pd.Timestamp(args.data_through)
    test_cutoff = pd.Timestamp(args.test_cutoff)
    if data_through.tz is not None:
        data_through = data_through.tz_convert("UTC").tz_localize(None)
    if test_cutoff.tz is not None:
        test_cutoff = test_cutoff.tz_convert("UTC").tz_localize(None)
    if data_through > test_cutoff:
        raise ValueError("data_through must not be later than the official test cutoff")
    start = pd.Timestamp(args.train_start or config.get("train_start", "2026-04-28T16:00:00Z"))
    if start.tz is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if config.get("full_cohort_daily"):
        raise ValueError("Retrain the validation checkpoint with the observed-cohort daily mode")
    daily = bool(config.get("observed_cohort_daily", False))
    input_cutoff = test_cutoff - pd.Timedelta(days=7) if daily else test_cutoff
    if daily and data_through < input_cutoff:
        raise ValueError("Data coverage must reach the official input cutoff")
    label_coverage = input_cutoff if daily else data_through
    gap_days = 7 if daily else 0
    last_label_cutoff = label_coverage - pd.Timedelta(days=7 + gap_days)
    cutoffs = list(pd.date_range(start, last_label_cutoff, freq="D" if daily else "7D"))
    cutoffs = [cutoff for cutoff in cutoffs if cutoff < test_cutoff]
    if not cutoffs:
        raise ValueError("No complete historical label windows exist before data_through")

    print(f"Building final training rows through {data_through} UTC", flush=True)
    training_rows = build_training_table(
        args.data_dir,
        cutoffs,
        data_through=label_coverage,
        history_days=int(config["history_days"]),
        prediction_days=7,
        min_history_activities=5,
        gap_days=gap_days,
        include_inactive_users=daily,
        labels_available_through=label_coverage if daily else None,
    )
    if training_rows.empty:
        raise ValueError("No eligible historical training examples were found")
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("max-train-samples must be positive")
        if len(training_rows) > args.max_train_samples:
            training_rows = training_rows.sample(
                n=args.max_train_samples, random_state=int(config["seed"])
            ).sort_values(["Time", "UserID"], ignore_index=True)
    test_users = pd.read_csv(args.test_input)
    if list(test_users.columns) != ["UserID"]:
        raise ValueError("test_input must contain exactly one UserID column")
    if test_users["UserID"].isna().any() or test_users["UserID"].duplicated().any():
        raise ValueError("test_input contains missing or duplicate UserID values")
    if args.max_test_users is not None:
        if args.max_test_users <= 0:
            raise ValueError("max-test-users must be positive")
        test_users = test_users.head(args.max_test_users).copy()
    test_rows = pd.DataFrame({"UserID": test_users["UserID"].astype("int64"), "Time": input_cutoff})

    graph = TemporalRelationalGraph.from_data_dir(
        args.data_dir, cohort_ids=pd.Index(test_rows["UserID"]) if daily else None
    )
    use_activity_history = bool(config.get("user_activity_features", False))
    activity_history = UserActivityHistory(graph.tables, weekly_only=daily) if use_activity_history else None
    encoder = TrainFeatureEncoder.from_training(
        _encoder_fit_batches(
            graph,
            cutoff=pd.Timestamp(training_rows["Time"].max()),
            history_rows=training_rows if use_activity_history else None,
            activity_history=activity_history,
        ),
        include_user_activity_features=use_activity_history,
        **({"activity_columns": activity_history.columns} if activity_history is not None else {}),
    )
    edge_types = tuple(graph.edges)
    model = build_activity_model(
        encoder,
        edge_types,
        architecture=config.get("architecture", "hetero_mean_baseline"),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        num_layers=int(config.get("num_layers", 2)),
        attention_heads=int(config.get("attention_heads", 4)),
    )
    cache_dir.mkdir(parents=True, exist_ok=False)
    num_hops = int(config.get("num_hops", 2))
    repo_context = config.get("architecture") == REPO_CONTEXT_ARCHITECTURE
    if (repo_context or config.get("architecture") == "relgnn_atomic_routes_v1") and num_hops != 2:
        raise ValueError("RelGNN checkpoint requires the two-hop sampled neighborhood")
    fanout = default_fanout_by_relation(num_hops)
    if config["fanout"] != "10_5":
        reverse_values = tuple(map(int, config["fanout"].split("_")))[:num_hops]
        if len(reverse_values) != num_hops:
            raise ValueError("checkpoint fanout does not match its num_hops")
        fanout = {
            edge_type: ((1,) * num_hops if not edge_type[1].startswith("rev_") else reverse_values)
            for edge_type in fanout
        }
    write_split_cache(
        graph,
        encoder,
        training_rows,
        cache_dir=cache_dir,
        split_name="train",
        fanout=fanout,
        seed=int(config["seed"]),
        training=True,
        oov_dropout=0,
        sampling_strategy=config.get("sampling_strategy", "uniform"),
        num_hops=num_hops,
        repo_context=repo_context,
        activity_history=activity_history,
    )
    write_split_cache(
        graph,
        encoder,
        test_rows,
        cache_dir=cache_dir,
        split_name="test",
        fanout=fanout,
        seed=int(config["seed"]) + 70_000,
        training=False,
        oov_dropout=0,
        sampling_strategy=config.get("sampling_strategy", "uniform"),
        num_hops=num_hops,
        repo_context=repo_context,
        activity_history=activity_history,
    )
    del activity_history
    del graph
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)
    labels = training_rows.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    prevalence = labels.mean(axis=0)
    base_pos_weight = np.divide(
        1 - prevalence, prevalence, out=np.ones_like(prevalence), where=prevalence > 0
    )
    pos_weight = np.power(base_pos_weight, float(config.get("pos_weight_power", 1.0)))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
        if config["use_pos_weight"]
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_start_time = time.time()
    for epoch in range(int(config["best_epoch"])):
        model.train()
        batches = cached_batches(
            cache_dir / "train",
            encoder,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            training=True,
            oov_dropout=float(config["oov_dropout"]),
        )
        total_loss, seen = 0.0, 0
        for batch in batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch.labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch.labels)
            seen += len(batch.labels)
        print(json.dumps({"epoch": epoch + 1, "loss": total_loss / max(seen, 1)}), flush=True)

    model.eval()
    probabilities = []
    with torch.no_grad():
        batches = cached_batches(
            cache_dir / "test",
            encoder,
            batch_size=batch_size,
            seed=seed + 70_000,
            epoch=0,
            training=False,
            oov_dropout=0,
        )
        for batch in batches:
            probabilities.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    probability_array = np.concatenate(probabilities)
    threshold_array = np.asarray(checkpoint["thresholds"], dtype=float)
    predictions = (probability_array >= threshold_array[None, :]).astype("int8")
    output = test_users.copy()
    for index, label in enumerate(LABEL_COLUMNS):
        output[label] = predictions[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite prediction file: {args.output.resolve()}")
    output.to_csv(args.output, index=False)
    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "encoder": encoder.to_dict(),
            "config": config,
            "thresholds": threshold_array.tolist(),
            "test_cutoff": str(test_cutoff),
            "input_cutoff": str(input_cutoff),
            "data_through": str(data_through),
        },
        model_output,
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "train_rows": len(training_rows),
                "test_users": len(output),
                "cutoff": str(input_cutoff),
                "data_through": str(data_through),
                "elapsed_seconds": time.time() - train_start_time,
                "saved": str(args.output.resolve()),
                "cache_dir": str(cache_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    if sys.argv[1:2] in (["calibrate"], ["predict"]):
        from formal_prediction import main as formal_main

        formal_main(sys.argv[1:])
    else:
        main()
