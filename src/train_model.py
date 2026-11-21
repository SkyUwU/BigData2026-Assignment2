"""Train and validate the relational activity GNN on chronological samples."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from activity_features import USER_ACTIVITY_FEATURE_COLUMNS, UserActivityHistory, user_activity_feature_frame
from data import ROW_TIME_COLUMN, as_utc_naive_timestamp
from feature_encoding import TrainFeatureEncoder
from feature_policy import FEATURE_COLUMNS_BY_TYPE, feature_columns
from gnn_model import RelGNNActivityGNN, build_activity_model
from graph_cache import cached_batches, write_split_cache
from labels import LABEL_COLUMNS
from model_input import SubgraphModelInput, TIME_DELTA_DAYS_COLUMN
from sampling_config import default_fanout_by_relation
from temporal_split import split_training_table
from temporal_subgraph import REPO_CONTEXT_ARCHITECTURE, TemporalRelationalGraph
from training_table import build_training_table
from sample_examples import batch_iterator


def weekly_cutoffs(start: str, end_exclusive: str) -> list[pd.Timestamp]:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end_exclusive)
    if start_time.tz is not None:
        start_time = start_time.tz_convert("UTC").tz_localize(None)
    if end_time.tz is not None:
        end_time = end_time.tz_convert("UTC").tz_localize(None)
    if start_time >= end_time:
        raise ValueError("cutoff range must be non-empty")
    return list(pd.date_range(start_time, end_time - pd.Timedelta(days=7), freq="7D"))


def _encoder_fit_batches(
    graph: TemporalRelationalGraph,
    *,
    cutoff: pd.Timestamp,
    history_rows: pd.DataFrame | None = None,
    activity_history: UserActivityHistory | None = None,
    chunk_size: int = 50_000,
):
    """Stream training-visible table rows into the train-only encoder."""
    for node_type, groups in FEATURE_COLUMNS_BY_TYPE.items():
        table = graph.tables[node_type]
        time_column = ROW_TIME_COLUMN[node_type]
        row_times = pd.to_datetime(table[time_column], errors="coerce", utc=True).dt.tz_localize(None)
        available_positions = np.flatnonzero(row_times.lt(cutoff).to_numpy())
        for start in range(0, len(available_positions), chunk_size):
            selected = table.iloc[available_positions[start : start + chunk_size]]
            selected_times = row_times.iloc[available_positions[start : start + chunk_size]]
            raw = selected.loc[:, list(feature_columns(node_type))].reset_index(drop=True).copy()
            if node_type == "Users":
                for column in (activity_history.columns if activity_history else USER_ACTIVITY_FEATURE_COLUMNS):
                    raw[column] = np.nan
            raw[TIME_DELTA_DAYS_COLUMN] = (
                (cutoff - selected_times).dt.total_seconds().to_numpy() / 86_400.0
            )
            yield SubgraphModelInput(
                user_id=0,
                cutoff=cutoff,
                node_features={node_type: raw},
                edge_index={},
                seed_node_index=0,
            )

    if history_rows is not None:
        if activity_history is None:
            raise ValueError("activity_history is required when fitting target history features")
        users = graph.tables["Users"]
        user_mapping = users.set_index("github_user_id")["user_id"]
        internal_ids = history_rows["UserID"].map(user_mapping)
        if internal_ids.isna().any():
            raise ValueError("A target user in the training rows is missing from Users.parquet")
        user_positions = users.set_index("user_id").index
        first_seen = users.set_index("user_id")[ROW_TIME_COLUMN["Users"]]
        timestamps = pd.to_datetime(first_seen, errors="coerce", utc=True).dt.tz_localize(None)
        user_time_by_id = pd.Series(timestamps.to_numpy(), index=user_positions)
        target_cutoffs = pd.to_datetime(history_rows["Time"], errors="coerce", utc=True).dt.tz_localize(None)
        target_first_seen = internal_ids.map(user_time_by_id)
        if target_cutoffs.isna().any():
            raise ValueError("Cannot derive training-user availability time")
        activity_values = user_activity_feature_frame(
            activity_history,
            internal_ids.astype("int64").tolist(),
            target_cutoffs.tolist(),
        )
        activity_values[TIME_DELTA_DAYS_COLUMN] = (
            (target_cutoffs - target_first_seen).dt.total_seconds().to_numpy() / 86_400.0
        )
        unseen = ~target_first_seen.lt(target_cutoffs)
        activity_values.loc[unseen.to_numpy(), list(activity_history.columns)] = 0.0
        activity_values.loc[unseen.to_numpy(), TIME_DELTA_DAYS_COLUMN] = np.nan
        for start in range(0, len(activity_values), chunk_size):
            raw = activity_values.iloc[start : start + chunk_size].reset_index(drop=True)
            yield SubgraphModelInput(
                user_id=0,
                cutoff=cutoff,
                node_features={"Users": raw},
                edge_index={},
                seed_node_index=0,
            )


def _metrics(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray, *,
    label_names=LABEL_COLUMNS,
) -> dict:
    predicted = probabilities >= thresholds[None, :]
    truth = labels.astype(bool)
    tp = (predicted & truth).sum(axis=0)
    fp = (predicted & ~truth).sum(axis=0)
    fn = (~predicted & truth).sum(axis=0)
    f1_denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        f1_denominator,
        out=np.zeros_like(tp, dtype=float),
        where=f1_denominator > 0,
    )
    # Match the assignment evaluator: an event with no true or predicted
    # positives has F1=1 rather than 0.
    f1[f1_denominator == 0] = 1.0
    micro_denominator = 2 * tp.sum() + fp.sum() + fn.sum()
    micro_f1 = float(2 * tp.sum() / micro_denominator) if micro_denominator else 1.0
    return {
        "score": float((f1.mean() + micro_f1) / 2),
        "macro_f1": float(f1.mean()),
        "micro_f1": micro_f1,
        "per_label": {
            label: {
                "f1": float(f1[i]),
                "precision": float(tp[i] / (tp[i] + fp[i])) if tp[i] + fp[i] else 0.0,
                "recall": float(tp[i] / (tp[i] + fn[i])) if tp[i] + fn[i] else 0.0,
                "prevalence": float(truth[:, i].mean()) if len(truth) else 0.0,
            }
            for i, label in enumerate(label_names)
        },
    }


def _tune_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Coordinate-search a small fixed grid once after model selection."""
    candidates = np.arange(0.10, 0.901, 0.05)
    thresholds = np.full(len(LABEL_COLUMNS), 0.5, dtype=float)
    for _ in range(2):
        changed = False
        for label_index in range(len(LABEL_COLUMNS)):
            best_score = _metrics(labels, probabilities, thresholds)["score"]
            best_threshold = thresholds[label_index]
            for candidate in candidates:
                proposed = thresholds.copy()
                proposed[label_index] = candidate
                score = _metrics(labels, probabilities, proposed)["score"]
                if score > best_score + 1e-12:
                    best_score, best_threshold = score, candidate
            if best_threshold != thresholds[label_index]:
                thresholds[label_index] = best_threshold
                changed = True
        if not changed:
            break
    return thresholds


def _forward_threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    times: pd.Series,
) -> tuple[dict, list[dict]]:
    """Score later validation weeks using thresholds fit only on earlier weeks."""
    time_values = pd.to_datetime(times, errors="coerce", utc=True).dt.tz_localize(None)
    if time_values.isna().any() or len(time_values) != len(labels):
        raise ValueError("Validation timestamps must align with labels and probabilities")
    weeks = sorted(time_values.unique())
    if len(weeks) < 2:
        raise ValueError("Threshold-aware checkpoint selection needs at least two validation weeks")

    scored_predictions = np.zeros_like(labels, dtype=bool)
    scored_mask = np.zeros(len(labels), dtype=bool)
    week_records: list[dict] = []
    for week_index in range(1, len(weeks)):
        fit_mask = time_values.isin(weeks[:week_index]).to_numpy()
        score_mask = time_values.eq(weeks[week_index]).to_numpy()
        thresholds = _tune_thresholds(labels[fit_mask], probabilities[fit_mask])
        scored_predictions[score_mask] = probabilities[score_mask] >= thresholds[None, :]
        scored_mask |= score_mask
        fold_metrics = _metrics(
            labels[score_mask],
            probabilities[score_mask],
            thresholds,
        )
        week_records.append({
            "cutoff": str(pd.Timestamp(weeks[week_index])),
            "threshold_fit_cutoffs": [str(pd.Timestamp(value)) for value in weeks[:week_index]],
            "thresholds": dict(zip(LABEL_COLUMNS, map(float, thresholds))),
            "metrics": fold_metrics,
        })
    aggregate = _metrics(
        labels[scored_mask],
        scored_predictions[scored_mask].astype("float32"),
        np.full(len(LABEL_COLUMNS), 0.5),
    )
    return aggregate, week_records


def _limit_rows(rows: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    """Take a reproducible sample across the complete temporal split."""
    if maximum is None or len(rows) <= maximum:
        return rows.copy()
    return rows.sample(n=maximum, random_state=seed).sort_values(
        ["Time", "UserID"], ignore_index=True
    )


@torch.no_grad()
def _predict_probabilities(model, cache_dir, encoder, *, seed, epoch, batch_size, device):
    model.eval()
    outputs = []
    batches = cached_batches(
        cache_dir,
        encoder,
        batch_size=batch_size,
        seed=seed,
        training=False,
        oov_dropout=0,
        epoch=epoch,
    )
    for batch in batches:
        outputs.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, len(LABEL_COLUMNS)))


def train_validation(args: argparse.Namespace) -> dict:
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    cache_dir = (args.cache_dir or output_dir.with_name(output_dir.name + ".cache")).resolve()
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing graph cache: {cache_dir}")
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("batch size, epochs, and patience must be positive")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("max-train-samples must be positive")
    if args.max_validation_samples is not None and args.max_validation_samples <= 0:
        raise ValueError("max-validation-samples must be positive")
    if args.cache_chunk_size <= 0:
        raise ValueError("cache-chunk-size must be positive")
    repo_context = args.architecture == REPO_CONTEXT_ARCHITECTURE
    if repo_context or args.architecture == RelGNNActivityGNN.architecture:
        if args.num_hops != 2 or args.num_layers not in (1, 2):
            raise ValueError(
                "RelGNN requires --num-hops 2 and --num-layers 1 or 2"
            )
        if repo_context and args.num_layers != 1:
            raise ValueError("Repo-context experiments require --num-layers 1")
        if args.hidden_size <= 0 or args.attention_heads <= 0:
            raise ValueError("hidden-size and attention-heads must be positive")
        if args.hidden_size % args.attention_heads:
            raise ValueError("hidden-size must be divisible by attention-heads")

    daily = bool(args.validation_cutoff)
    gap_days = getattr(args, "gap_days", 7) if daily else 0
    if daily:
        data_through = as_utc_naive_timestamp(args.data_through)
        validation_cutoffs = [
            as_utc_naive_timestamp(value) for value in args.validation_cutoff
        ]
        if len(validation_cutoffs) not in (1, 2):
            raise ValueError("Provide one or two --validation-cutoff values")
        first_validation = validation_cutoffs[0]
        label_delay = pd.Timedelta(days=7 + gap_days)
        if len(validation_cutoffs) == 2 and first_validation + label_delay > validation_cutoffs[1]:
            raise ValueError("The first validation answers must be available by the second input cutoff")
        train_cutoffs = list(pd.date_range(
            as_utc_naive_timestamp(args.train_start), first_validation - label_delay, freq="D"
        ))
        train_rows = build_training_table(
            args.data_dir, train_cutoffs, data_through=data_through, include_inactive_users=True,
            gap_days=gap_days, labels_available_through=first_validation,
        )
        validation_rows = _daily_validation_rows(args.data_dir, validation_cutoffs, data_through, gap_days)
        reference_dir = getattr(args, "validation_reference_data_dir", None)
        if reference_dir is not None:
            reference_rows = _daily_validation_rows(reference_dir, validation_cutoffs, data_through, gap_days)
            validation_rows = _lock_validation_rows(validation_rows, reference_rows)
            for cutoff, count in validation_rows.groupby("Time").size().items():
                print(f"Locked validation cutoff {cutoff}: {count} users, labels matched", flush=True)
    else:
        train_cutoffs = weekly_cutoffs(args.train_start, args.validation_start)
        validation_cutoffs = weekly_cutoffs(args.validation_start, args.validation_end)
        rows = build_training_table(
            args.data_dir, train_cutoffs + validation_cutoffs, data_through=args.data_through,
            history_days=args.history_days,
        )
        train_rows, validation_rows, excluded_rows = split_training_table(
            rows, validation_start=args.validation_start, validation_end=args.validation_end,
            prediction_days=7,
        )
        if not excluded_rows.empty:
            raise ValueError(f"Candidate cutoffs unexpectedly produced {len(excluded_rows)} excluded samples")
    train_rows = _limit_rows(train_rows, args.max_train_samples, args.seed)
    single_week = daily and len(validation_cutoffs) == 1
    selection_skip = set(getattr(args, "selection_skip", ()))
    if selection_skip and (not single_week or selection_skip == set(LABEL_COLUMNS)):
        raise ValueError("Selection exclusions require a single validation week and at least one scored label")
    selection_indices = [i for i, label in enumerate(LABEL_COLUMNS) if label not in selection_skip]
    selection_labels = [LABEL_COLUMNS[i] for i in selection_indices]
    if daily and not single_week and args.max_validation_samples is not None:
        if args.max_validation_samples < 2:
            raise ValueError("The validation sample cap must allow both weeks")
        validation_rows = pd.concat([
            _limit_rows(week, args.max_validation_samples // 2, args.seed + 1 + index)
            for index, (_, week) in enumerate(validation_rows.groupby("Time", sort=True))
        ], ignore_index=True)
    else:
        validation_rows = _limit_rows(validation_rows, args.max_validation_samples, args.seed + 1)
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Both training and validation splits must contain rows")
    if daily and validation_rows["Time"].nunique() != len(validation_cutoffs):
        raise ValueError("Each validation cutoff must have cohort users already visible")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Loading temporal relational graph on {device}; this may require substantial RAM.", flush=True)
    start_time = time.time()
    graph = TemporalRelationalGraph.from_data_dir(args.data_dir)
    activity_history = UserActivityHistory(graph.tables, weekly_only=daily)
    copy_last_week_metrics = None
    if daily:
        users = graph.tables["Users"].set_index("github_user_id")
        scoring_rows = validation_rows.loc[validation_rows["Time"].eq(validation_cutoffs[-1])]
        scoring_users = users.loc[scoring_rows["UserID"]]
        counts = user_activity_feature_frame(
            activity_history,
            scoring_users["user_id"].tolist(),
            scoring_rows["Time"].tolist(),
        )
        copy_last_week_metrics = _metrics(
            scoring_rows.loc[:, list(LABEL_COLUMNS)].to_numpy(),
            counts.to_numpy() > 0,
            np.full(len(LABEL_COLUMNS), 0.5),
        )
    cutoff = pd.Timestamp(train_rows["Time"].max())
    encoder = TrainFeatureEncoder.from_training(
        _encoder_fit_batches(
            graph,
            cutoff=cutoff,
            history_rows=train_rows,
            activity_history=activity_history,
            chunk_size=args.encoder_chunk_size,
        ),
        include_user_activity_features=True,
        activity_columns=activity_history.columns,
    )
    edge_types = tuple(graph.edges)
    model = build_activity_model(
        encoder, edge_types, architecture=args.architecture,
        hidden_size=args.hidden_size, dropout=args.dropout,
        num_layers=args.num_layers, attention_heads=args.attention_heads,
    ).to(device)
    labels = train_rows.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    prevalence = labels.mean(axis=0)
    base_pos_weight = np.divide(
        1 - prevalence, prevalence, out=np.ones_like(prevalence), where=prevalence > 0
    )
    pos_weight = np.power(base_pos_weight, args.pos_weight_power)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
        if args.use_pos_weight
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    fanout = default_fanout_by_relation(args.num_hops)
    if args.fanout != "10_5":
        reverse_values = tuple(map(int, args.fanout.split("_")))[:args.num_hops]
        if len(reverse_values) != args.num_hops:
            raise ValueError("fanout must specify one reverse fanout per hop")
        fanout = {
            edge_type: ((1,) * args.num_hops if not edge_type[1].startswith("rev_") else reverse_values)
            for edge_type in fanout
        }

    cache_dir.mkdir(parents=True, exist_ok=False)
    train_cache = write_split_cache(
        graph,
        encoder,
        train_rows,
        cache_dir=cache_dir,
        split_name="train",
        fanout=fanout,
        seed=args.seed,
        training=True,
        oov_dropout=0,
        sampling_strategy=args.sampling_strategy,
        num_hops=args.num_hops,
        repo_context=repo_context,
        chunk_size=args.cache_chunk_size,
        activity_history=activity_history,
    )
    validation_cache = write_split_cache(
        graph,
        encoder,
        validation_rows,
        cache_dir=cache_dir,
        split_name="validation",
        fanout=fanout,
        seed=args.seed + 50_000,
        training=False,
        oov_dropout=0,
        sampling_strategy=args.sampling_strategy,
        num_hops=args.num_hops,
        repo_context=repo_context,
        chunk_size=args.cache_chunk_size,
        activity_history=activity_history,
    )
    del activity_history
    del graph
    gc.collect()

    validation_labels = validation_rows.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    train_rate_thresholds = np.full(len(LABEL_COLUMNS), 0.5)
    history = []
    best_score, best_epoch, best_state, stale_epochs = -1.0, 0, None, 0
    best_forward_weeks: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        running_loss, seen = 0.0, 0
        batches = cached_batches(
            cache_dir / "train",
            encoder,
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
            training=True,
            oov_dropout=args.oov_dropout,
        )
        for batch in batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, batch.labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch.labels)
            seen += len(batch.labels)
        probabilities = _predict_probabilities(
            model,
            cache_dir / "validation",
            encoder,
            seed=args.seed + 50_000,
            epoch=0,
            batch_size=args.batch_size,
            device=device,
        )
        score_at_half = _metrics(validation_labels, probabilities, train_rate_thresholds)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(seen, 1),
            "validation_score_at_0_5": score_at_half["score"],
            "validation_macro_f1_at_0_5": score_at_half["macro_f1"],
            "validation_micro_f1_at_0_5": score_at_half["micro_f1"],
        }
        selection_score, forward_weeks = score_at_half["score"], []
        if single_week:
            if selection_skip:
                selection_score = _metrics(
                    validation_labels[:, selection_indices],
                    probabilities[:, selection_indices],
                    train_rate_thresholds[selection_indices],
                    label_names=selection_labels,
                )["score"]
            epoch_record["validation_selection_score"] = selection_score
            epoch_record["validation_push_f1_at_0_5"] = score_at_half["per_label"]["push"]["f1"]
        else:
            forward_metrics, forward_weeks = _forward_threshold_metrics(
                validation_labels, probabilities, validation_rows["Time"],
            )
            selection_score = forward_metrics["score"]
            for key in ("score", "macro_f1", "micro_f1"):
                epoch_record[f"validation_forward_{key}"] = forward_metrics[key]
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if selection_score > best_score:
            best_score, best_epoch = selection_score, epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale_epochs = 0
            best_forward_weeks = forward_weeks
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    validation_probabilities = _predict_probabilities(
        model,
        cache_dir / "validation",
        encoder,
        seed=args.seed + 50_000,
        epoch=0,
        batch_size=args.batch_size,
        device=device,
    )
    tuned_thresholds = (
        train_rate_thresholds if single_week
        else _tune_thresholds(validation_labels, validation_probabilities)
    )
    final_metrics = _metrics(validation_labels, validation_probabilities, tuned_thresholds)
    selected_fixed_threshold_metrics = _metrics(
        validation_labels,
        validation_probabilities,
        np.full(len(LABEL_COLUMNS), 0.5),
    )
    per_week = {}
    for cutoff_value, week_rows in validation_rows.groupby("Time", sort=True):
        mask = validation_rows["Time"].eq(cutoff_value).to_numpy()
        per_week[str(cutoff_value)] = _metrics(
            validation_labels[mask], validation_probabilities[mask], tuned_thresholds
        )
    baseline_probabilities = np.broadcast_to(prevalence, validation_labels.shape)
    baseline_metrics = _metrics(validation_labels, baseline_probabilities, train_rate_thresholds)

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "model_state": best_state,
        "encoder": encoder.to_dict(),
        "edge_types": [list(edge_type) for edge_type in edge_types],
        "config": {
            "architecture": args.architecture,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "attention_heads": args.attention_heads,
            "num_hops": args.num_hops,
            "dropout": args.dropout,
            "fanout": args.fanout,
            "sampling_strategy": args.sampling_strategy,
            "history_days": args.history_days,
            "observed_cohort_daily": daily,
            "gap_days": gap_days,
            "train_start": str(train_cutoffs[0]),
            "batch_size": args.batch_size,
            "use_pos_weight": args.use_pos_weight,
            "pos_weight_power": args.pos_weight_power,
            "pos_weight_by_label": dict(zip(LABEL_COLUMNS, map(float, pos_weight))),
            "user_activity_features": True,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "oov_dropout": args.oov_dropout,
            "best_epoch": best_epoch,
            "checkpoint_selection": (
                "validation_score_at_0_5" if single_week else "forward_validation_threshold_score"
            ),
            "selection_labels": selection_labels,
            "selection_skipped_labels": sorted(selection_skip),
            "seed": args.seed,
        },
        "thresholds": tuned_thresholds.tolist(),
        "validation_cutoffs": [str(value) for value in validation_cutoffs],
        "labels": list(LABEL_COLUMNS),
    }
    torch.save(checkpoint, output_dir / "validation_checkpoint.pt")
    result = {
        "run_config": checkpoint["config"],
        "device": str(device),
        "train_cutoffs": [str(value) for value in train_cutoffs],
        "validation_cutoffs": [str(value) for value in validation_cutoffs],
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "excluded_rows": 0,
        "best_epoch": best_epoch,
        "use_pos_weight": args.use_pos_weight,
        "pos_weight_power": args.pos_weight_power,
        "pos_weight_by_label": dict(zip(LABEL_COLUMNS, map(float, pos_weight))),
        "best_validation_selection_score": best_score,
        "best_validation_forward_score": None if single_week else best_score,
        "best_validation_forward_weeks": best_forward_weeks,
        "selected_checkpoint_validation_metrics_at_0_5": selected_fixed_threshold_metrics,
        "tuned_thresholds": None if single_week else dict(zip(LABEL_COLUMNS, map(float, tuned_thresholds))),
        "validation_metrics_tuned": None if single_week else final_metrics,
        "validation_metrics_tuned_note": (
            "Single-week selection uses 0.5; calibrate on a separate week."
            if single_week else "Thresholds are fit on the full validation period; this score is for diagnostics, "
            "not an independent estimate of test performance."
        ),
        "validation_metrics_by_week": per_week,
        "train_rate_baseline_at_0_5": baseline_metrics,
        "copy_last_week_baseline": copy_last_week_metrics,
        "history": history,
        "elapsed_seconds": time.time() - start_time,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "cache_dir": str(cache_dir),
        "train_cache": train_cache,
        "validation_cache": validation_cache,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def _daily_validation_rows(data_dir, cutoffs, data_through, gap_days):
    return pd.concat([
        build_training_table(
            data_dir, [cutoff], data_through=data_through, include_inactive_users=True,
            gap_days=gap_days, labels_available_through=available_through,
        )
        for cutoff, available_through in zip(cutoffs, [*cutoffs[1:], data_through])
    ], ignore_index=True)


def _lock_validation_rows(validation_rows, reference_rows):
    keys = ["Time", "UserID"]
    reference = reference_rows.set_index(keys, verify_integrity=True)
    expanded = validation_rows.set_index(keys, verify_integrity=True)
    missing = reference.index.difference(expanded.index)
    if len(missing):
        raise ValueError(f"Expanded data is missing {len(missing)} reference validation rows")
    locked = expanded.loc[reference.index]
    changed = locked[list(LABEL_COLUMNS)].ne(reference[list(LABEL_COLUMNS)]).any(axis=1)
    if changed.any():
        raise ValueError(f"Validation labels differ for {changed.sum()} reference rows")
    return locked.reset_index()[validation_rows.columns]


def main() -> None:
    preset = argparse.ArgumentParser(add_help=False)
    preset.add_argument("--formal", action="store_true", help="Use the formal submission training defaults")
    preset_args, _ = preset.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[preset])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-through", required=not preset_args.formal)
    parser.add_argument("--train-start", default="2026-04-28T16:00:00Z")
    parser.add_argument("--validation-start", default="2026-10-06T16:00:00Z")
    parser.add_argument("--validation-end", default="2026-10-27T16:00:00Z")
    parser.add_argument(
        "--validation-cutoff", action="append", default=[],
        help="Enable daily training; supply one cutoff for fixed-0.5 selection or two for forward selection.",
    )
    parser.add_argument("--gap-days", type=int, help="Days from input cutoff to the label window (daily default: 7).")
    parser.add_argument(
        "--selection-skip", nargs="*", choices=LABEL_COLUMNS, default=(),
        help="Omit these labels only from single-week checkpoint selection.",
    )
    parser.add_argument(
        "--validation-reference-data-dir", type=Path,
        help="Original parquet tables used to lock the daily validation users and verify their labels.",
    )
    parser.add_argument("--history-days", type=int, default=90)
    parser.add_argument("--fanout", choices=("10_5", "5_2", "10_5_2"), default="10_5")
    parser.add_argument("--num-hops", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--sampling-strategy", choices=("uniform", "latest"), default="uniform",
        help="Choose uniformly among past neighbors or prioritize the newest past neighbors.",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--architecture",
        choices=(RelGNNActivityGNN.architecture, REPO_CONTEXT_ARCHITECTURE, "hetero_mean_baseline"),
        default=RelGNNActivityGNN.architecture,
    )
    parser.add_argument("--num-layers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--oov-dropout", type=float, default=0.05)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument(
        "--pos-weight-power", type=float, default=1.0,
        help="Exponent applied to each training-label negative/positive ratio (1.0 keeps the full ratio).",
    )
    parser.add_argument("--encoder-chunk-size", type=int, default=50_000)
    parser.add_argument("--cache-chunk-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    if preset_args.formal:
        parser.set_defaults(
            data_through="2026-11-13T16:00:00Z",
            train_start="2026-09-05T16:00:00Z",
            validation_cutoff=None,
            gap_days=0,
            num_layers=2,
            sampling_strategy="latest",
            batch_size=32,
            epochs=10,
            selection_skip=("push",),
            use_pos_weight=True,
            pos_weight_power=0.5,
        )
    args = parser.parse_args()
    if args.formal and args.validation_cutoff is None:
        args.validation_cutoff = ["2026-11-06T16:00:00Z"]
    if args.gap_days is not None and (args.gap_days < 0 or not args.validation_cutoff):
        parser.error("--gap-days requires daily validation and a non-negative value")
    if args.gap_days is None:
        args.gap_days = 7
    if args.pos_weight_power <= 0:
        parser.error("--pos-weight-power must be greater than zero")
    if args.validation_reference_data_dir is not None and (
        not args.validation_cutoff or args.max_validation_samples is not None
    ):
        parser.error("--validation-reference-data-dir requires daily validation without a sample cap")
    train_validation(args)


if __name__ == "__main__":
    main()
