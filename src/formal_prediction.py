"""Calibrate and predict with a frozen formal training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from activity_features import UserActivityHistory
from data import as_utc_naive_timestamp
from feature_encoding import TrainFeatureEncoder
from gnn_model import build_activity_model
from labels import LABEL_COLUMNS
from sample_examples import batch_iterator, example_inputs
from sampling_config import default_fanout_by_relation
from temporal_subgraph import REPO_CONTEXT_ARCHITECTURE, TemporalRelationalGraph
from train_model import _daily_validation_rows, _metrics, _tune_thresholds


@torch.inference_mode()
def predict_rows(data_dir, rows, checkpoint, device, *, official=False):
    config = checkpoint["config"]
    encoder = TrainFeatureEncoder.from_dict(checkpoint["encoder"])
    graph = TemporalRelationalGraph.from_data_dir(
        data_dir, cohort_ids=pd.Index(rows["UserID"]) if official else None,
    )
    edge_types = tuple(tuple(edge) for edge in checkpoint["edge_types"])
    model = build_activity_model(
        encoder, edge_types,
        architecture=config["architecture"],
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
        attention_heads=config["attention_heads"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    history = (
        UserActivityHistory(graph.tables, weekly_only=config["observed_cohort_daily"])
        if config["user_activity_features"] else None
    )
    num_hops = config["num_hops"]
    fanout = default_fanout_by_relation(num_hops)
    if config["fanout"] != "10_5":
        reverse = tuple(map(int, config["fanout"].split("_")[:num_hops]))
        if len(reverse) != num_hops:
            raise ValueError("Checkpoint fanout does not match its hop count")
        fanout = {edge: reverse if edge[1].startswith("rev_") else caps for edge, caps in fanout.items()}
    examples = example_inputs(
        graph, encoder, rows, fanout=fanout,
        seed=config["seed"] + (70_000 if official else 50_000),
        epoch=0, training=False, oov_dropout=0,
        sampling_strategy=config["sampling_strategy"],
        num_hops=num_hops, repo_context=config["architecture"] == REPO_CONTEXT_ARCHITECTURE,
        activity_history=history,
    )
    batches = batch_iterator(examples, config["batch_size"])
    return np.concatenate([
        torch.sigmoid(model(batch.to(device))).cpu().numpy() for batch in batches
    ])


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate")
    predict = commands.add_parser("predict")
    for command in (calibrate, predict):
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--data-through", required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--cpu", action="store_true")
    calibrate.add_argument("--calibration-cutoff", default="2026-11-13T16:00:00Z")
    predict.add_argument("--answer-start", default="2026-11-21T16:00:00Z")
    predict.add_argument("--input-cutoff", required=True)
    predict.add_argument("--test-input", type=Path, required=True)
    predict.add_argument("--thresholds", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    if checkpoint["labels"] != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint label order does not match the task")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    coverage = as_utc_naive_timestamp(args.data_through)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "calibrate":
        cutoff = as_utc_naive_timestamp(args.calibration_cutoff)
        selection_end = max(map(as_utc_naive_timestamp, checkpoint["validation_cutoffs"]))
        selection_end += pd.Timedelta(days=7 + checkpoint["config"]["gap_days"])
        if cutoff < selection_end:
            raise ValueError("Calibration must start after the checkpoint selection answers end")
        rows = _daily_validation_rows(args.data_dir, [cutoff], coverage, 0)
        if rows.empty:
            raise ValueError("No historical users are eligible for calibration")
        probabilities = predict_rows(args.data_dir, rows, checkpoint, device)
        labels = rows.loc[:, list(LABEL_COLUMNS)].to_numpy()
        thresholds = _tune_thresholds(labels, probabilities)
        result = {
            "checkpoint_sha256": checkpoint_hash,
            "calibration_cutoff": str(cutoff),
            "calibration_end": str(cutoff + pd.Timedelta(days=7)),
            "labels": list(LABEL_COLUMNS),
            "thresholds": thresholds.tolist(),
            "users": len(rows),
            "calibration_metrics": _metrics(labels, probabilities, thresholds),
        }
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
    else:
        cutoff = as_utc_naive_timestamp(args.input_cutoff)
        answer_start = as_utc_naive_timestamp(args.answer_start)
        if not cutoff <= coverage <= answer_start:
            raise ValueError("Require input cutoff <= data coverage <= official answer start")
        calibration = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if calibration["checkpoint_sha256"] != checkpoint_hash or calibration["labels"] != list(LABEL_COLUMNS):
            raise ValueError("Thresholds do not belong to this checkpoint")
        if as_utc_naive_timestamp(calibration["calibration_end"]) > cutoff:
            raise ValueError("Calibration answers must be available by the input cutoff")
        thresholds = np.asarray(calibration["thresholds"], dtype=float)
        if thresholds.shape != (len(LABEL_COLUMNS),) or not np.all((0 <= thresholds) & (thresholds <= 1)):
            raise ValueError("Threshold file must contain one valid threshold per label")
        users = pd.read_csv(args.test_input, dtype={"UserID": str})
        if (list(users.columns) != ["UserID"] or users.empty
                or not users["UserID"].str.fullmatch(r"[0-9]+").fillna(False).all()):
            raise ValueError("test_input must contain a nonempty UserID column of integers")
        users["UserID"] = users["UserID"].astype("int64")
        if users["UserID"].duplicated().any():
            raise ValueError("test_input contains duplicate UserIDs")
        rows = users.assign(Time=cutoff)
        probabilities = predict_rows(args.data_dir, rows, checkpoint, device, official=True)
        users[list(LABEL_COLUMNS)] = (probabilities >= thresholds).astype("int8")
        users.to_csv(args.output, index=False, mode="x")
    print(f"Saved {args.output} at cutoff {cutoff} UTC", flush=True)
