"""Check whether validation thresholds transfer to later validation weeks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feature_encoding import TrainFeatureEncoder
from gnn_model import build_activity_model
from graph_cache import cached_batches
from labels import LABEL_COLUMNS
from train_model import _metrics, _tune_thresholds


def load_cached_weeks(validation_dir: Path) -> list[str]:
    """Read per-example cutoffs in the same order used by cached_batches."""
    manifest = json.loads((validation_dir / "manifest.json").read_text(encoding="utf-8"))
    cutoffs: list[str] = []
    for file_name in manifest["files"]:
        stored = torch.load(validation_dir / file_name, map_location="cpu", weights_only=False)
        for model_input, _label in stored:
            cutoff = pd.Timestamp(model_input.cutoff)
            cutoffs.append(str(cutoff))
    return cutoffs


def check_thresholds(checkpoint_path: Path, validation_dir: Path, output_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    encoder = TrainFeatureEncoder.from_dict(checkpoint["encoder"])
    edge_types = tuple(tuple(edge_type) for edge_type in checkpoint["edge_types"])
    model = build_activity_model(
        encoder,
        edge_types,
        architecture=config.get("architecture", "hetero_mean_baseline"),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        num_layers=int(config.get("num_layers", 2)),
        attention_heads=int(config.get("attention_heads", 4)),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    batches = cached_batches(
        validation_dir,
        encoder,
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]) + 50_000,
        training=False,
        oov_dropout=0,
        epoch=0,
    )
    with torch.no_grad():
        for batch in batches:
            labels.append(batch.labels.numpy())
            probabilities.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())

    truth = np.concatenate(labels)
    scores = np.concatenate(probabilities)
    weeks = np.asarray(load_cached_weeks(validation_dir))
    if len(weeks) != len(truth):
        raise ValueError(f"Cache cutoff count {len(weeks)} != prediction count {len(truth)}")
    if truth.shape[1] != len(LABEL_COLUMNS):
        raise ValueError(f"Expected {len(LABEL_COLUMNS)} labels, found {truth.shape[1]}")

    saved_thresholds = np.asarray(checkpoint["thresholds"], dtype=float)
    result = {
        "device": str(device),
        "weeks": sorted(set(weeks.tolist())),
        "saved_thresholds": dict(zip(LABEL_COLUMNS, saved_thresholds.tolist())),
        "saved_thresholds_by_week": {},
        "forward_threshold_check": [],
        "note": (
            "Forward checks use the same saved checkpoint. The checkpoint's epoch was selected "
            "using the full validation period, so these are threshold-stability diagnostics, "
            "not an independent final test."
        ),
    }

    ordered_weeks = result["weeks"]
    for week in ordered_weeks:
        mask = weeks == week
        result["saved_thresholds_by_week"][week] = _metrics(
            truth[mask], scores[mask], saved_thresholds
        )

    # Fit thresholds only on earlier validation weeks, then score the next week.
    for index in range(1, len(ordered_weeks)):
        prior_weeks = ordered_weeks[:index]
        test_week = ordered_weeks[index]
        fit_mask = np.isin(weeks, prior_weeks)
        test_mask = weeks == test_week
        thresholds = _tune_thresholds(truth[fit_mask], scores[fit_mask])
        result["forward_threshold_check"].append(
            {
                "fit_weeks": prior_weeks,
                "test_week": test_week,
                "thresholds": dict(zip(LABEL_COLUMNS, thresholds.tolist())),
                "metrics": _metrics(truth[test_mask], scores[test_mask], thresholds),
                "metrics_at_0_5": _metrics(
                    truth[test_mask], scores[test_mask], np.full(len(LABEL_COLUMNS), 0.5)
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_thresholds(args.checkpoint, args.validation_cache, args.output)


if __name__ == "__main__":
    main()
