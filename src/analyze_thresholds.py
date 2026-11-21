"""Analyze higher validation threshold candidates from a saved checkpoint/cache."""

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
from train_model import _metrics


def _candidate_thresholds(maximum: float) -> np.ndarray:
    base = np.arange(0.10, 0.9001, 0.05)
    upper = np.array([value for value in (0.925, 0.95, 0.975, 0.99) if value <= maximum + 1e-12])
    values = np.unique(np.concatenate((base, upper)))
    return values[values <= maximum + 1e-12]


def _tune(labels: np.ndarray, probabilities: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    thresholds = np.full(len(LABEL_COLUMNS), 0.5, dtype=float)
    for _ in range(2):
        changed = False
        for index in range(len(LABEL_COLUMNS)):
            best_score = _metrics(labels, probabilities, thresholds)["score"]
            best_threshold = thresholds[index]
            for candidate in candidates:
                proposed = thresholds.copy()
                proposed[index] = candidate
                score = _metrics(labels, probabilities, proposed)["score"]
                if score > best_score + 1e-12:
                    best_score, best_threshold = score, float(candidate)
            if best_threshold != thresholds[index]:
                thresholds[index] = best_threshold
                changed = True
        if not changed:
            break
    return thresholds


def _forward_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    times: pd.Series,
    maximum: float,
) -> dict:
    candidates = _candidate_thresholds(maximum)
    weeks = sorted(times.unique())
    if len(weeks) < 2:
        raise ValueError("At least two validation weeks are required for forward analysis")

    fold_reports = []
    prediction_blocks = []
    label_blocks = []
    for index in range(1, len(weeks)):
        fit_mask = times.isin(weeks[:index]).to_numpy()
        score_mask = times.eq(weeks[index]).to_numpy()
        thresholds = _tune(labels[fit_mask], probabilities[fit_mask], candidates)
        fold_metrics = _metrics(labels[score_mask], probabilities[score_mask], thresholds)
        fold_reports.append({
            "scored_week": str(pd.Timestamp(weeks[index])),
            "threshold_fit_weeks": [str(pd.Timestamp(value)) for value in weeks[:index]],
            "thresholds": dict(zip(LABEL_COLUMNS, map(float, thresholds))),
            "metrics": fold_metrics,
            "upper_boundary_labels": [
                label for label, value in zip(LABEL_COLUMNS, thresholds)
                if np.isclose(value, candidates[-1])
            ],
        })
        prediction_blocks.append(probabilities[score_mask] >= thresholds[None, :])
        label_blocks.append(labels[score_mask])

    predicted = np.concatenate(prediction_blocks, axis=0)
    truth = np.concatenate(label_blocks, axis=0)
    aggregate = _metrics(truth, predicted.astype(np.float32), np.full(len(LABEL_COLUMNS), 0.5))
    return {
        "candidate_maximum": maximum,
        "candidate_values": candidates.tolist(),
        "forward_score": aggregate["score"],
        "forward_macro_f1": aggregate["macro_f1"],
        "forward_micro_f1": aggregate["micro_f1"],
        "folds": fold_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output.resolve()}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("labels") != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint label order does not match the task")
    encoder = TrainFeatureEncoder.from_dict(checkpoint["encoder"])
    config = checkpoint["config"]
    manifest = json.loads((args.validation_cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("split") != "validation":
        raise ValueError("The supplied cache is not a validation split")

    batch_size = args.batch_size or int(config["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_activity_model(
        encoder,
        tuple(tuple(edge) for edge in checkpoint["edge_types"]),
        architecture=config.get("architecture", "hetero_mean_baseline"),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        num_layers=int(config.get("num_layers", 2)),
        attention_heads=int(config.get("attention_heads", 4)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    probabilities, labels = [], []
    with torch.no_grad():
        for batch in cached_batches(
            args.validation_cache,
            encoder,
            batch_size=batch_size,
            seed=int(config["seed"]) + 50_000,
            epoch=0,
            training=False,
            oov_dropout=0,
        ):
            batch = batch.to(device)
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
            labels.append(batch.labels.cpu().numpy())
    probabilities_array = np.concatenate(probabilities)
    labels_array = np.concatenate(labels)

    cutoffs = []
    for filename in manifest["files"]:
        chunk = torch.load(args.validation_cache / filename, map_location="cpu", weights_only=False)
        cutoffs.extend(pd.Timestamp(model_input.cutoff) for model_input, _ in chunk)
    if len(cutoffs) != len(labels_array) or len(labels_array) != int(manifest["examples"]):
        raise ValueError("Cache examples, cutoffs, and predictions are not aligned")
    times = pd.Series(pd.to_datetime(cutoffs, utc=True).tz_localize(None))

    reports = [_forward_report(labels_array, probabilities_array, times, maximum)
               for maximum in (0.90, 0.95, 0.99)]
    full_validation = []
    for maximum in (0.90, 0.95, 0.99):
        candidates = _candidate_thresholds(maximum)
        thresholds = _tune(labels_array, probabilities_array, candidates)
        full_validation.append({
            "candidate_maximum": maximum,
            "thresholds": dict(zip(LABEL_COLUMNS, map(float, thresholds))),
            "metrics_diagnostic_only": _metrics(labels_array, probabilities_array, thresholds),
        })

    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_cache": str(args.validation_cache.resolve()),
        "examples": int(len(labels_array)),
        "cutoffs": [str(pd.Timestamp(value)) for value in sorted(times.unique())],
        "threshold_policy": "Tune thresholds on earlier validation weeks; score only the next week.",
        "forward_comparisons": reports,
        "full_validation_diagnostics_only": full_validation,
        "test_data_used": False,
        "training_performed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
