"""Check whether cached validation predictions depend on mini-batch grouping."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from feature_encoding import TrainFeatureEncoder
from gnn_model import build_activity_model, collate_subgraphs
from labels import LABEL_COLUMNS


def _predict(model, examples, order, batch_size, device):
    outputs = {}
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            inputs = [examples[index][0] for index in indices]
            batch = collate_subgraphs(inputs).to(device)
            probabilities = torch.sigmoid(model(batch)).cpu().numpy()
            for index, row in zip(indices, probabilities):
                outputs[index] = row
    return np.stack([outputs[index] for index in range(len(examples))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output.resolve()}")
    if args.max_examples <= 1 or args.batch_size <= 0:
        raise ValueError("max-examples must exceed 1 and batch-size must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("labels") != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint label order does not match the task")
    manifest_path = args.validation_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "validation":
        raise ValueError("The supplied cache is not a validation split")

    examples = []
    for filename in manifest["files"]:
        chunk = torch.load(args.validation_cache / filename, map_location="cpu", weights_only=False)
        examples.extend(chunk)
        if len(examples) >= args.max_examples:
            break
    examples = examples[: args.max_examples]
    if len(examples) < 2:
        raise ValueError("Validation cache has fewer than two examples")

    encoder = TrainFeatureEncoder.from_dict(checkpoint["encoder"])
    config = checkpoint["config"]
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()

    canonical = list(range(len(examples)))
    shuffled = canonical.copy()
    random.Random(args.seed).shuffle(shuffled)
    predictions = {
        "single": _predict(model, examples, canonical, 1, device),
        "batch": _predict(model, examples, canonical, args.batch_size, device),
        "shuffled_batch": _predict(model, examples, shuffled, args.batch_size, device),
    }
    thresholds = np.asarray(checkpoint["thresholds"], dtype=float)
    reference = predictions["single"]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_cache": str(args.validation_cache.resolve()),
        "device": str(device),
        "examples": len(examples),
        "batch_size": args.batch_size,
        "comparisons": {},
        "test_data_used": False,
        "training_performed": False,
    }
    for name, values in predictions.items():
        if name == "single":
            continue
        absolute_difference = np.abs(values - reference)
        reference_labels = reference >= thresholds[None, :]
        candidate_labels = values >= thresholds[None, :]
        report["comparisons"][name] = {
            "max_absolute_probability_difference": float(absolute_difference.max()),
            "mean_absolute_probability_difference": float(absolute_difference.mean()),
            "changed_binary_predictions": int(np.count_nonzero(reference_labels != candidate_labels)),
            "total_binary_predictions": int(reference_labels.size),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
