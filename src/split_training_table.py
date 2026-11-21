"""Write chronological train/validation partitions from a label table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from temporal_split import split_training_table


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("--input must end in .parquet or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split historical user-activity labels by prediction time."
    )
    parser.add_argument("--input", type=Path, required=True, help="Training table (.parquet or .csv).")
    parser.add_argument("--validation-start", required=True, help="Validation start time in UTC.")
    parser.add_argument("--validation-end", required=True, help="Validation end time in UTC.")
    parser.add_argument("--prediction-days", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    table = _read_table(args.input)
    train, validation, excluded = split_training_table(
        table,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        prediction_days=args.prediction_days,
    )

    output_files = {
        "train.parquet": train,
        "validation.parquet": validation,
        "excluded_boundary_rows.parquet": excluded,
    }
    existing = [args.output_dir / name for name in output_files if (args.output_dir / name).exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing split files: {paths}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in output_files.items():
        frame.to_parquet(args.output_dir / name, index=False)

    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Excluded boundary/out-of-window rows: {len(excluded):,}")
    print(f"Saved partitions to: {args.output_dir}")


if __name__ == "__main__":
    main()
