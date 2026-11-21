"""Inspect Parquet table schemas without loading the full tables into memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def inspect_table(path: Path) -> None:
    """Print a table's row count, schema, and row-group count."""
    parquet_file = pq.ParquetFile(path)

    print(f"\n=== {path.name} ===")
    print(f"Rows: {parquet_file.metadata.num_rows:,}")
    print(f"Row groups: {parquet_file.metadata.num_row_groups:,}")
    print("Columns:")
    for field in parquet_file.schema_arrow:
        print(f"  - {field.name}: {field.type}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the assignment's Parquet table schemas."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("C:/Users/User/Desktop/assignment2/data/db"),
        help="Folder containing the assignment Parquet files.",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"Data folder does not exist: {args.data_dir}")

    parquet_paths = sorted(args.data_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No .parquet files found in: {args.data_dir}")

    print(f"Data folder: {args.data_dir}")
    print(f"Found {len(parquet_paths)} Parquet tables.")
    for path in parquet_paths:
        inspect_table(path)


if __name__ == "__main__":
    main()
