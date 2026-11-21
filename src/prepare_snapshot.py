"""Small command-line entry point for checking a historical graph snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import load_snapshot
from relations import build_relation_edges


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the assignment tables as they were available before a UTC cutoff."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("C:/Users/User/Desktop/assignment2/data/db"),
        help="Folder containing the assignment Parquet files.",
    )
    parser.add_argument(
        "--cutoff",
        required=True,
        help="Prediction time in UTC, for example 2026-09-15T00:00:00.",
    )
    args = parser.parse_args()

    tables = load_snapshot(args.data_dir, args.cutoff)
    edges = build_relation_edges(tables)
    print(f"Snapshot cutoff (UTC): {args.cutoff}")
    for table_name, table in tables.items():
        print(f"{table_name}: {len(table):,} rows available")
    print(f"\nTyped edge directions (including reverse): {len(edges)}")
    for (source, relation, target), (source_ids, _) in edges.items():
        print(f"{source} -[{relation}]-> {target}: {len(source_ids):,} edges")


if __name__ == "__main__":
    main()
