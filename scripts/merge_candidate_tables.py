#!/usr/bin/env python3
"""Merge arXiv candidate TSV files while preserving query provenance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "arxiv_id",
    "published",
    "updated",
    "title",
    "authors",
    "categories",
    "abstract",
    "queries",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    records: dict[str, dict[str, str]] = {}
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                arxiv_id = row["arxiv_id"]
                if arxiv_id not in records:
                    records[arxiv_id] = {field: row.get(field, "") for field in FIELDS}
                    continue
                old_queries = set(filter(None, records[arxiv_id]["queries"].split("; ")))
                new_queries = set(filter(None, row.get("queries", "").split("; ")))
                records[arxiv_id]["queries"] = "; ".join(sorted(old_queries | new_queries))

    ordered = sorted(
        records.values(),
        key=lambda row: (row["published"], row["arxiv_id"]),
        reverse=True,
    )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(ordered)


if __name__ == "__main__":
    main()
