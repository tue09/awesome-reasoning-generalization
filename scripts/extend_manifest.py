#!/usr/bin/env python3
"""Append newly selected papers to an existing manifest, keeping existing rows unchanged."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_manifest import build_row, load_external, load_metadata, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("metadata", nargs="*", type=Path)
    args = parser.parse_args()

    with args.selection.open(encoding="utf-8") as handle:
        selection = list(csv.DictReader(handle, delimiter="\t"))
    with args.manifest.open(encoding="utf-8") as handle:
        rows = {row["paper_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

    metadata = {**load_metadata(args.metadata), **load_external()}
    missing = []
    for selected in selection:
        paper_id = selected["paper_id"]
        if paper_id in rows:
            continue
        if paper_id not in metadata:
            missing.append(paper_id)
            continue
        rows[paper_id] = build_row(selected, metadata[paper_id])

    if missing:
        raise SystemExit("Missing metadata for: " + ", ".join(missing))

    write_manifest(args.manifest, [rows[item["paper_id"]] for item in selection])


if __name__ == "__main__":
    main()
