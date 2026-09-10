#!/usr/bin/env python3
"""Verify that every manifest entry has a nontrivial PDF file."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    repo_root = Path(__file__).resolve().parents[1]
    bad: list[tuple[str, str, str]] = []
    for row in rows:
        path = repo_root / row["local_file"]
        if not path.exists():
            bad.append((row["arxiv_id"], str(path), "missing"))
            continue
        size = path.stat().st_size
        with path.open("rb") as handle:
            magic = handle.read(5)
        if size < 10_000 or magic != b"%PDF-":
            bad.append((row["arxiv_id"], str(path), f"invalid ({size} bytes)"))
            continue
        check = subprocess.run(
            ["pdfinfo", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            detail = check.stderr.strip().splitlines()[0] if check.stderr.strip() else "pdfinfo failed"
            bad.append((row["arxiv_id"], str(path), detail))

    print(f"manifest={len(rows)} valid={len(rows) - len(bad)} invalid={len(bad)}")
    for arxiv_id, path, problem in bad:
        print(f"{arxiv_id}\t{problem}\t{path}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
