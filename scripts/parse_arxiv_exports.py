#!/usr/bin/env python3
"""Merge arXiv Atom exports into a deduplicated TSV candidate table."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def arxiv_id(url: str) -> str:
    return re.sub(r"v\d+$", "", url.rstrip("/").rsplit("/", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    records: dict[str, dict[str, str]] = {}
    for path in args.xml:
        root = ET.parse(path).getroot()
        query = path.stem.removeprefix("arxiv_")
        entries = root.findall(f"{ATOM}entry")
        total_text = root.findtext(f"{OPENSEARCH}totalResults")
        total = int(total_text) if total_text and total_text.isdigit() else len(entries)
        if total > len(entries):
            print(
                f"warning: {path} contains {len(entries)} of {total} results; "
                "split the query by submission date before screening",
                file=sys.stderr,
            )
        for entry in entries:
            identifier = arxiv_id(clean(entry.findtext(f"{ATOM}id")))
            if not identifier:
                continue
            authors = "; ".join(
                clean(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")
            )
            categories = "; ".join(
                c.attrib.get("term", "") for c in entry.findall(f"{ATOM}category")
            )
            rec = records.setdefault(
                identifier,
                {
                    "arxiv_id": identifier,
                    "published": clean(entry.findtext(f"{ATOM}published"))[:10],
                    "updated": clean(entry.findtext(f"{ATOM}updated"))[:10],
                    "title": clean(entry.findtext(f"{ATOM}title")),
                    "authors": authors,
                    "categories": categories,
                    "abstract": clean(entry.findtext(f"{ATOM}summary")),
                    "queries": "",
                },
            )
            rec["queries"] = "; ".join(sorted(set(filter(None, rec["queries"].split("; "))) | {query}))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arxiv_id", "published", "updated", "title", "authors", "categories", "abstract", "queries"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(records.values(), key=lambda row: (row["published"], row["arxiv_id"]), reverse=True))


if __name__ == "__main__":
    main()
