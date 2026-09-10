#!/usr/bin/env python3
"""Append newly selected arXiv papers to an existing manifest."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path


ATOM = "{http://www.w3.org/2005/Atom}"


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u2014", "-")).strip()


def identifier(url: str) -> str:
    return re.sub(r"v\d+$", "", url.rstrip("/").rsplit("/", 1)[-1])


def slug(text: str, limit: int = 72) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    return value[:limit].rstrip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("metadata", nargs="+", type=Path)
    args = parser.parse_args()

    with args.selection.open(encoding="utf-8") as handle:
        selection = list(csv.DictReader(handle, delimiter="\t"))
    with args.manifest.open(encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle, delimiter="\t"))

    fields = list(existing[0])
    rows = {row["arxiv_id"]: row for row in existing}
    metadata: dict[str, dict[str, str]] = {}
    for metadata_path in args.metadata:
        root = ET.parse(metadata_path).getroot()
        for entry in root.findall(f"{ATOM}entry"):
            arxiv_id = identifier(clean(entry.findtext(f"{ATOM}id")))
            metadata[arxiv_id] = {
                "title": clean(entry.findtext(f"{ATOM}title")),
                "authors": "; ".join(
                    clean(author.findtext(f"{ATOM}name"))
                    for author in entry.findall(f"{ATOM}author")
                ),
                "published": clean(entry.findtext(f"{ATOM}published"))[:10],
                "updated": clean(entry.findtext(f"{ATOM}updated"))[:10],
                "abstract": clean(entry.findtext(f"{ATOM}summary")),
                "categories": "; ".join(
                    category.attrib.get("term", "")
                    for category in entry.findall(f"{ATOM}category")
                ),
            }

    missing = []
    for selected in selection:
        arxiv_id = selected["arxiv_id"]
        if arxiv_id in rows:
            continue
        if arxiv_id not in metadata:
            missing.append(arxiv_id)
            continue
        meta = metadata[arxiv_id]
        year = int(arxiv_id[:2]) + 2000
        folder = str(year) if year >= 2025 else "foundations"
        rows[arxiv_id] = {
            **selected,
            "published": meta["published"],
            "updated": meta["updated"],
            "title": meta["title"],
            "authors": meta["authors"],
            "categories": meta["categories"],
            "source_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "local_file": f"paper/{folder}/{arxiv_id}_{slug(meta['title'])}.pdf",
            "abstract": meta["abstract"],
        }

    if missing:
        raise SystemExit("Missing metadata for: " + ", ".join(missing))

    ordered = [rows[item["arxiv_id"]] for item in selection]
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(ordered)


if __name__ == "__main__":
    main()
