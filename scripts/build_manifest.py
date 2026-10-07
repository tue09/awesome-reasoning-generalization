#!/usr/bin/env python3
"""Join the curated selection with authoritative arXiv metadata."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from taxonomy import leaf_index


ATOM = "{http://www.w3.org/2005/Atom}"
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external_metadata.tsv"

FIELDS = [
    "paper_id", "citation_key", "role", "pillar", "section", "leaf", "subgroup", "taxonomy_id",
    "in_survey", "published", "updated", "title", "authors", "categories", "inclusion_note",
    "source_url", "pdf_url", "local_file", "abstract",
]


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("—", "-")).strip()


def identifier(url: str) -> str:
    return re.sub(r"v\d+$", "", url.rstrip("/").rsplit("/", 1)[-1])


def slug(text: str, limit: int = 72) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    return value[:limit].rstrip("_")


def load_metadata(paths: list[Path]) -> dict[str, dict[str, str]]:
    """Parse arXiv Atom exports keyed by version-free identifier."""
    metadata: dict[str, dict[str, str]] = {}
    for metadata_path in paths:
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
                "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            }
    return metadata


def load_external(path: Path = EXTERNAL) -> dict[str, dict[str, str]]:
    """Hand-entered metadata for papers without an arXiv version."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            row["paper_id"]: {**row, "updated": "", "abstract": "", "categories": ""}
            for row in csv.DictReader(handle, delimiter="\t")
        }


def local_path(paper_id: str, year: int, title: str) -> str:
    """Reuse an archived PDF for this paper if one exists; otherwise name a new one."""
    file_id = paper_id.replace(":", "_")
    for archive in ("paper", "excluded_paper"):
        existing = sorted((ROOT / archive).glob(f"*/{file_id}_*.pdf"))
        if existing:
            return existing[0].relative_to(ROOT).as_posix()
    folder = str(year) if year >= 2025 else "foundations"
    return f"paper/{folder}/{file_id}_{slug(title)}.pdf"


def build_row(selected: dict[str, str], meta: dict[str, str]) -> dict[str, str]:
    paper_id = selected["paper_id"]
    if ":" in paper_id:
        year = int(meta["published"][:4])
    else:
        year = int(paper_id[:2]) + 2000
    pillar = section = leaf = ""
    if selected["taxonomy_id"]:
        pillar_obj, section_obj, leaf_obj = leaf_index()[selected["taxonomy_id"]]
        pillar, section, leaf = pillar_obj.name, section_obj.name, leaf_obj.name
        if selected["subgroup"] and selected["subgroup"] not in leaf_obj.subgroups:
            raise SystemExit(f"Unknown subgroup {selected['subgroup']!r} for {paper_id}")
    return {
        **selected,
        "pillar": pillar,
        "section": section,
        "leaf": leaf,
        "published": meta["published"],
        "updated": meta["updated"],
        "title": meta["title"],
        "authors": meta["authors"],
        "categories": meta["categories"],
        "source_url": meta["source_url"],
        "pdf_url": meta["pdf_url"],
        "local_file": local_path(paper_id, year, meta["title"]),
        "abstract": meta["abstract"],
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("metadata", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    with args.selection.open(encoding="utf-8") as handle:
        selection = list(csv.DictReader(handle, delimiter="\t"))

    metadata = {**load_metadata(args.metadata), **load_external()}
    missing = [row["paper_id"] for row in selection if row["paper_id"] not in metadata]
    if missing:
        raise SystemExit("Missing metadata for: " + ", ".join(missing))

    write_manifest(args.output, [build_row(row, metadata[row["paper_id"]]) for row in selection])


if __name__ == "__main__":
    main()
