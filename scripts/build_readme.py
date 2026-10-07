#!/usr/bin/env python3
"""Build the Awesome-style README from the audited paper manifest and the shared taxonomy."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from taxonomy import TAXONOMY


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "paper_manifest.tsv"
RESOURCES = ROOT / "data" / "resources.tsv"
OUTPUT = ROOT / "README.md"

UPDATED = "7 October 2026"
SEARCH_DATE = "10 September 2026"
MAX_AUTHORS = 12

YEAR_BADGE_COLOR = {
    "2026": "red",
    "2025": "orange",
    "2024": "yellow",
}

CLAIM_ELEMENTS = [
    ("Task and fixed unit", "What is solved, and which components stay constant: weights, prompt, controller, verifier, memory, or tools?"),
    ("Reference exposure", 'Which stage defines "seen": pre-training, mid-training, SFT or distillation, RL, or test-time adaptation?'),
    ("Test shift", "Which axis moves: instance, surface form, composition, length or depth, difficulty, task or domain, language or modality, environment or tool, or required knowledge?"),
    ("Evidence", "How is transfer shown: behavioral outcomes, representational associations, causal interventions, or formal results under explicit assumptions?"),
]

PILLAR_BRANCHES = {
    "training": "Pre-training, mid-training, and post-training (SFT, RL, hybrid)",
    "inference": "Trajectory restructuring, compute scaling, state assessment, externalization",
    "architecture": "Recurrent depth, information routing, recurrent memory",
    "analysis": "Behavioral, mechanistic, and theoretical evidence",
}


def clean(text: str) -> str:
    """Keep generated Markdown plain and compatible with repository style rules."""
    return re.sub(r"\s+", " ", text.replace(chr(0x2014), ":").replace(chr(0x2013), "-")).strip()


def anchor(text: str) -> str:
    """GitHub heading anchor."""
    return re.sub(r"[^\w\- ]", "", clean(text).lower()).replace(" ", "-")


def link(text: str) -> str:
    return f"[{text}](#{anchor(text)})"


def authors(row: dict[str, str]) -> str:
    names = [clean(name) for name in row["authors"].split(";") if name.strip()]
    if len(names) > MAX_AUTHORS:
        names = names[:10] + ["et al."]
    return ", ".join(names)


def paper_item(row: dict[str, str]) -> str:
    year = row["published"][:4]
    color = YEAR_BADGE_COLOR.get(year, "lightgrey")
    title = clean(row["title"])
    names = authors(row)
    return (
        f"- **{title}**{'' if title.endswith(('?', '!', '.')) else '.'} "
        f"*{names}*{'' if names.endswith('.') else '.'} "
        f"[[Paper]]({row['source_url']}) [[PDF]]({row['pdf_url']}) "
        f"![](https://img.shields.io/badge/year-{year}-{color})"
    )


def newest_first(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (row["published"], row["paper_id"]), reverse=True)


def load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def resource_tables(resources: list[dict[str, str]], kind: str) -> list[str]:
    if kind == "training":
        header = ["Dataset", "Year", "Domain", "Problems / tasks", "Supervision"]
        cells = lambda r: [r["year"], r["axis"], r["train"], r["measure"]]  # noqa: E731
    else:
        header = ["Benchmark", "Year", "Evaluation axis", "Train", "Test", "Metric"]
        cells = lambda r: [r["year"], r["axis"], r["train"] or "-", r["test"] or "-", r["measure"]]  # noqa: E731
    lines: list[str] = []
    groups: dict[str, list[dict[str, str]]] = {}
    for row in resources:
        if row["kind"] == kind:
            groups.setdefault(row["group"], []).append(row)
    for group, rows in groups.items():
        lines.extend([f"**{group}**", "", "| " + " | ".join(header) + " |", "|" + " --- |" * len(header)])
        for row in rows:
            name = f"[{row['name']}]({row['url']})" if row["url"] else row["name"]
            lines.append("| " + " | ".join([name, *cells(row)]) + " |")
        lines.append("")
    return lines


def build() -> str:
    rows = load(MANIFEST)
    resources = load(RESOURCES)
    core = [row for row in rows if row["role"] == "core"]
    surveys = [row for row in rows if row["role"] == "survey"]
    in_survey = sum(row["in_survey"] == "yes" for row in core)
    years = Counter(row["published"][:4] for row in core)
    earlier = sum(count for year, count in years.items() if year < "2024")

    by_leaf: dict[str, list[dict[str, str]]] = {}
    for row in core:
        by_leaf.setdefault(row["taxonomy_id"], []).append(row)
    known = {f"{p.id}.{s.id}.{leaf.id}" for p in TAXONOMY for s in p.sections for leaf in s.leaves}
    unknown = sorted(set(by_leaf) - known)
    if unknown:
        raise ValueError(f"Manifest uses unknown taxonomy ids: {unknown}")

    lines = [
        '<div align="center">',
        "",
        "# Awesome Reasoning Generalization",
        "",
        "### Beyond the Training Distribution: A Survey of Reasoning Generalization in Large Language Models",
        "",
        "[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)",
        f"![Papers](https://img.shields.io/badge/papers-{len(core)}-6f42c1)",
        f"![2024 papers](https://img.shields.io/badge/2024%20papers-{years['2024']}-{YEAR_BADGE_COLOR['2024']})",
        f"![2025 papers](https://img.shields.io/badge/2025%20papers-{years['2025']}-{YEAR_BADGE_COLOR['2025']})",
        f"![2026 papers](https://img.shields.io/badge/2026%20papers-{years['2026']}-{YEAR_BADGE_COLOR['2026']})",
        "[![GitHub last commit](https://img.shields.io/github/last-commit/tue09/awesome-reasoning-generalization?logo=github&color=blue)](https://github.com/tue09/awesome-reasoning-generalization/commits/main)",
        "",
        "</div>",
        "",
        (
            f"> **Status:** Updated on {UPDATED} to follow the latest version of the survey (literature search through {SEARCH_DATE}). "
            f"The list contains {len(core)} studies organized by the survey's taxonomy: {in_survey} discussed in the survey "
            f"and {len(core) - in_survey} more from the same screening. By first arXiv posting, {years['2026']} are from 2026, "
            f"{years['2025']} from 2025, {years['2024']} from 2024, and {earlier} are earlier foundations."
        ),
        "",
        "Progress in LLM reasoning is commonly measured on benchmarks that stay close to the training distribution. This "
        "list collects work on whether the reasoning procedures that large language models acquire still hold when "
        "notation, problem structure, length, difficulty, domain, language, or tool interface changes, and on what "
        "training, inference, and architecture contribute to that transfer. The companion text is in [survey.md](survey.md).",
        "",
        "## Contents",
        "",
        f"- {link('Reading a Generalization Claim')}",
        f"- {link('Taxonomy')}",
        f"- {link('Paper List')}",
    ]
    for pillar in TAXONOMY:
        lines.append(f"  - {link(pillar.name)}")
        for section in pillar.sections:
            leaf_links = ", ".join(link(leaf.name) for leaf in section.leaves)
            lines.append(f"    - {link(section.name)}: {leaf_links}")
    lines.extend(
        [
            f"- {link('Datasets and Benchmarks')}",
            f"- {link('Related Surveys')}",
            f"- {link('Citation')}",
            "",
            "## Reading a Generalization Claim",
            "",
            "A reasoning-generalization result can be interpreted only when four elements are explicit.",
            "",
            "| Element | Question |",
            "| --- | --- |",
        ]
    )
    lines.extend(f"| **{name}** | {question} |" for name, question in CLAIM_ELEMENTS)
    lines.extend(
        [
            "",
            "Claims also differ in what is held fixed. *Model-level* generalization concerns the learned parameters under a "
            "fixed decoding procedure; *procedure-level* generalization adds prompting, search, or another inference "
            "algorithm; *system-level* generalization further adds retrievers, verifiers, memories, tools, and environments. "
            "A tool-augmented system can solve longer problems even when the base model has not learned a length-general algorithm.",
            "",
            "## Taxonomy",
            "",
            "| Pillar | Central question | Branches |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(f"| **{p.name}** | {p.question} | {PILLAR_BRANCHES[p.id]} |" for p in TAXONOMY)
    lines.extend(
        [
            "",
            '<p align="center">',
            '  <a href="assets/main_taxonomy.pdf"><img src="assets/main_taxonomy.svg" width="100%" alt="Taxonomy of reasoning generalization in large language models"></a>',
            "</p>",
            "",
            "The first three pillars are interventions, ordered by where they act: on parameters, on test-time computation "
            "with the parameters fixed, and on the model's computational structure. Training is organized by stage and, "
            "within each stage, by the component a method modifies. The fourth pillar separates behavioral, mechanistic, "
            "and theoretical evidence. Each paper appears once, under its primary source of generalization.",
            "",
            "## Paper List",
            "",
        ]
    )

    for pillar in TAXONOMY:
        lines.extend([f"### {pillar.name}", "", f"*{pillar.question}*", ""])
        for section in pillar.sections:
            lines.extend([f"#### {section.name}", "", f"*{section.description}*", ""])
            for leaf in section.leaves:
                papers = by_leaf.get(f"{pillar.id}.{section.id}.{leaf.id}", [])
                lines.extend([f"##### {leaf.name}", "", f"> {leaf.description}", ""])
                if leaf.subgroups:
                    for subgroup in leaf.subgroups:
                        members = newest_first([row for row in papers if row["subgroup"] == subgroup])
                        lines.extend([f"**{subgroup}**", ""])
                        lines.extend(paper_item(row) for row in members)
                        lines.append("")
                    stray = [row for row in papers if row["subgroup"] not in leaf.subgroups]
                    if stray:
                        raise ValueError(f"Papers without a valid subgroup in {leaf.name}: {[r['paper_id'] for r in stray]}")
                else:
                    lines.extend(paper_item(row) for row in newest_first(papers))
                    lines.append("")

    lines.extend(
        [
            "## Datasets and Benchmarks",
            "",
            "Resources summarized in Appendix C of the survey. A benchmark is not in or out of distribution by itself: its "
            "status depends on the evaluated model's pre-training, adaptation data, and task definition. Counts follow the "
            'cited releases, and "~" marks rounded counts.',
            "",
            "### Training Datasets",
            "",
        ]
    )
    lines.extend(resource_tables(resources, "training"))
    lines.extend(["### Evaluation Benchmarks", ""])
    lines.extend(resource_tables(resources, "evaluation"))
    lines.extend(
        [
            "## Related Surveys",
            "",
            *[paper_item(row) for row in newest_first(surveys)],
            "",
            "## Citation",
            "",
            "The paper citation will be added when the survey is released. To cite the living repository in the meantime:",
            "",
            "```bibtex",
            "@misc{awesome_reasoning_generalization_2026,",
            "  title        = {Awesome Reasoning Generalization},",
            "  author       = {{Awesome Reasoning Generalization Contributors}},",
            "  year         = {2026},",
            "  howpublished = {\\url{https://github.com/tue09/awesome-reasoning-generalization}},",
            "  note         = {Accessed: YYYY-MM-DD}",
            "}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
