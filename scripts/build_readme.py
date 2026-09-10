#!/usr/bin/env python3
"""Build the Awesome-style README from the audited paper manifest."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper_manifest.tsv"
OUTPUT = ROOT / "README.md"


PILLAR_QUESTIONS = [
    (
        "Training for Generalization",
        "How can learning produce reasoning skills that transfer to unfamiliar problems?",
    ),
    (
        "Inference for Generalization",
        "How can a fixed model solve unfamiliar problems through changes at inference?",
    ),
    (
        "Architecture for Generalization",
        "Which structural properties support transferable reasoning?",
    ),
    (
        "Analysis of Generalization",
        "When does reasoning generalize, and what explains its successes and failures?",
    ),
]


PAPER_SECTIONS = OrderedDict(
    [
        (
            "Training for Generalization",
            [
                "Pre-training and mid-training",
                "Post-training: supervised adaptation",
                "Post-training: RL and reward design",
                "Post-training: hybrid and self-improvement",
            ],
        ),
        (
            "Inference for Generalization",
            [
                "Prompt elicitation and decomposition",
                "Sampling and search",
                "Verification, repair, and stopping",
                "Tools and memory",
            ],
        ),
        (
            "Architecture for Generalization",
            [
                "Recurrent depth and adaptive computation",
                "Position, attention, and locality",
                "Modules, symbols, and structured state",
                "Latent recurrence and equilibrium computation",
            ],
        ),
        (
            "Analysis of Generalization",
            [
                "Behavioral observations",
                "Empirical mechanisms and explanations",
                "Theoretical analysis",
            ],
        ),
    ]
)


MID_TRAINING_BRANCHES = {
    "curriculum learning",
    "data and modular supervision",
    "mid-training and verified synthesis",
    "mid-training/position curriculum",
    "post-training/length transfer",
    "reasoning-skill mixtures",
    "regularization",
    "synthetic data",
}

SUPERVISED_BRANCHES = {
    "post-training/SFT data adaptation",
    "post-training/SFT target design",
    "post-training/causal debiasing",
    "post-training/distillation",
    "post-training/invariant distillation",
    "post-training/negative traces",
    "post-training/on-policy distillation",
    "post-training/self-distillation",
}

HYBRID_BRANCHES = {
    "post-training/SFT plus RL",
    "post-training/SFT versus RL",
    "post-training/SFT-to-RL handoff",
    "post-training/controller-aware",
    "post-training/hybrid and self-improvement",
}


def clean(text: str) -> str:
    """Keep generated Markdown plain and compatible with repository style rules."""
    return text.replace(chr(0x2014), ":").replace(chr(0x2013), "-").strip()


def classify(row: dict[str, str]) -> tuple[str, str] | None:
    pillar = row["pillar"]
    branch = row["branch"]

    if pillar == "training":
        if branch in MID_TRAINING_BRANCHES:
            return "Training for Generalization", "Pre-training and mid-training"
        if branch in SUPERVISED_BRANCHES:
            return "Training for Generalization", "Post-training: supervised adaptation"
        if branch in HYBRID_BRANCHES:
            return "Training for Generalization", "Post-training: hybrid and self-improvement"
        return "Training for Generalization", "Post-training: RL and reward design"

    if pillar == "inference":
        mapping = {
            "prompt decomposition": "Prompt elicitation and decomposition",
            "hint routing": "Prompt elicitation and decomposition",
            "error-localized search": "Sampling and search",
            "test-time thinking": "Sampling and search",
            "monitoring and control": "Verification, repair, and stopping",
            "verification and error detection": "Verification, repair, and stopping",
            "memory": "Tools and memory",
            "tool use": "Tools and memory",
        }
        return "Inference for Generalization", mapping[branch]

    if pillar == "architecture":
        if branch in {"recurrent depth", "recurrence and halting", "recurrent dynamics"}:
            subsection = "Recurrent depth and adaptive computation"
        elif branch in {"positional structure", "locality and recurrence"}:
            subsection = "Position, attention, and locality"
        elif branch in {
            "neuro-symbolic modularity",
            "structured scratchpads",
            "hybrid recurrence",
            "recurrent tool use",
        }:
            subsection = "Modules, symbols, and structured state"
        else:
            subsection = "Latent recurrence and equilibrium computation"
        return "Architecture for Generalization", subsection

    if pillar == "analysis":
        if branch in {
            "behavioral observations",
            "contamination and hybrid controls",
            "contamination-resistant evaluation",
        }:
            subsection = "Behavioral observations"
        elif branch in {
            "behavior and controlled mechanisms",
            "empirical mechanisms",
            "mechanisms and explanations",
        }:
            subsection = "Empirical mechanisms and explanations"
        else:
            subsection = "Theoretical analysis"
        return "Analysis of Generalization", subsection

    return None


def anchor(text: str) -> str:
    return clean(text).lower().replace(":", "").replace(",", "").replace(" ", "-")


def paper_item(row: dict[str, str]) -> str:
    title = clean(row["title"])
    authors = clean(row["authors"].replace(";", ","))
    date = row["published"][:7]
    return (
        f"- **{title}**. *{authors}*. "
        f"[[paper]({row['source_url']})] [[pdf]({row['pdf_url']})], {date}."
    )


def load_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def build() -> str:
    rows = load_rows()
    core = [row for row in rows if row["pillar"] in {"training", "inference", "architecture", "analysis"}]
    year_2026 = sum(row["published"].startswith("2026") for row in core)

    grouped = {
        pillar: {subsection: [] for subsection in subsections}
        for pillar, subsections in PAPER_SECTIONS.items()
    }
    for row in core:
        location = classify(row)
        if location is None:
            raise ValueError(f"Unclassified core paper: {row['arxiv_id']}")
        pillar, subsection = location
        grouped[pillar][subsection].append(row)

    for subsections in grouped.values():
        for papers in subsections.values():
            papers.sort(key=lambda row: (row["published"], row["arxiv_id"]), reverse=True)

    lines = [
        '<div align="center">',
        "",
        "# Awesome Reasoning Generalization",
        "",
        "### Beyond the Training Distribution: A Survey of Reasoning Generalization in Large Language Models",
        "",
        "[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)",
        f"![Core papers](https://img.shields.io/badge/core%20papers-{len(core)}-6f42c1)",
        f"![2026 papers](https://img.shields.io/badge/2026%20papers-{year_2026}-1f77b4)",
        "[![GitHub last commit](https://img.shields.io/github/last-commit/tue09/awesome-reasoning-generalization?logo=github&color=blue)](https://github.com/tue09/awesome-reasoning-generalization/commits/main)",
        "",
        "</div>",
        "",
        "> **Status:** The literature search was updated on 8 September 2026. The corpus contains 124 core studies, including 86 first posted in 2026.",
        "",
        "## Contents",
        "",
        "- [Taxonomy](#taxonomy)",
        "- [Paper list](#paper-list)",
    ]

    for pillar, subsections in PAPER_SECTIONS.items():
        lines.append(f"  - [{pillar}](#{anchor(pillar)})")
        for subsection in subsections:
            lines.append(f"    - [{subsection}](#{anchor(subsection)})")

    lines.extend(
        [
            "- [Citation](#citation)",
            "",
            "## Taxonomy",
            "",
            "| Pillar | Central question |",
            "| --- | --- |",
        ]
    )

    for pillar, question in PILLAR_QUESTIONS:
        lines.append(f"| **{pillar}** | {question} |")

    lines.extend(
        [
            "",
            '<p align="center">',
            '  <a href="main_taxonomy.pdf"><img src="main_taxonomy.svg" width="100%" alt="Taxonomy of reasoning generalization in large language models"></a>',
            "</p>",
            "",
            "The first three pillars concern interventions. The fourth separates behavioral observations from empirical mechanisms and theoretical results. Each paper receives one primary manifest label, even when it informs several sections.",
            "",
            "## Paper list",
            "",
        ]
    )

    for pillar, subsections in PAPER_SECTIONS.items():
        pillar_count = sum(len(grouped[pillar][subsection]) for subsection in subsections)
        lines.extend([f"### {pillar} ({pillar_count})", ""])
        for subsection in subsections:
            papers = grouped[pillar][subsection]
            lines.extend([f"#### {subsection} ({len(papers)})", ""])
            lines.extend(paper_item(row) for row in papers)
            lines.append("")

    lines.extend(
        [
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
