#!/usr/bin/env python3
"""Draw the taxonomy figure (assets/main_taxonomy.{svg,png,pdf}) from scripts/taxonomy.py."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import date
from html import escape
from pathlib import Path

from taxonomy import TAXONOMY

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "paper_manifest.tsv"
ASSETS = ROOT / "assets"

WIDTH = 2400
PALETTE = {
    "training": {"main": "#7c3aed", "text": "#5b21b6", "tint": "#ede9fe", "hatch": "#b7a6e8", "box": "#f7f4ff", "leaf": "#faf8ff", "leaf_stroke": "#c7b8ef", "arrow": "purple"},
    "inference": {"main": "#3b82d0", "text": "#1e40af", "tint": "#dbeafe", "hatch": "#9fc5ef", "box": "#f4f9ff", "leaf": "#f7fbff", "leaf_stroke": "#abccee", "arrow": "blue"},
    "architecture": {"main": "#2d8a4e", "text": "#065f46", "tint": "#d1fae5", "hatch": "#8bc7a1", "box": "#f3fbf6", "leaf": "#f6fcf8", "leaf_stroke": "#a5d5b5", "arrow": "green"},
    "analysis": {"main": "#d27a0a", "text": "#92400e", "tint": "#fef3c7", "hatch": "#edc477", "box": "#fffaf0", "leaf": "#fffdf7", "leaf_stroke": "#efc982", "arrow": "orange"},
}

# Representative studies shown in each leaf box (one line per section).
EXAMPLES = {
    "training.pre": "Training-Stage Interplay; Grokking in the Wild; Axiomatic Training; Nexus; Complexity Control",
    "training.mid": "DeepSeekMath; Biological Reasoning Models; Additional Logic Training; Reasoning Paradigms; Warm Up Before You Train",
    "training.sft": "MEND; STEPS; Self-Improving Transformers; Theorem-SFT; Meta-RFFT; TAIL; Composable CoT; GEOSD; IGA",
    "training.rl": "SFT Memorizes, RL Generalizes; pass@k boundary; SynLogic; RACES; h1; GraphPRM; GRAIN; CoRPO",
    "training.hybrid": "X-Reasoner; ReasonXL; Rejuvenation; SFT-then-RL baseline audit; LUFFY; Prefix-RFT; SRFT; HPT",
    "inference.trajectory": "Least-to-Most; Skills-in-Context; million-step decomposition; Self-Discover; Constraint-First; Test-Time Hinting",
    "inference.compute": "PRM-guided MCTS; compute-optimal test-time scaling; error-localized scaling; latent test-time compute",
    "inference.assessment": "Residual-stream error directions; UPAIR; thought calibration; ORCA; joint reasoner-verifier; VDS-TTT",
    "inference.externalization": "Tool use for length transfer; Chain-of-Abstraction; TECTON; SMART; ReasoningBank; MILES",
    "architecture.depth": "Universal and Looped Transformers; Loop, Think, & Generalize; retrofitted recurrence; PoLar; Think-at-Hard; Equilibrium Reasoners",
    "architecture.routing": "Position Coupling; Abacus embeddings; TAPE; Randomized YaRN; attention bias calibration; RegularGPT; MORSE; sparse MoE",
    "architecture.memory": "Rational Transductors; SST V2; fast-slow latent recurrence; Thinking States; PENCIL universal computer",
    "analysis.behavioral": "Faith and Fate; MathGAP; Compositional-ARC; length generalization but not robustly; shortest path; GSM-Symbolic; Reversal Curse",
    "analysis.mechanistic": "Shared circuits; decompiling to RASP; Hopping Too Late; two-hop interfaces; grokked transformers; critical windows",
    "analysis.theoretical": "Globality barrier; shortcuts to automata; RASP-L; Curriculum II; length-generalization bounds; algebraic decomposition",
}

FIGURE_SECTION_NAMES = {
    "training.pre": "Pre-training",
    "training.mid": "Mid-training",
    "inference.trajectory": "Trajectory restructuring",
    "inference.compute": "Compute scaling",
    "inference.assessment": "State assessment",
    "inference.externalization": "Externalization",
    "architecture.depth": "Recurrent depth",
    "architecture.routing": "Information routing",
    "architecture.memory": "Recurrent memory",
    "analysis.behavioral": "Behavioral evidence",
    "analysis.mechanistic": "Mechanistic evidence",
    "analysis.theoretical": "Theoretical evidence",
}

ROW_STEP = 56
PILLAR_GAP = 46
TOP = 118


def text(x: float, y: float, value: str, cls: str, fill: str | None = None, anchor: str | None = None) -> str:
    attrs = f'x="{x:g}" y="{y:g}" class="{cls}"'
    if anchor:
        attrs += f' text-anchor="{anchor}"'
    if fill:
        attrs += f' fill="{fill}"'
    return f"  <text {attrs}>{escape(value)}</text>"


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 3, arrow: str | None = None) -> str:
    marker = f' marker-end="url(#arrow-{arrow})"' if arrow else ""
    return f'  <line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="{width:g}"{marker}/>'


def rect(x: float, y: float, w: float, h: float, rx: float, fill: str, stroke: str, width: float, sketchy: bool = True) -> str:
    flt = ' filter="url(#sketchy)"' if sketchy else ""
    return f'  <rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" fill="{fill}" stroke="{stroke}" stroke-width="{width:g}"{flt}/>'


def overlay(x: float, y: float, w: float, h: float, rx: float, fill: str, opacity: float) -> str:
    return f'  <rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" fill="{fill}" opacity="{opacity:g}"/>'


def leaf_box(out: list[str], x: float, y: float, w: float, pal: dict[str, str], head: str, body: str) -> None:
    out.append(rect(x, y - 25, w, 50, 12, pal["leaf"], pal["leaf_stroke"], 2.4))
    out.append(text(x + 20, y - 6, head, "leaf-head", pal["text"]))
    out.append(text(x + 20, y + 15, body, "leaf-text"))


def corpus_counts() -> tuple[int, int, Counter]:
    with MANIFEST.open(encoding="utf-8") as handle:
        core = [row for row in csv.DictReader(handle, delimiter="\t") if row["role"] == "core"]
    years = Counter(row["published"][:4] if row["published"][:4] >= "2024" else "earlier" for row in core)
    return len(core), sum(row["in_survey"] == "yes" for row in core), years


def build_svg() -> str:
    # Layout rows: each section is one row, except post-training, whose three children share a parent box.
    layout = []  # (pillar, [rows]); row = dict(kind, key, label, y)
    y = TOP
    for pillar in TAXONOMY:
        rows = []
        for section in pillar.sections:
            key = f"{pillar.id}.{section.id}"
            rows.append({"key": key, "section": section, "y": y, "child": bool(section.parent)})
            y += ROW_STEP
        layout.append((pillar, rows))
        y += PILLAR_GAP
    height = int(y + 10)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" height="{height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">Taxonomy of reasoning generalization in large language models</title>',
        '  <desc id="desc">A left-to-right tree with four pillars: training, inference, architecture, and analysis. Each pillar branches into the sections of the survey and their leaf categories, with representative studies.</desc>',
        "  <defs>",
        "    <style>",
        "      text { font-family: 'Comic Sans MS', 'Segoe Print', 'Bradley Hand', cursive, sans-serif; }",
        "      .title { font-size: 31px; font-weight: 700; fill: #172033; }",
        "      .subtitle { font-size: 17px; fill: #586174; }",
        "      .root { font-size: 25px; font-weight: 700; fill: #263143; }",
        "      .pillar { font-size: 23px; font-weight: 700; }",
        "      .branch { font-size: 19px; font-weight: 700; }",
        "      .leaf-head { font-size: 17px; font-weight: 700; }",
        "      .leaf-text { font-size: 15px; fill: #303744; }",
        "      .tiny { font-size: 13px; fill: #657083; }",
        "    </style>",
        '    <filter id="sketchy" x="-3%" y="-3%" width="106%" height="106%">',
        '      <feTurbulence type="turbulence" baseFrequency="0.025" numOctaves="4" seed="7" result="noise"/>',
        '      <feDisplacementMap in="SourceGraphic" in2="noise" scale="2.2" xChannelSelector="R" yChannelSelector="G"/>',
        "    </filter>",
    ]
    for pid, pal in PALETTE.items():
        out.append(f'    <pattern id="hatch-{pid}" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">')
        out.append(f'      <line x1="0" y1="0" x2="0" y2="9" stroke="{pal["hatch"]}" stroke-width="1.4" opacity="0.42"/>')
        out.append("    </pattern>")
        out.append(f'    <marker id="arrow-{pal["arrow"]}" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="9" markerHeight="9" orient="auto">')
        out.append(f'      <path d="M 2 3 L 10 6 L 2 9 Z" fill="{pal["main"]}"/>')
        out.append("    </marker>")
    out.extend(
        [
            '    <pattern id="hatch-gray" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">',
            '      <line x1="0" y1="0" x2="0" y2="12" stroke="#aeb5bd" stroke-width="1.5" opacity="0.42"/>',
            "    </pattern>",
            "  </defs>",
            "",
            f'  <rect x="0" y="0" width="{WIDTH}" height="{height}" fill="#fffdfa"/>',
            text(WIDTH / 2, 45, "Beyond the Training Distribution", "title", anchor="middle"),
            text(WIDTH / 2, 76, "Taxonomy of Reasoning Generalization in Large Language Models", "subtitle", anchor="middle"),
        ]
    )

    pillar_centers = []
    for pillar, rows in layout:
        pillar_centers.append((rows[0]["y"] + rows[-1]["y"]) / 2)
    root_mid = (pillar_centers[0] + pillar_centers[-1]) / 2
    root_half = max(242, 0.3 * (pillar_centers[-1] - pillar_centers[0]))
    root_top, root_bottom = root_mid - root_half, root_mid + root_half
    out.append("")
    out.append("  <!-- Root -->")
    out.append(rect(28, root_top, 118, root_bottom - root_top, 29, "url(#hatch-gray)", "#626870", 4))
    out.append(overlay(28, root_top, 118, root_bottom - root_top, 29, "#f1f3f5", 0.40))
    out.append(f'  <text x="87" y="{root_mid + 9:g}" text-anchor="middle" class="root" transform="rotate(-90 87 {root_mid:g})">Reasoning Generalization in LLMs</text>')
    out.append(line(154, root_mid, 282, root_mid, "#667085", 4))
    out.append(line(282, pillar_centers[0], 282, pillar_centers[-1], "#667085", 4))

    for index, ((pillar, rows), center) in enumerate(zip(layout, pillar_centers), start=1):
        pal = PALETTE[pillar.id]
        out.append("")
        out.append(f"  <!-- {pillar.name} -->")
        out.append(line(282, center, 304, center, pal["main"], 4, pal["arrow"]))
        out.append(rect(312, center - 40, 286, 80, 15, f"url(#hatch-{pillar.id})", pal["main"], 4))
        out.append(overlay(312, center - 40, 286, 80, 15, pal["tint"], 0.45))
        first, second = pillar.name.rsplit(" ", 1)
        out.append(
            f'  <text x="455" y="{center - 10:g}" text-anchor="middle" class="pillar" fill="{pal["text"]}">'
            f'<tspan x="455">{escape(f"{index}. {first}")}</tspan><tspan x="455" dy="29">{escape(second)}</tspan></text>'
        )
        # Trunk from the pillar to its section boxes.
        parents = [row for row in rows if not row["child"]]
        children = [row for row in rows if row["child"]]
        branch_ys = [row["y"] for row in parents]
        if children:
            post_y = (children[0]["y"] + children[-1]["y"]) / 2
            branch_ys.append(post_y)
        out.append(line(604, center, 628, center, pal["main"]))
        out.append(line(628, min(branch_ys + [center]), 628, max(branch_ys + [center]), pal["main"]))
        for row in parents:
            key = row["key"]
            out.append(line(628, row["y"], 644, row["y"], pal["main"], 3, pal["arrow"]))
            out.append(rect(652, row["y"] - 20, 330, 40, 10, pal["box"], pal["main"], 3))
            out.append(text(817, row["y"] + 7, FIGURE_SECTION_NAMES[key], "branch", pal["text"], "middle"))
            out.append(line(990, row["y"], 1012, row["y"], pal["main"], 3, pal["arrow"]))
            head = "  •  ".join(leaf.name for leaf in row["section"].leaves)
            leaf_box(out, 1020, row["y"], 1345, pal, head, EXAMPLES[key])
        if children:
            out.append(line(628, post_y, 644, post_y, pal["main"], 3, pal["arrow"]))
            out.append(rect(652, post_y - 20, 330, 40, 10, pal["box"], pal["main"], 3))
            out.append(text(817, post_y + 7, children[0]["section"].parent.capitalize(), "branch", pal["text"], "middle"))
            out.append(line(990, post_y, 1000, post_y, pal["main"]))
            out.append(line(1000, children[0]["y"], 1000, children[-1]["y"], pal["main"]))
            for row in children:
                section = row["section"]
                out.append(line(1000, row["y"], 1012, row["y"], pal["main"], 3, pal["arrow"]))
                out.append(rect(1020, row["y"] - 20, 330, 40, 10, pal["box"], pal["main"], 2.6))
                label = {"SFT": "Supervised fine-tuning", "RL": "Reinforcement learning", "Hybrid": "Hybrid training"}[section.figure_name]
                out.append(text(1185, row["y"] + 7, label, "branch", pal["text"], "middle"))
                out.append(line(1358, row["y"], 1382, row["y"], pal["main"], 3, pal["arrow"]))
                head = "  •  ".join(leaf.name for leaf in section.leaves)
                leaf_box(out, 1390, row["y"], 975, pal, head, EXAMPLES[row["key"]])

    total, in_survey, years = corpus_counts()
    footer = (
        f"Corpus: {total} studies ({in_survey} discussed in the survey + {total - in_survey} from the same screening)  •  "
        f"2023 and earlier: {years['earlier']}  •  2024: {years['2024']}  •  2025: {years['2025']}  •  2026: {years['2026']}  "
        f"•  Updated {date(2026, 10, 7).isoformat()}"
    )
    out.append("")
    out.append(text(WIDTH - 20, height - 16, footer, "tiny", anchor="end"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    svg = build_svg()
    (ASSETS / "main_taxonomy.svg").write_text(svg, encoding="utf-8")
    import cairosvg

    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(ASSETS / "main_taxonomy.png"), scale=1.5)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(ASSETS / "main_taxonomy.pdf"), scale=0.75)
    print("Wrote assets/main_taxonomy.{svg,png,pdf}")


if __name__ == "__main__":
    main()
