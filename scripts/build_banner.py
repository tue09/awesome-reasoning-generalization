#!/usr/bin/env python3
"""Draw the README banner (assets/banner.svg) as a self-contained SVG with text converted to outlines.

Fonts: the variable TTFs of Inter and JetBrains Mono from github.com/google/fonts (SIL Open Font License),
placed in the directory given by --fonts.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 400
LEFT = 68
TEXT_MAX = 580
TOP = 76

BLUE, ORANGE, GREEN, PURPLE, PINK = "#6ea8ff", "#ff9d5c", "#4fd1a1", "#b18cff", "#f78fb3"


def num(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


class Face:
    def __init__(self, path: Path, **axes: float):
        font = TTFont(path)
        tags = {a.axisTag for a in font["fvar"].axes}
        self.font = instantiateVariableFont(font, {k: v for k, v in axes.items() if k in tags})
        self.upem = self.font["head"].unitsPerEm
        self.cmap = self.font.getBestCmap()
        self.glyphs = self.font.getGlyphSet()
        self.hmtx = self.font["hmtx"]
        self.pairs = self._pair_subtables()

    def _pair_subtables(self):
        if "GPOS" not in self.font:
            return []
        gpos = self.font["GPOS"].table
        indices = sorted({i for fr in gpos.FeatureList.FeatureRecord if fr.FeatureTag == "kern" for i in fr.Feature.LookupListIndex})
        out = []
        for i in indices:
            lookup = gpos.LookupList.Lookup[i]
            for st in lookup.SubTable:
                if lookup.LookupType == 9:
                    st = st.ExtSubTable
                if st.LookupType == 2:
                    out.append(st)
        return out

    def kern(self, left: str, right: str) -> float:
        for st in self.pairs:
            cov = st.Coverage.glyphs
            if left not in cov:
                continue
            if st.Format == 1:
                for rec in st.PairSet[cov.index(left)].PairValueRecord:
                    if rec.SecondGlyph == right:
                        return getattr(rec.Value1, "XAdvance", 0) or 0
            else:
                c1 = st.ClassDef1.classDefs.get(left, 0)
                c2 = st.ClassDef2.classDefs.get(right, 0)
                value = st.Class1Record[c1].Class2Record[c2].Value1
                adv = getattr(value, "XAdvance", 0) if value is not None else 0
                if adv:
                    return adv
        return 0

    def _advances(self, text: str, size: float, tracking: float):
        s = size / self.upem
        names = [self.cmap[ord(c)] for c in text]
        for i, name in enumerate(names):
            adv = self.hmtx[name][0]
            if i + 1 < len(names):
                adv += self.kern(name, names[i + 1])
            yield name, adv * s + tracking

    def width(self, text: str, size: float, tracking: float = 0) -> float:
        return sum(adv for _, adv in self._advances(text, size, tracking)) - tracking

    def path(self, text: str, x: float, y: float, size: float, tracking: float = 0) -> str:
        s = size / self.upem
        pen = SVGPathPen(self.glyphs, ntos=num)
        for name, adv in self._advances(text, size, tracking):
            self.glyphs[name].draw(TransformPen(pen, (s, 0, 0, -s, x, y)))
            x += adv
        return pen.getCommands()


def blob(cx: float, cy: float, rx: float, ry: float, seed: int, n: int = 48) -> tuple[str, callable]:
    """Smooth closed organic shape, and a function giving its radius scale at an angle."""
    rng = random.Random(seed)
    waves = [(k, rng.uniform(0.02, 0.06), rng.uniform(0, 2 * math.pi)) for k in (2, 3, 5)]

    def scale(theta: float) -> float:
        return 1 + sum(a * math.sin(k * theta + p) for k, a, p in waves)

    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        r = scale(t)
        pts.append((cx + rx * r * math.cos(t), cy + ry * r * math.sin(t)))
    d = [f"M{num(pts[0][0])} {num(pts[0][1])}"]
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C{num(c1[0])} {num(c1[1])} {num(c2[0])} {num(c2[1])} {num(p2[0])} {num(p2[1])}")
    return "".join(d) + "Z", scale


def cubic(p0, c1, c2, p1, t):
    u = 1 - t
    return tuple(u**3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t**3 * d for a, b, c, d in zip(p0, c1, c2, p1))


def build(fonts: Path) -> str:
    inter = fonts / "Inter[opsz,wght].ttf"
    mono = fonts / "JetBrainsMono[wght].ttf"
    title_face = Face(inter, wght=760, opsz=32)
    body_face = Face(inter, wght=420, opsz=20)
    pill_face = Face(inter, wght=600, opsz=14)
    mono_face = Face(mono, wght=500)

    out: list[str] = []
    add = out.append

    # Title block.
    title_size = 58
    lines = ["Awesome Reasoning", "Generalization"]
    while max(title_face.width(t, title_size, -1.2) for t in lines) > TEXT_MAX:
        title_size -= 1
    kicker = "CURATED PAPER LIST  ·  SURVEY COMPANION"
    add(f'<path fill="#8ea0c8" d="{mono_face.path(kicker, LEFT, TOP, 14, 1.6)}"/>')
    add(f'<path fill="#f4f6fb" d="{title_face.path(lines[0], LEFT, TOP + 66, title_size, -1.2)}"/>')
    add(f'<path fill="url(#titleGrad)" d="{title_face.path(lines[1], LEFT, TOP + 66 + title_size * 1.08, title_size, -1.2)}"/>')
    sub = ["Beyond the Training Distribution: A Survey of Reasoning", "Generalization in Large Language Models"]
    sub_size = 20
    while max(body_face.width(t, sub_size) for t in sub) > TEXT_MAX:
        sub_size -= 0.5
    sub_y = TOP + 66 + title_size * 1.08 + 48
    for i, text in enumerate(sub):
        add(f'<path fill="#aab4cc" d="{body_face.path(text, LEFT, sub_y + i * sub_size * 1.4, sub_size)}"/>')

    # Pillar pills, colored as in Figure 1 of the survey.
    x, top, h = LEFT, sub_y + sub_size * 1.4 + 26, 32
    for label, color in (("Training", BLUE), ("Inference", ORANGE), ("Architecture", GREEN), ("Analysis", PURPLE)):
        w = pill_face.width(label, 15) + 32
        add(f'<rect x="{num(x)}" y="{num(top)}" width="{num(w)}" height="{h}" rx="16" fill="{color}" fill-opacity=".12" stroke="{color}" stroke-opacity=".55"/>')
        add(f'<path fill="{color}" d="{pill_face.path(label, x + 16, top + 21.3, 15)}"/>')
        x += w + 10
    assert top + h < H - 24, "pills overflow"

    # Illustration: a training distribution and reasoning chains that leave it.
    cx, cy, rx, ry = 826, 200, 112, 96
    shape, scale = blob(cx, cy, rx, ry, seed=7)
    add('<g filter="url(#soft)" opacity=".9">')
    add(f'<path d="{shape}" fill="url(#distGrad)"/>')
    add("</g>")
    add(f'<path d="{shape}" fill="none" stroke="{BLUE}" stroke-opacity=".5" stroke-width="1.5" stroke-dasharray="2 6" stroke-linecap="round"/>')

    rng = random.Random(3)
    dots = []
    while len(dots) < 70:
        t = rng.uniform(0, 2 * math.pi)
        r = math.sqrt(rng.random()) * 0.86 * scale(t)
        dots.append((cx + rx * r * math.cos(t), cy + ry * r * math.sin(t), rng.uniform(1.3, 2.4), rng.uniform(0.25, 0.7)))
    add("".join(f'<circle cx="{num(a)}" cy="{num(b)}" r="{num(r)}" fill="#a9c8ff" fill-opacity="{c:.2f}"/>' for a, b, r, c in dots))
    add(f'<path fill="{BLUE}" fill-opacity=".8" d="{mono_face.path("training distribution", cx - mono_face.width("training distribution", 12.5, 0.4) / 2, cy + ry + 34, 12.5, 0.4)}"/>')

    # In-distribution chain (solid) to the boundary.
    theta = math.radians(-4)
    exit_pt = (cx + rx * scale(theta) * math.cos(theta), cy + ry * scale(theta) * math.sin(theta))
    chain = [(760, 238), (797, 190), (843, 214), (887, 178), exit_pt]
    add(f'<polyline points="{" ".join(f"{num(a)},{num(b)}" for a, b in chain)}" fill="none" stroke="#e8eefc" stroke-opacity=".85" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')

    # Out-of-distribution chains (dashed, animated) toward four shifts.
    targets = [
        ("composition", ORANGE, (1080, 72)),
        ("length", GREEN, (1132, 152)),
        ("difficulty", PURPLE, (1124, 240)),
        ("domain", PINK, (1066, 322)),
    ]
    ends = []
    for i, (label, color, end) in enumerate(targets):
        c1 = (exit_pt[0] + 72, exit_pt[1] + (end[1] - exit_pt[1]) * 0.15)
        c2 = (end[0] - 84, end[1])
        d = f"M{num(exit_pt[0])} {num(exit_pt[1])}C{num(c1[0])} {num(c1[1])} {num(c2[0])} {num(c2[1])} {num(end[0])} {num(end[1])}"
        add(f'<path d="{d}" fill="none" stroke="{color}" stroke-opacity=".9" stroke-width="2" stroke-linecap="round" class="flow"/>')
        for t in (0.42, 0.74):
            px, py = cubic(exit_pt, c1, c2, end, t)
            add(f'<circle cx="{num(px)}" cy="{num(py)}" r="3.6" fill="#0d1328" stroke="{color}" stroke-width="1.8"/>')
        ends.append((label, color, end, i))

    for label, color, (ex, ey), i in ends:
        add(f'<circle cx="{num(ex)}" cy="{num(ey)}" r="15" fill="{color}" opacity=".22" class="halo d{i}"/>')
        add(f'<circle cx="{num(ex)}" cy="{num(ey)}" r="6.5" fill="{color}"/>')
        add(f'<path fill="{color}" d="{mono_face.path(label, ex + 24, ey + 4.6, 13.5, 0.3)}"/>')

    # In-distribution nodes drawn last so they sit on top of the line.
    for k, (a, b) in enumerate(chain):
        r = 6.5 if k == len(chain) - 1 else 5
        add(f'<circle cx="{num(a)}" cy="{num(b)}" r="{r}" fill="#f4f6fb" stroke="{BLUE}" stroke-width="2"/>')

    right_edge = max(ex + 24 + mono_face.width(lbl, 13.5, 0.3) for lbl, _, (ex, _), _ in ends)
    assert right_edge < W - 40, f"labels overflow: {right_edge}"

    defs = f"""<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#0a0f1f"/><stop offset=".55" stop-color="#10163a"/><stop offset="1" stop-color="#1b1238"/>
</linearGradient>
<radialGradient id="glow" cx="{cx}" cy="{cy}" r="420" gradientUnits="userSpaceOnUse">
<stop offset="0" stop-color="#3f63e0" stop-opacity=".32"/><stop offset="1" stop-color="#3f63e0" stop-opacity="0"/>
</radialGradient>
<radialGradient id="glow2" cx="1180" cy="40" r="300" gradientUnits="userSpaceOnUse">
<stop offset="0" stop-color="{PURPLE}" stop-opacity=".18"/><stop offset="1" stop-color="{PURPLE}" stop-opacity="0"/>
</radialGradient>
<radialGradient id="distGrad" cx="{cx}" cy="{cy}" r="{rx * 1.1}" gradientUnits="userSpaceOnUse">
<stop offset="0" stop-color="{BLUE}" stop-opacity=".42"/><stop offset=".7" stop-color="{BLUE}" stop-opacity=".16"/><stop offset="1" stop-color="{BLUE}" stop-opacity=".05"/>
</radialGradient>
<linearGradient id="titleGrad" x1="{LEFT}" y1="0" x2="{LEFT + 460}" y2="0" gradientUnits="userSpaceOnUse">
<stop offset="0" stop-color="{BLUE}"/><stop offset=".5" stop-color="{PURPLE}"/><stop offset="1" stop-color="{PINK}"/>
</linearGradient>
<pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="1" fill="#ffffff" fill-opacity=".07"/>
</pattern>
<linearGradient id="gridFade" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="#fff" stop-opacity=".25"/><stop offset=".6" stop-color="#fff" stop-opacity="1"/>
</linearGradient>
<mask id="gridMask"><rect width="{W}" height="{H}" fill="url(#gridFade)"/></mask>
<filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6"/></filter>
<clipPath id="frame"><rect width="{W}" height="{H}" rx="24"/></clipPath>
</defs>"""

    style = """<style>
.flow{stroke-dasharray:3 7;animation:flow 1.6s linear infinite}
@keyframes flow{to{stroke-dashoffset:-20}}
.halo{transform-box:fill-box;transform-origin:center;animation:pulse 3.2s ease-in-out infinite}
.d1{animation-delay:-.8s}.d2{animation-delay:-1.6s}.d3{animation-delay:-2.4s}
@keyframes pulse{0%,100%{opacity:.12;transform:scale(.75)}50%{opacity:.4;transform:scale(1.2)}}
@media (prefers-reduced-motion:reduce){.flow,.halo{animation:none}}
</style>"""

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Awesome Reasoning Generalization: Beyond the Training Distribution, a survey of reasoning generalization in large language models">',
            "<title>Awesome Reasoning Generalization</title>",
            defs,
            style,
            '<g clip-path="url(#frame)">',
            f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
            f'<rect width="{W}" height="{H}" fill="url(#glow)"/>',
            f'<rect width="{W}" height="{H}" fill="url(#glow2)"/>',
            f'<rect width="{W}" height="{H}" fill="url(#grid)" mask="url(#gridMask)"/>',
            *out,
            "</g>",
            f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="23.5" fill="none" stroke="#ffffff" stroke-opacity=".08"/>',
            "</svg>",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "banner.svg")
    args = parser.parse_args()
    args.out.write_text(build(args.fonts), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
