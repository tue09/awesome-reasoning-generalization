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
TITLE_Y = 123

BLUE, ORANGE, GREEN, PURPLE, PINK = "#6ea8ff", "#ff9d5c", "#4fd1a1", "#b18cff", "#f78fb3"
CYAN, AMBER = "#8be9ff", "#ffd166"


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


def check(x: float, y: float, r: float, color: str) -> str:
    """Colored badge with a check mark."""
    d = f"M{num(x - r * 0.42)} {num(y + r * 0.02)}L{num(x - r * 0.1)} {num(y + r * 0.34)}L{num(x + r * 0.45)} {num(y - r * 0.32)}"
    return (
        f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}" fill="{color}"/>'
        f'<path d="{d}" fill="none" stroke="#0b1020" stroke-width="{num(r * 0.3)}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def robot(x: float, y: float) -> str:
    """A small robot standing at (x, y), right arm raised toward the unseen problems."""
    body, limb = 'fill="url(#robotBody)"', "#b9c7ec"
    parts = [
        '<ellipse cx="0" cy="4" rx="46" ry="7" fill="#000" fill-opacity=".35"/>',
        f'<rect x="-21" y="-28" width="13" height="24" rx="6" fill="{limb}"/>',
        f'<rect x="8" y="-28" width="13" height="24" rx="6" fill="{limb}"/>',
        f'<ellipse cx="-15" cy="-3" rx="11" ry="5.5" fill="{BLUE}"/>',
        f'<ellipse cx="15" cy="-3" rx="11" ry="5.5" fill="{BLUE}"/>',
        f'<path d="M-33 -68L-49 -42" stroke="{limb}" stroke-width="12" stroke-linecap="round"/>',
        f'<circle cx="-50" cy="-40" r="7.5" fill="{BLUE}"/>',
        f'<path d="M33 -68L57 -90" stroke="{limb}" stroke-width="12" stroke-linecap="round"/>',
        f'<circle cx="59" cy="-92" r="7.5" fill="{BLUE}"/>',
        f'<rect x="-35" y="-84" width="70" height="60" rx="18" {body}/>',
        '<circle cx="0" cy="-54" r="10" fill="#0d1328"/>',
        f'<circle cx="0" cy="-54" r="5" fill="{PINK}" class="beat"/>',
        f'<rect x="-10" y="-92" width="20" height="10" rx="3" fill="{limb}"/>',
        f'<rect x="-57" y="-138" width="11" height="28" rx="5" fill="{BLUE}"/>',
        f'<rect x="46" y="-138" width="11" height="28" rx="5" fill="{BLUE}"/>',
        f'<rect x="-49" y="-162" width="98" height="74" rx="26" {body}/>',
        '<rect x="-39" y="-152" width="78" height="54" rx="18" fill="#0d1328"/>',
        '<g class="blink">'
        f'<rect x="-23" y="-138" width="13" height="19" rx="6.5" fill="{CYAN}"/><rect x="10" y="-138" width="13" height="19" rx="6.5" fill="{CYAN}"/>'
        '<circle cx="-14" cy="-133" r="2.6" fill="#fff"/><circle cx="19" cy="-133" r="2.6" fill="#fff"/>'
        "</g>",
        f'<path d="M-9 -110Q0 -103 9 -110" fill="none" stroke="{CYAN}" stroke-width="2.6" stroke-linecap="round"/>',
        f'<ellipse cx="-28" cy="-110" rx="5.5" ry="3" fill="{PINK}" fill-opacity=".6"/>',
        f'<ellipse cx="28" cy="-110" rx="5.5" ry="3" fill="{PINK}" fill-opacity=".6"/>',
        f'<path d="M0 -162V-180" stroke="{limb}" stroke-width="4" stroke-linecap="round"/>',
        f'<circle cx="0" cy="-186" r="13" fill="{AMBER}" opacity=".25" class="halo"/>',
        f'<circle cx="0" cy="-186" r="7" fill="{AMBER}"/>',
    ]
    return f'<g transform="translate({num(x)} {num(y)})">' + "".join(parts) + "</g>"


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
    add(f'<path fill="#f4f6fb" d="{title_face.path(lines[0], LEFT, TITLE_Y, title_size, -1.2)}"/>')
    add(f'<path fill="url(#titleGrad)" d="{title_face.path(lines[1], LEFT, TITLE_Y + title_size * 1.08, title_size, -1.2)}"/>')
    sub = ["Beyond the Training Distribution: A Survey of Reasoning", "Generalization in Large Language Models"]
    sub_size = 20
    while max(body_face.width(t, sub_size) for t in sub) > TEXT_MAX:
        sub_size -= 0.5
    sub_y = TITLE_Y + title_size * 1.08 + 48
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

    # Illustration: a model trained on familiar problems reasons step by step through unseen ones.
    cx, cy, rx, ry = 776, 206, 88, 112
    shape, _ = blob(cx, cy, rx, ry, seed=7)
    add('<g filter="url(#soft)" opacity=".9">')
    add(f'<path d="{shape}" fill="url(#distGrad)"/>')
    add("</g>")
    add(f'<path d="{shape}" fill="none" stroke="{BLUE}" stroke-opacity=".55" stroke-width="1.5" stroke-dasharray="2 6" stroke-linecap="round"/>')

    # Solved training problems, each with a short reasoning chain.
    for y in (150, 198, 246):
        x0 = cx - 54
        add(f'<rect x="{num(x0)}" y="{y - 16}" width="108" height="32" rx="10" fill="#0d1430" fill-opacity=".6" stroke="{BLUE}" stroke-opacity=".45"/>')
        nodes = [x0 + 18 + 21 * k for k in range(3)]
        add(f'<path d="M{num(nodes[0])} {y}H{num(nodes[-1])}" stroke="#cfe0ff" stroke-opacity=".7" stroke-width="2"/>')
        add("".join(f'<circle cx="{num(n)}" cy="{y}" r="4.5" fill="#e8f0ff"/>' for n in nodes))
        add(check(x0 + 89, y, 8, BLUE))

    # Unseen problems the robot points to.
    labels = [("longer", ORANGE), ("harder", GREEN), ("new domains", PURPLE), ("new combinations", PINK)]
    card_size, card_h, card_x = 15, 40, 1048
    card_w = max(pill_face.width(text, card_size) for text, _ in labels) + 62
    assert card_x + card_w < W - 36, f"cards overflow: {card_x + card_w}"
    hand = (932 + 59 + 8, 336 - 92)
    for i, (text, color) in enumerate(labels):
        y = 132 + i * 54
        d = f"M{num(hand[0])} {num(hand[1])}C{num(hand[0] + 24)} {num(hand[1])} {num(card_x - 26)} {y} {card_x} {y}"
        add(f'<path d="{d}" fill="none" stroke="{color}" stroke-opacity=".85" stroke-width="2" stroke-linecap="round" class="flow"/>')
        add(f'<rect x="{card_x}" y="{y - card_h / 2}" width="{num(card_w)}" height="{card_h}" rx="12" fill="{color}" fill-opacity=".12" stroke="{color}" stroke-opacity=".6"/>')
        add(f'<path fill="{color}" d="{pill_face.path(text, card_x + 16, y + 5.4, card_size)}"/>')
        add(f'<g class="pop p{i}">{check(card_x + card_w - 22, y, 10, color)}</g>')

    # Captions under the two regions.
    for text, x_mid, color in (("training distribution", cx, BLUE), ("unseen problems", card_x + card_w / 2, "#c9b6ff")):
        add(f'<path fill="{color}" fill-opacity=".85" d="{mono_face.path(text, x_mid - mono_face.width(text, 13, 0.4) / 2, 352, 13, 0.4)}"/>')

    # The robot, with a thought bubble of numbered reasoning steps.
    add(robot(932, 336))
    bubble = "#1a2352"
    add(f'<circle cx="952" cy="132" r="3.4" fill="{bubble}" stroke="#ffffff" stroke-opacity=".22"/>')
    add(f'<circle cx="962" cy="117" r="5.2" fill="{bubble}" stroke="#ffffff" stroke-opacity=".22"/>')
    add(f'<rect x="868" y="50" width="168" height="56" rx="28" fill="{bubble}" stroke="#ffffff" stroke-opacity=".22"/>')
    steps = [902, 940, 978]
    add(f'<path d="M{steps[0]} 78H1012" stroke="#cfe0ff" stroke-opacity=".6" stroke-width="2"/>')
    add("".join(f'<circle cx="{sx}" cy="78" r="11.5" fill="{bubble}"/>' for sx in [*steps, 1012]))
    for k, sx in enumerate(steps):
        digit = str(k + 1)
        add(f'<g class="step s{k}"><circle cx="{sx}" cy="78" r="11.5" fill="#e8f0ff"/>'
            f'<path fill="#0d1328" d="{pill_face.path(digit, sx - pill_face.width(digit, 13) / 2, 82.6, 13)}"/></g>')
    add(f'<g class="step s3">{check(1012, 78, 11.5, GREEN)}</g>')

    defs = f"""<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#0a0f1f"/><stop offset=".55" stop-color="#10163a"/><stop offset="1" stop-color="#1b1238"/>
</linearGradient>
<radialGradient id="glow" cx="900" cy="210" r="440" gradientUnits="userSpaceOnUse">
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
<linearGradient id="robotBody" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#f7f9ff"/><stop offset="1" stop-color="#c4d0f0"/>
</linearGradient>
<filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6"/></filter>
<clipPath id="frame"><rect width="{W}" height="{H}" rx="24"/></clipPath>
</defs>"""

    style = """<style>
.flow{stroke-dasharray:3 6;animation:flow 1.4s linear infinite}
@keyframes flow{to{stroke-dashoffset:-18}}
.halo,.blink,.beat,.pop{transform-box:fill-box;transform-origin:center}
.halo{animation:pulse 2.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.15;transform:scale(.8)}50%{opacity:.45;transform:scale(1.25)}}
.blink{animation:blink 4.5s infinite}
@keyframes blink{0%,93%,100%{transform:scaleY(1)}96%{transform:scaleY(.12)}}
.beat{animation:beat 1.6s ease-in-out infinite}
@keyframes beat{0%,100%{opacity:.55}50%{opacity:1}}
.step{animation:step 3.6s infinite both}
.s1{animation-delay:.35s}.s2{animation-delay:.7s}.s3{animation-delay:1.05s}
@keyframes step{0%,6%{opacity:.3}14%,86%{opacity:1}100%{opacity:.3}}
.pop{animation:pop 3.6s infinite both}
.p0{animation-delay:1.3s}.p1{animation-delay:1.5s}.p2{animation-delay:1.7s}.p3{animation-delay:1.9s}
@keyframes pop{0%,4%{transform:scale(.4);opacity:.3}12%{transform:scale(1.15);opacity:1}18%,86%{transform:scale(1);opacity:1}100%{transform:scale(.4);opacity:.3}}
@media (prefers-reduced-motion:reduce){.flow,.halo,.blink,.beat,.step,.pop{animation:none}}
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
