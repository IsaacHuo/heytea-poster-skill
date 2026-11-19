"""Dual-role typography engine (functional / expressive / hybrid).

Design notes (see references/typography-system.md):
- All variation amplitudes are in *em* units, so they scale with the font, not the canvas.
- Variation is produced by smooth, low-frequency, correlated noise (sum of a few sines),
  never by independent per-point jitter -> no broken strokes, no mechanical shake.
- Rhythm, not randomness: only a few glyphs (tilt_slots) are allowed to lean noticeably;
  baseline is a slow wave along the line; emphasised glyphs are enlarged on purpose.
- Structure protection: complex glyphs (long outline perimeter) get damped warps and
  width changes, because component separation (e.g. 柿 -> 木市) is the main failure mode.
- monoline=True re-draws each glyph with a constant-width round pen along its skeleton
  (marker / felt-pen handwriting). Pen width is constant in *output* pixels, like a real pen,
  even when a glyph is enlarged.
- Every glyph is emitted as its own SVG group (data-char) and the exact text is kept
  in <title>, so output stays traceable and editable.
"""
from __future__ import annotations

import difflib
import json
import math
import re
import zlib
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from PIL import Image

from . import paths

FONTS = {
    "wenkai": "LXGWWenKai-Regular.ttf",
    "noto_sans": "NotoSansSC[wght].ttf",
    "kuaile": "ZCOOLKuaiLe-Regular.ttf",
    "longcang": "LongCang-Regular.ttf",
    "mashanzheng": "MaShanZheng-Regular.ttf",
    "zhimangxing": "ZhiMangXing-Regular.ttf",
    "xiaowei": "ZCOOLXiaoWei-Regular.ttf",
}
FALLBACK_CHAIN = ("wenkai", "noto_sans")


@dataclass
class TextParams:
    font: str = "wenkai"
    weight: int = 400                  # variable fonts only
    rotation_range: float = 0.0        # degrees; max lean for tilt-slot glyphs
    baseline_variation: float = 0.0    # em
    scale_variation: float = 0.0       # fraction of size
    spacing_variation: float = 0.0     # em
    stroke_irregularity: float = 0.0   # em, outline displacement amplitude
    curve_softness: float = 0.0        # 0..1 corner rounding
    width_variation: float = 0.0       # fraction of glyph aspect
    line_break_strategy: str = "none"  # none | balanced | semantic | stagger
    density: float = 0.5               # 0 loose .. 1 tight (tracking + leading)
    legibility_threshold: float = 0.0  # OCR similarity lower bound (0 disables)
    monoline: bool = False
    pen_width: float = 0.07            # em at base size
    pen_variation: float = 0.0         # fraction, slow pressure change
    tilt_slots: float = 0.25           # share of glyphs allowed to lean
    emphasis_scale: float = 1.0
    max_chars_per_line: int = 0        # 0 = auto
    line_height: float = 1.3


# Preset amplitudes are defined at intensity 0.5; intensity scales them linearly.
STYLES: dict[str, dict] = {
    "functional_clean": dict(font="noto_sans", weight=500, density=0.55, legibility_threshold=0.95),
    "functional_warm": dict(font="wenkai", density=0.55, legibility_threshold=0.95),
    "playful_handwritten": dict(font="wenkai", monoline=True, pen_width=0.058, pen_variation=0.3,
                                rotation_range=7, baseline_variation=0.1, scale_variation=0.12,
                                spacing_variation=0.07, stroke_irregularity=0.016, curve_softness=0.5,
                                width_variation=0.07, line_break_strategy="semantic", density=0.3,
                                tilt_slots=0.25, emphasis_scale=1.3, legibility_threshold=0.6),
    "quiet_note": dict(font="wenkai", monoline=True, pen_width=0.055, pen_variation=0.1,
                       rotation_range=2.5, baseline_variation=0.03, scale_variation=0.04,
                       spacing_variation=0.03, stroke_irregularity=0.008, curve_softness=0.4,
                       width_variation=0.03, line_break_strategy="semantic", density=0.25,
                       tilt_slots=0.15, emphasis_scale=1.1, legibility_threshold=0.7),
    "marker_round": dict(font="kuaile", rotation_range=6, baseline_variation=0.07, scale_variation=0.14,
                         spacing_variation=0.06, stroke_irregularity=0.018, curve_softness=0.6,
                         width_variation=0.07, line_break_strategy="semantic", density=0.4,
                         emphasis_scale=1.25, legibility_threshold=0.6),
    "casual_pen": dict(font="longcang", rotation_range=5, baseline_variation=0.07, scale_variation=0.1,
                       spacing_variation=0.05, stroke_irregularity=0.01, curve_softness=0.3,
                       width_variation=0.05, line_break_strategy="semantic", density=0.35,
                       emphasis_scale=1.2, legibility_threshold=0.5),
    "brush_banzi": dict(font="mashanzheng", rotation_range=6, baseline_variation=0.08, scale_variation=0.15,
                        spacing_variation=0.06, stroke_irregularity=0.012, curve_softness=0.3,
                        width_variation=0.06, line_break_strategy="semantic", density=0.4,
                        emphasis_scale=1.25, legibility_threshold=0.5),
}
AMPLITUDE_KEYS = ("rotation_range", "baseline_variation", "scale_variation", "spacing_variation",
                  "stroke_irregularity", "width_variation", "pen_variation")
DEFAULT_STYLE = {"functional": "functional_clean", "expressive": "playful_handwritten",
                 "hybrid": "playful_handwritten"}


class MissingGlyph(ValueError):
    pass


# ---------------------------------------------------------------- noise

def _rng(seed: int, *salt) -> np.random.Generator:
    return np.random.default_rng((int(seed) * 1_000_003 + zlib.crc32(repr(salt).encode())) % (2**32))


def smooth1d(seed: int, salt, period: float):
    """Correlated noise in ~[-1,1]: sum of 3 sines with nearby frequencies."""
    r = _rng(seed, "1d", salt)
    f = (2 * math.pi / period) * r.uniform(0.6, 1.5, 3)
    ph = r.uniform(0, 2 * math.pi, 3)
    a = np.array([1.0, 0.55, 0.3])
    return lambda t: float(np.sum(a * np.sin(f * t + ph)) / a.sum() * 1.6)


def field2d(seed: int, salt, wavelength: float):
    """Smooth 2-D vector field; returns f(x, y) -> (dx, dy) arrays in ~[-1,1]."""
    r = _rng(seed, "2d", salt)
    k = 4
    ang = r.uniform(0, 2 * math.pi, (2, k))
    mag = (2 * math.pi / wavelength) * r.uniform(0.7, 1.4, (2, k))
    ph = r.uniform(0, 2 * math.pi, (2, k))
    def f(x, y):
        out = []
        for c in range(2):
            s = 0
            for j in range(k):
                s = s + np.sin(mag[c, j] * (np.cos(ang[c, j]) * x + np.sin(ang[c, j]) * y) + ph[c, j])
            out.append(s / math.sqrt(k))
        return out
    return f


# ---------------------------------------------------------------- fonts

class _Flatten(BasePen):
    def __init__(self, gs, steps=10):
        super().__init__(gs)
        self.contours, self.cur, self.steps = [], None, steps

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._getCurrentPoint()
        for i in range(1, self.steps + 1):
            t = i / self.steps; u = 1 - t
            self.cur.append((u**3 * p0[0] + 3*u*u*t * p1[0] + 3*u*t*t * p2[0] + t**3 * p3[0],
                             u**3 * p0[1] + 3*u*u*t * p1[1] + 3*u*t*t * p2[1] + t**3 * p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        for i in range(1, self.steps + 1):
            t = i / self.steps; u = 1 - t
            self.cur.append((u*u * p0[0] + 2*u*t * p1[0] + t*t * p2[0],
                             u*u * p0[1] + 2*u*t * p1[1] + t*t * p2[1]))

    def _closePath(self):
        if self.cur and len(self.cur) > 2:
            self.contours.append(np.array(self.cur, dtype=float))
        self.cur = None

    _endPath = _closePath


@lru_cache(maxsize=16)
def _font(key: str, weight: int = 400):
    if key not in FONTS:
        raise ValueError(f"unknown font '{key}', choose from {sorted(FONTS)}")
    p = paths.FONTS / FONTS[key]
    if not p.exists():
        raise FileNotFoundError(f"font file missing: {p} - run: python scripts/fetch_fonts.py")
    tt = TTFont(p)
    gs = tt.getGlyphSet(location={"wght": weight}) if "fvar" in tt else tt.getGlyphSet()
    upm = tt["head"].unitsPerEm
    os2 = tt["OS/2"]
    asc, desc = os2.sTypoAscender / upm, os2.sTypoDescender / upm
    if asc - desc < 0.8:
        asc, desc = tt["hhea"].ascent / upm, tt["hhea"].descent / upm
    return dict(tt=tt, gs=gs, cmap=tt.getBestCmap(), upm=upm, asc=asc, desc=desc, hmtx=tt["hmtx"])


def has_glyph(font: str, ch: str) -> bool:
    return ord(ch) in _font(font)["cmap"]


@lru_cache(maxsize=4096)
def _glyph(font: str, weight: int, ch: str):
    """(contours in em, y-up, advance em, font used). Falls back along FALLBACK_CHAIN."""
    for key in (font,) + tuple(k for k in FALLBACK_CHAIN if k != font):
        f = _font(key, weight if key == font else 400)
        name = f["cmap"].get(ord(ch))
        if name is None:
            continue
        pen = _Flatten(f["gs"])
        f["gs"][name].draw(pen)
        adv = f["hmtx"][name][0] / f["upm"]
        return tuple(c / f["upm"] for c in pen.contours), adv, key
    raise MissingGlyph(f"no font in ({font}, {', '.join(FALLBACK_CHAIN)}) covers {ch!r} U+{ord(ch):04X}")


# ---------------------------------------------------------------- glyph raster ops

def _resample(c: np.ndarray, step: float) -> np.ndarray:
    pts = np.vstack([c, c[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    out = []
    for i, L in enumerate(seg):
        n = max(1, int(math.ceil(L / step)))
        t = np.arange(n)[:, None] / n
        out.append(pts[i] + t * (pts[i + 1] - pts[i]))
    return np.vstack(out)


def _thin(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, vectorised."""
    img = img.astype(np.uint8).copy()
    for _ in range(200):
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            m = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any():
                img[m] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def _prune(sk: np.ndarray, n: int) -> np.ndarray:
    sk = sk.astype(np.uint8).copy()
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    for _ in range(n):
        nb = cv2.filter2D(sk, -1, k, borderType=cv2.BORDER_CONSTANT)
        ends = (sk == 1) & (nb <= 1)
        if not ends.any():
            break
        sk[ends] = 0
    return sk.astype(bool)


@dataclass
class _GlyphOut:
    char: str
    font: str
    contours: list            # list of (N,2) arrays in output px (after transform)
    box: tuple                # (x0,y0,x1,y1)
    transform: dict


def _glyph_shapes(ch, gi, p: TextParams, seed, amp, size_scale, pen_px_em, R, line_box):
    """Warp + raster-process one glyph; return contours in local em (y-down) and advance."""
    contours, adv, used = _glyph(p.font, p.weight, ch)
    if not contours:
        return [], adv, used
    asc, desc = line_box
    m = 0.2
    perim = sum(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum() for c in contours)
    protect = min(1.0, 7.0 / max(perim, 1e-6))            # structure protection for dense glyphs
    pts_all = []
    disp = field2d(seed, ("warp", gi), 0.45)
    a = p.stroke_irregularity * amp * protect
    for c in contours:
        c = _resample(c, 0.008)
        x, yd = c[:, 0], asc - c[:, 1]
        if a > 0:
            dx, dy = disp(x, yd)
            x, yd = x + a * dx, yd + a * dy
        pts_all.append(np.stack([(x + m) * R, (yd + m) * R], 1))
    W, H = int(math.ceil((adv + 2 * m) * R)), int(math.ceil((asc - desc + 2 * m) * R))
    acc = np.zeros((H, W), np.int16)
    for c in pts_all:   # non-zero winding: accumulate signed fills
        tmp = np.zeros((H, W), np.uint8)
        cv2.fillPoly(tmp, [np.round(c * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
        area = 0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
        acc += tmp.astype(np.int16) * (1 if area > 0 else -1)
    mask = (acc != 0).astype(np.float32)
    if p.monoline:
        pre = cv2.GaussianBlur(mask, (0, 0), 0.012 * R) > 0.5
        sk = _prune(_thin(pre), int(0.03 * R))
        dt = cv2.distanceTransform(np.where(sk, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
        yy, xx = np.mgrid[0:H, 0:W] / R
        pv = field2d(seed, ("pen", gi), 0.6)(xx, yy)[0]
        r = 0.5 * pen_px_em * R / size_scale * (1 + p.pen_variation * amp * 0.5 * pv)
        mask = (dt <= r).astype(np.float32)
    sig = max(0.6, p.curve_softness * 0.03 * R)
    mask = cv2.GaussianBlur(mask, (0, 0), sig)
    binm = (mask > 0.5).astype(np.uint8) * 255
    cs, hier = cv2.findContours(binm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 0.3, True).reshape(-1, 2).astype(float)
        if len(c) >= 3:
            out.append(c / R - m)        # local em, y-down from line top
    return out, adv, used


# ---------------------------------------------------------------- line breaking

_PUNCT = "，。！？、；：,.!?;: "


def _wlen(s: str) -> float:
    return sum(0.55 if ord(c) < 0x2E80 else 1.0 for c in s)


def _tokens(para: str) -> list[str]:
    """Latin words (with trailing space) stay whole; each CJK char is its own token."""
    return re.findall(r"[A-Za-z0-9'’\-]+ ?|[^A-Za-z0-9'’\-]", para)


def break_lines(text: str, strategy: str, max_chars: int) -> list[str]:
    lines = []
    for para in text.split("\n"):
        if strategy == "none" or _wlen(para) <= max_chars:
            lines.append(para); continue
        if strategy in ("semantic", "stagger"):
            # break after CJK/Latin punctuation, and at a script change CJK<->Latin
            segs = [x for x in re.split(r"(?<=[，。！？、；：,.!?;:])|(?<=[\u4e00-\u9fff]) (?=[A-Za-z])|(?<=[\u4e00-\u9fff])(?=[A-Za-z])", para) if x]
        else:
            segs = [para]
        for seg in segs:
            seg = seg.strip(" ")
            if _wlen(seg) <= max_chars:
                lines.append(seg); continue
            toks = _tokens(seg)
            n = math.ceil(_wlen(seg) / max_chars)
            target = _wlen(seg) / n
            cur = ""
            for t in toks:
                if cur and _wlen(cur + t) > target + 0.5:
                    lines.append(cur.rstrip(" ")); cur = ""
                cur += t
            if cur.strip():
                lines.append(cur.rstrip(" "))
    merged = []
    for l in lines:
        if merged and _wlen(l.strip(_PUNCT)) <= 1 and _wlen(merged[-1]) + _wlen(l) <= max_chars + 2:
            merged[-1] += l
        else:
            merged.append(l)
    return merged


# ---------------------------------------------------------------- result

@dataclass
class RenderResult:
    text: str
    role: str
    style: str
    intensity: float
    seed: int
    params: dict
    font_size: float
    color: str
    width: int
    height: int
    glyphs: list = field(default_factory=list)   # [{char,font,box,transform}]
    lines: list = field(default_factory=list)
    legibility: Optional[float] = None
    ocr_text: Optional[str] = None
    attempts: int = 1
    ocr_unverifiable: bool = False
    warnings: list = field(default_factory=list)
    _shapes: list = field(default_factory=list, repr=False)

    def svg(self) -> str:
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
               f'viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{_esc(self.text)}">',
               f'<title>{_esc(self.text)}</title>',
               f'<desc>{_esc(json.dumps(dict(role=self.role, style=self.style, intensity=self.intensity, seed=self.seed), ensure_ascii=False))}</desc>',
               f'<g fill="{self.color}" fill-rule="evenodd">']
        for i, (g, cs) in enumerate(zip(self.glyphs, self._shapes)):
            if not cs:
                continue
            d = " ".join("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in c) + " Z" for c in cs)
            out.append(f'<path id="glyph-{i}" data-char="{_esc(g["char"])}" data-font="{g["font"]}" d="{d}"/>')
        out.append("</g></svg>")
        return "\n".join(out)

    def mask(self, scale: float = 1.0, ss: int = 3) -> np.ndarray:
        W, H = int(round(self.width * scale)), int(round(self.height * scale))
        big = np.zeros((H * ss, W * ss), np.uint8)
        for cs in self._shapes:
            if cs:
                tmp = np.zeros_like(big)
                cv2.fillPoly(tmp, [np.round(c * scale * ss * 4).astype(np.int32) for c in cs], 255,
                             lineType=cv2.LINE_8, shift=2)
                np.maximum(big, tmp, out=big)
        return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)

    def image(self, scale: float = 1.0) -> Image.Image:
        a = self.mask(scale)
        rgb = tuple(int(self.color[i:i + 2], 16) for i in (1, 3, 5))
        arr = np.zeros(a.shape + (4,), np.uint8)
        arr[..., :3] = rgb
        arr[..., 3] = a
        return Image.fromarray(arr, "RGBA")

    def meta(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        return d

    def save(self, stem: str | Path, output_format: str = "both") -> dict:
        stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
        out = {}
        if output_format in ("svg", "both"):
            p = stem.with_suffix(".svg"); p.write_text(self.svg(), encoding="utf-8"); out["svg"] = str(p)
        if output_format in ("png", "both"):
            p = stem.with_suffix(".png"); self.image().save(p); out["png"] = str(p)
        p = stem.with_suffix(".json")
        p.write_text(json.dumps(self.meta(), ensure_ascii=False, indent=1), encoding="utf-8"); out["json"] = str(p)
        return out


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------- params

def resolve_params(role: str, style: Optional[str], intensity: float, overrides: Optional[dict]) -> tuple[str, TextParams]:
    if role not in DEFAULT_STYLE:
        raise ValueError(f"role must be functional|expressive|hybrid, got {role!r}")
    style = style or DEFAULT_STYLE[role]
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {sorted(STYLES)}")
    base = dict(STYLES[style])
    f = max(0.0, min(2.0, intensity / 0.5))
    for k in AMPLITUDE_KEYS:
        if k in base:
            base[k] = base[k] * f
    if role == "functional":   # functional text never gets shape variation
        for k in AMPLITUDE_KEYS:
            base[k] = 0.0
        base["curve_softness"] = 0.0
        base["emphasis_scale"] = 1.0
        base["monoline"] = False
        base.setdefault("legibility_threshold", 0.95)
    valid = {fl.name for fl in fields(TextParams)}
    for k, v in (overrides or {}).items():
        if k not in valid:
            raise ValueError(f"unknown text parameter {k!r}")
        base[k] = v
    return style, TextParams(**base)


def _auto_emphasis(text: str) -> set[int]:
    idx = set()
    for m in re.finditer(r"[“\"'「『‘]([^”\"'」』’]{1,3})[”\"'」』’]", text):
        idx.update(range(m.start(1), m.end(1)))
    return idx


# ---------------------------------------------------------------- main API

def _layout(text, role, p: TextParams, seed, font_size, layout, emphasis, protect_idx):
    amp = 1.0
    f0 = _font(p.font, p.weight)
    line_box = (f0["asc"], f0["desc"])
    lh = (f0["asc"] - f0["desc"])
    R = int(min(512, max(256, font_size * 1.3)))
    vertical = layout == "vertical"
    straight = layout == "straight"
    max_chars = p.max_chars_per_line or (8 if role != "functional" else 18)
    strategy = "stagger" if layout == "stagger" else p.line_break_strategy
    lines = break_lines(text, strategy, max_chars)
    n_scale = smooth1d(seed, "scale", 3.3)
    n_width = smooth1d(seed, "width", 2.7)
    n_rot = smooth1d(seed, "rot", 2.2)
    n_slot = smooth1d(seed, "slot", 1.7)
    n_base = smooth1d(seed, "base", 4.0)
    n_base2 = smooth1d(seed, "base2", 1.3)
    n_space = smooth1d(seed, "space", 2.1)
    # monoline/thinned glyphs keep the font advance but lose side-bearing ink, so handwriting needs
    # near-zero or slightly negative tracking to read as one written phrase (probe: 0.11em looked letter-spaced)
    tracking = (1 - p.density) * 0.1 - 0.06 if role != "functional" else (0.5 - p.density) * 0.1
    leading = p.line_height + (1 - p.density) * 0.25
    # pick tilt slots: glyphs with the largest |slot noise| (rhythm, not uniform jitter)
    chars_idx = [i for i, c in enumerate(text) if c not in "\n "]
    slot_val = {i: abs(n_slot(i)) for i in chars_idx}
    k_slots = int(round(p.tilt_slots * len(chars_idx)))
    slots = set(sorted(chars_idx, key=lambda i: -slot_val[i])[:k_slots]) if p.rotation_range > 0 else set()
    glyphs, shapes = [], []
    gi_text = 0
    pen_em = p.pen_width
    for li, line in enumerate(lines):
        # find where this line starts in the text (to keep global index for emphasis/protect)
        start = text.find(line, gi_text)
        start = gi_text if start < 0 else start
        gi_text = start + len(line)
        cursor = 0.0
        if strategy == "stagger" and li % 2 == 1:
            cursor = 0.6 + 0.25 * n_space(li * 7.0)
        for j, ch in enumerate(line):
            gi = start + j
            if ch == " ":
                cursor += 0.3; continue
            prot = 0.35 if gi in protect_idx else 1.0
            s = 1 + p.scale_variation * n_scale(gi) * 0.5
            if gi in emphasis:
                s *= p.emphasis_scale
            w = 1 + p.width_variation * n_width(gi) * prot
            if gi in slots:
                rot = p.rotation_range * (1 if n_slot(gi) > 0 else -1) * prot
            else:
                rot = p.rotation_range * 0.25 * n_rot(gi) * prot
            pos = cursor + 0.5
            bl = p.baseline_variation * (0.75 * n_base(pos / 1.0) + 0.25 * n_base2(gi))
            if straight:
                rot, bl = 0.0, 0.0
            cs, adv, used = _glyph_shapes(ch, gi, p, seed, amp * prot, s, pen_em, R, line_box)
            if ch in "，。、；：,.;:" :
                s = min(s, 1.0)
            latin = ord(ch) < 0x2E80
            cjk_punct = ch in "，。、；：！？"
            sp = (tracking * 0.2 + p.spacing_variation * 0.3 * n_space(gi)) if latin else tracking + p.spacing_variation * n_space(gi)
            ink_cx = adv / 2
            if cs and not latin and not cjk_punct and role != "functional":
                xs = np.concatenate([c[:, 0] for c in cs])
                ink_cx = float(xs.min() + xs.max()) / 2
                gw = min(adv, float(xs.max() - xs.min()) + 0.16) * s * w
            else:
                gw = adv * s * w
            if cjk_punct:
                sp *= 0.3
            # anchor: glyph centre; layout positions in em
            if vertical:
                cx = -(li * leading) + bl
                cy = cursor + lh * s / 2
                step = lh * s * 0.92 + sp
            else:
                cx = cursor + gw / 2
                cy = li * leading * 1.0 + lh / 2 - (s - 1) * lh * 0.35 + bl
                step = gw + sp
                if cjk_punct:
                    cx -= gw * 0.22
                    step = gw * 0.55 + sp
            th = math.radians(rot)
            cth, sth = math.cos(th), math.sin(th)
            gc = np.array([ink_cx, lh / 2])
            out = []
            for c in cs:
                q = (c - gc) * np.array([s * w, s])
                q = np.stack([q[:, 0] * cth - q[:, 1] * sth, q[:, 0] * sth + q[:, 1] * cth], 1)
                out.append(q + np.array([cx, cy]))
            shapes.append(out)
            glyphs.append(dict(char=ch, font=used, index=gi, line=li, transform=dict(
                x=round(cx, 4), y=round(cy, 4), scale=round(s, 4), width=round(w, 4), rotate=round(rot, 3),
                baseline=round(bl, 4))))
            cursor += step
    # to pixels
    allp = [q for cs in shapes for q in cs]
    if not allp:
        raise ValueError("text renders no visible glyphs")
    P = np.vstack(allp)
    pad = 0.12
    x0, y0 = P.min(0) - pad
    x1, y1 = P.max(0) + pad
    W = int(math.ceil((x1 - x0) * font_size)); H = int(math.ceil((y1 - y0) * font_size))
    px = [[(c - [x0, y0]) * font_size for c in cs] for cs in shapes]
    for g, cs in zip(glyphs, px):
        if cs:
            q = np.vstack(cs)
            g["box"] = [round(float(v), 1) for v in (*q.min(0), *q.max(0))]
    return lines, glyphs, px, W, H


_ocr = None


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", s.lower())


def _reading_order(res):
    """Group OCR boxes into rows when their vertical spans overlap >40%, rows top-down,
    boxes left-right. Wavy baselines split one line into offset boxes; a fixed y-bucket
    sort would scramble them (observed: 今天喝点好的 -> 喝易点好的今天)."""
    items = []
    for r in res:
        ys = [p[1] for p in r[0]]; xs = [p[0] for p in r[0]]
        items.append((min(ys), max(ys), min(xs), r))
    items.sort(key=lambda t: t[0])
    rows = []
    for it in items:
        for row in rows:
            y0, y1 = row["y0"], row["y1"]
            ov = min(y1, it[1]) - max(y0, it[0])
            if ov > 0.4 * min(y1 - y0, it[1] - it[0]):
                row["items"].append(it); row["y0"] = min(y0, it[0]); row["y1"] = max(y1, it[1]); break
        else:
            rows.append(dict(y0=it[0], y1=it[1], items=[it]))
    rows.sort(key=lambda r: r["y0"])
    return [it[3] for row in rows for it in sorted(row["items"], key=lambda t: t[2])]


def _join(items) -> str:
    """Concatenate OCR boxes; when two boxes on a row overlap horizontally the detector
    often reads the seam glyph twice (observed 以门店 -> 以以门店), so drop a repeated
    seam character only in that overlapping case."""
    out, prev = "", None
    for r in items:
        t = r[1]
        xs = [p[0] for p in r[0]]
        if prev is not None and out and t and min(xs) < prev[1] and t[0] == out[-1]:
            t = t[1:]
        out += t
        prev = (min(xs), max(xs))
    return out


def ocr_similarity(img: Image.Image, text: str) -> tuple[float, str]:
    """OCR the rendered text on white and compare with the intended text."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    # The DB text detector needs a margin proportional to glyph size: with a fixed 20px pad
    # a clean 12月5日 at 110px read as 12月515日 at every scale; 60px+ read it exactly.
    pad = max(24, int(0.5 * min(img.height, 320)))
    bg = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), "white")
    bg.paste(img.convert("RGBA"), (pad, pad), img.convert("RGBA"))
    base = np.asarray(bg)
    a = _norm(text)
    if not a:
        return 1.0, ""
    best = (-1.0, "")
    # OCR detectors are scale sensitive (observed 12月5日 -> 12月515日 at one size only):
    # read at two scales and keep the better reading. Never more lenient than one honest read.
    for target in (900, 420):
        arr = base
        h = arr.shape[0]
        if h > target or target == 420:
            s = min(1.0, target / h) if target == 900 else min(1.6, max(0.5, target / max(h, 1)))
            arr = cv2.resize(base, (max(8, int(base.shape[1] * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        res, _ = _ocr(arr[..., ::-1])
        got = _join(_reading_order(res)) if res else ""
        sim = round(difflib.SequenceMatcher(None, a, _norm(got)).ratio(), 3)
        if sim > best[0]:
            best = (sim, got)
        if sim >= 0.999:
            break
    return best


_baseline_cache: dict = {}


def ocr_baseline(text: str) -> float:
    """OCR similarity of a clean functional rendering of the same text. If OCR cannot read
    even that (rare glyphs like 龘靐), a low score on the styled text says nothing about
    our deformation - the text is OCR-unverifiable and needs human review instead."""
    if text not in _baseline_cache:
        r = render_text(text, role="functional", style="functional_clean", font_size=96, check_legibility=False)
        _baseline_cache[text] = ocr_similarity(r.image(), text)[0]
    return _baseline_cache[text]


def render_text(text: str, role: str = "expressive", style: Optional[str] = None, intensity: float = 0.5,
                layout: str = "organic", seed: int = 0, output_format: str = "svg", font_size: float = 160,
                color: str = "#1b1a18", emphasis: Optional[list[int]] = None, protect: Optional[list[str]] = None,
                overrides: Optional[dict] = None, check_legibility: bool = True, max_attempts: int = 4,
                out_path: Optional[str | Path] = None) -> RenderResult:
    """Render text in a typographic role.

    role: functional (exact, clean) | expressive (lettering) | hybrid (expressive, but
    'protect' substrings - e.g. a product name - get damped deformation).
    layout: organic | straight | stagger | vertical.
    If OCR similarity < legibility_threshold the intensity is reduced (x0.7) and re-rendered.
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    if layout not in ("organic", "straight", "stagger", "vertical"):
        raise ValueError(f"unknown layout {layout!r}")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError("color must be #rrggbb")
    protect_idx = set()
    for sub in protect or []:
        for m in re.finditer(re.escape(sub), text):
            protect_idx.update(range(m.start(), m.end()))
        if sub not in text:
            raise ValueError(f"protected substring {sub!r} not found in text")
    if role == "hybrid" and not protect:
        protect_idx = set(range(len(text)))
    emph = set(emphasis) if emphasis is not None else _auto_emphasis(text)
    cur = intensity
    warnings = []
    res = None
    for attempt in range(1, max_attempts + 1):
        sname, p = resolve_params(role, style, cur, overrides)
        lines, glyphs, shapes, W, H = _layout(text, role, p, seed, font_size, layout, emph, protect_idx)
        fb = sorted({g["font"] for g in glyphs if g["font"] != p.font})
        res = RenderResult(text=text, role=role, style=sname, intensity=round(cur, 3), seed=seed,
                           params=asdict(p), font_size=font_size, color=color, width=W, height=H,
                           glyphs=glyphs, lines=lines, attempts=attempt, warnings=list(warnings), _shapes=shapes)
        if fb:
            res.warnings.append(f"fallback font used for some glyphs: {fb}")
        if not check_legibility or p.legibility_threshold <= 0:
            break
        probe = res
        if layout == "vertical":   # OCR is unreliable on vertical CJK; probe the same params horizontally
            _, g2, s2, W2, H2 = _layout(text, role, p, seed, font_size, "organic", emph, protect_idx)
            probe = RenderResult(text=text, role=role, style=sname, intensity=cur, seed=seed, params={},
                                 font_size=font_size, color=color, width=W2, height=H2, glyphs=g2, _shapes=s2)
        scale = min(1.0, 96 / font_size)
        res.legibility, res.ocr_text = ocr_similarity(probe.image(scale), text)
        if res.legibility >= p.legibility_threshold:
            break
        if attempt == 1 and role != "functional":
            b = ocr_baseline(text)
            if b < p.legibility_threshold:
                res.ocr_unverifiable = True
                res.warnings.append(f"ocr_unverifiable: OCR reads clean rendering at only {b}; "
                                    f"styled {res.legibility} - human check required, intensity kept")
                break
        if role == "functional" or attempt == max_attempts:
            res.warnings.append(f"legibility {res.legibility} < threshold {p.legibility_threshold}")
            break
        warnings.append(f"attempt {attempt}: legibility {res.legibility} at intensity {round(cur, 3)}; reducing")
        cur *= 0.7
    if out_path:
        res.save(out_path, output_format)
    return res


# ---------------------------------------------------------------- natural-language control

_RULES = [
    # (pattern, effect)
    (r"更随意|随意一点|随手|潦草一点|更乱|放开", dict(intensity=+0.15)),
    (r"像随手画|随手画出来", dict(intensity=+0.1, overrides=dict(monoline=True, stroke_irregularity_mul=1.4))),
    (r"不要太幼稚|别太幼稚|成熟一点|不幼稚", dict(style_avoid=("marker_round",), overrides=dict(scale_variation_mul=0.6, emphasis_scale=1.12))),
    (r"安静|克制|减少.*夸张|收一点|稳重", dict(intensity_mul=0.5, style="quiet_note")),
    (r"俏皮|活泼|轻快|欢快", dict(intensity=+0.1, layout="stagger")),
    (r"更有拙趣|拙一点|更拙", dict(intensity=+0.12, overrides=dict(width_variation_mul=1.3))),
    (r"清晰|看得清|易读", dict(intensity_mul=0.75)),
    (r"圆润|柔软|软软", dict(overrides=dict(curve_softness=0.8), style_prefer="marker_round")),
    (r"竖排|竖着", dict(layout="vertical")),
    (r"毛笔|草台|围挡", dict(style="brush_banzi")),
]


_OTHER = r"其他|其余|信息|正文|小字|辅助|说明|日期|价格|时间|地址"
_TITLE = r"标题|主标|大字|题目"


def _clauses_for(instruction: str, target: str) -> list[str]:
    """Split on clause punctuation and keep the clauses addressed to target.
    '标题更有拙趣感，其他信息保持清晰' -> title gets clause 1 only, info gets clause 2 only.
    Clauses naming neither apply to both."""
    out = []
    for c in re.split(r"[，,。；;、\n]|但是|但|而", instruction):
        if not c.strip():
            continue
        to_title, to_other = bool(re.search(_TITLE, c)), bool(re.search(_OTHER, c))
        if target == "title" and to_other and not to_title:
            continue
        if target == "other" and to_title and not to_other:
            continue
        out.append(c)
    return out


def interpret_instruction(instruction: str, style: Optional[str] = None, intensity: float = 0.5,
                          layout: str = "organic", role: str = "expressive", target: str = "title") -> dict:
    """Map a natural-language request onto engine controls. Returns a dict usable as
    render_text(**kwargs) plus 'notes' explaining every change (traceable, not magic).
    target: 'title' (headline lettering) or 'other' (subtitle / notes); clauses addressed
    to the other target are ignored and listed in notes."""
    style = style or DEFAULT_STYLE[role]
    sname, base = resolve_params(role, style, 0.5, None)
    over: dict = {}
    notes = []
    clauses = _clauses_for(instruction, target)
    skipped = [c for c in re.split(r"[，,。；;、\n]", instruction) if c.strip() and c not in clauses]
    if skipped:
        notes.append(f"not applied to {target}: {skipped}")
    scoped = "，".join(clauses)
    for pat, eff in _RULES:
        if not re.search(pat, scoped):
            continue
        notes.append(f"'{re.search(pat, scoped).group(0)}' -> {eff}")
        intensity += eff.get("intensity", 0)
        intensity *= eff.get("intensity_mul", 1)
        if "layout" in eff:
            layout = eff["layout"]
        if "style" in eff:
            style = eff["style"]
        if eff.get("style_prefer") and style != "quiet_note":
            style = eff["style_prefer"]
        if style in eff.get("style_avoid", ()):
            style = "playful_handwritten"
        for k, v in eff.get("overrides", {}).items():
            if k.endswith("_mul"):
                key = k[:-4]
                ref = STYLES[style].get(key, getattr(base, key)) * max(0.0, min(2.0, intensity / 0.5))
                over[key] = ref * v
            else:
                over[k] = v
    intensity = round(max(0.0, min(1.0, intensity)), 3)
    return dict(style=style, intensity=intensity, layout=layout, role=role, overrides=over, notes=notes)
