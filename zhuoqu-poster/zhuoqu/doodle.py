"""Hand-drawn line vocabulary (doodle primitives).

Principles (references/illustration-system.md):
- one near-black pen, constant width, round caps; colour fills are separate, slightly
  offset blobs ("coloured in after drawing, over the line").
- wobble = smooth low-frequency displacement along the path normal (no per-point jitter);
  closed shapes do not close exactly (overshoot / gap at the seam).
- primitives are vocabulary, not pictures: the planner decides what each one *does*.

All coordinates are canvas pixels. A Drawing collects strokes/fills and renders to
RGBA (supersampled) or SVG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .typography import smooth1d


def _resample(pts: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    if L[-1] == 0:
        return pts
    n = max(2, int(L[-1] / step) + 1)
    t = np.linspace(0, L[-1], n)
    return np.stack([np.interp(t, L, pts[:, 0]), np.interp(t, L, pts[:, 1])], 1)


def wobble(pts, amp: float, seed: int, salt, wavelength: float = 120.0, closed: bool = False,
           overshoot: float = 0.0) -> np.ndarray:
    """Displace a polyline along its normal with smooth noise. amp, wavelength in px."""
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    p = _resample(pts, 3.0)
    if closed and overshoot:
        k = max(2, int(overshoot / 3.0))
        p = np.vstack([p, p[1:k + 1]])            # run past the start: hand-drawn seam
    d = np.gradient(p, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    n = smooth1d(seed, ("wob",) + tuple(np.atleast_1d(salt)), wavelength)
    off = np.array([n(x) for x in s]) * amp
    return p + nrm * off[:, None]


@dataclass
class Drawing:
    width: int
    height: int
    seed: int = 0
    pen: float = 6.0                  # px
    ink: str = "#1b1a18"
    amp: float = 1.0                  # global wobble multiplier
    strokes: list = field(default_factory=list)   # (pts, width, color)
    fills: list = field(default_factory=list)     # (pts, color, alpha)
    _n: int = 0

    # ------------------------------------------------ low level
    def _salt(self):
        self._n += 1
        return self._n

    def path(self, pts, closed=False, width=None, color=None, amp=None, wavelength=None, overshoot=None):
        pts = np.asarray(pts, float)
        size = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 10
        a = (amp if amp is not None else 0.012 * size + 0.6) * self.amp
        wl = wavelength or max(40.0, size * 0.7)
        ov = overshoot if overshoot is not None else (0.06 * size if closed else 0)
        q = wobble(pts, a, self.seed, self._salt(), wl, closed, ov)
        self.strokes.append((q, width or self.pen, color or self.ink))
        return q

    def fill(self, pts, color, alpha=1.0, offset=(0.0, 0.0), amp=None):
        pts = np.asarray(pts, float)
        size = float(np.ptp(pts, axis=0).max())
        q = wobble(pts, (amp if amp is not None else 0.03 * size) * self.amp, self.seed, self._salt(), size * 0.5, True)
        self.fills.append((q + np.asarray(offset), color, alpha))
        return q

    # ------------------------------------------------ shapes
    @staticmethod
    def _ellipse(cx, cy, rx, ry, a0=0.0, a1=2 * math.pi, n=64):
        t = np.linspace(a0, a1, n)
        return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)

    def line(self, p0, p1, **kw):
        return self.path([p0, p1], **kw)

    def circle(self, cx, cy, r, **kw):
        a0 = -math.pi / 2 + 0.3 * smooth1d(self.seed, ("c", self._n), 3)(1)
        return self.path(self._ellipse(cx, cy, r, r * 0.96, a0, a0 + 2 * math.pi), closed=True, **kw)

    def rect(self, x0, y0, x1, y1, **kw):
        """Wobbly frame: four separate strokes whose corners overshoot (like 0234)."""
        w, h = x1 - x0, y1 - y0
        o = 0.025 * max(w, h)
        n = smooth1d(self.seed, ("rect", self._salt()), 2.0)
        e = [n(i) * o for i in range(8)]
        self.path([(x0 - o + e[0], y0 + e[1]), (x1 + o, y0 + e[2])], **kw)
        self.path([(x1 + e[3], y0 - o), (x1 + e[4], y1 + o)], **kw)
        self.path([(x1 + o, y1 + e[5]), (x0 - o, y1 + e[6])], **kw)
        self.path([(x0 + e[7], y1 + o), (x0, y0 - o)], **kw)

    def arrow(self, p0, p1, head=None, **kw):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        mid = (p0 + p1) / 2 + np.array([-(p1 - p0)[1], (p1 - p0)[0]]) * 0.12
        t = np.linspace(0, 1, 20)[:, None]
        curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * mid + t ** 2 * p1
        self.path(curve, **kw)
        d = p1 - curve[-4]; d /= np.linalg.norm(d) + 1e-9
        h = head or 0.18 * np.linalg.norm(p1 - p0)
        for ang in (2.6, -2.6):
            c, s = math.cos(ang), math.sin(ang)
            v = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])
            self.path([p1, p1 + v * h], **kw)

    def dotted(self, p0, p1, n=6, **kw):
        """Drip / motion dots along a line (0029, 0235)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = np.linalg.norm(p1 - p0)
        d = (p1 - p0) / (L + 1e-9)
        seg = L / (2 * n)
        for i in range(n):
            a = p0 + d * seg * 2 * i
            self.path([a, a + d * seg * 0.6], **kw)

    def strike(self, x0, y, x1, **kw):
        """Strike-through edit (0201, 0043): slightly rising, overshooting."""
        w = x1 - x0
        self.path([(x0 - 0.05 * w, y + 0.04 * w), (x1 + 0.05 * w, y - 0.03 * w)], **kw)

    def underline(self, x0, y, x1, **kw):
        w = x1 - x0
        self.path([(x0, y), (x0 + w * 0.5, y + w * 0.015), (x1, y - w * 0.01)], **kw)

    # ------------------------------------------------ objects
    def cup(self, cx, base_y, h, lid=True, straw=False, fill=None, **kw):
        """Tapered takeaway cup, bottom centre at (cx, base_y)."""
        tw, bw = h * 0.36, h * 0.27
        top = base_y - h
        body = [(cx - tw, top), (cx - bw, base_y), (cx + bw, base_y), (cx + tw, top)]
        if fill:
            lvl = top + h * 0.25
            k = (lvl - top) / h
            wl = tw - (tw - bw) * k
            self.fill([(cx - wl, lvl), (cx - bw, base_y), (cx + bw, base_y), (cx + wl, lvl)], fill,
                      offset=(h * 0.03, -h * 0.02))
        self.path(body, **kw)
        if lid:
            self.path([(cx - tw * 1.08, top), (cx + tw * 1.08, top)], **kw)
            self.path(self._ellipse(cx, top, tw * 0.9, h * 0.09, math.pi, 2 * math.pi, 24), **kw)
        if straw:
            self.path([(cx + tw * 0.2, top - h * 0.08), (cx + tw * 0.45, top - h * 0.42)], **kw)
        return dict(top=top, left=cx - tw, right=cx + tw)

    def figure(self, x, y, h, pose="stand", facing=1, **kw):
        """Stick figure, feet at (x, y). Poses: stand, push, leap, sit, lean, carry, wave.
        Returns anchor points (hands, head) so other elements can attach precisely."""
        r = h * 0.11
        f = facing
        P = {
            "stand": dict(head=(0, -0.89), hip=(0, -0.42), sh=(0, -0.72), hands=[(-0.2, -0.45), (0.2, -0.45)], feet=[(-0.14, 0), (0.14, 0)]),
            "push": dict(head=(0.22, -0.8), hip=(-0.05, -0.4), sh=(0.14, -0.66), hands=[(0.46, -0.66), (0.46, -0.56)], feet=[(-0.36, 0), (0.08, 0)]),
            "leap": dict(head=(0.16, -1.0), hip=(-0.02, -0.5), sh=(0.1, -0.8), hands=[(-0.12, -1.16), (0.42, -1.06)], feet=[(-0.4, -0.36), (0.12, -0.06)]),
            "sit": dict(head=(0, -0.62), hip=(0, -0.17), sh=(0, -0.46), hands=[(0.24, -0.26), (-0.22, -0.24)], feet=[(0.34, 0), (0.24, 0)]),
            "lean": dict(head=(-0.14, -0.86), hip=(0.04, -0.42), sh=(-0.07, -0.69), hands=[(-0.3, -0.72), (0.24, -0.5)], feet=[(0.02, 0), (0.3, 0)]),
            "carry": dict(head=(0, -0.88), hip=(0, -0.42), sh=(0, -0.71), hands=[(-0.18, -1.02), (0.18, -1.02)], feet=[(-0.16, 0), (0.12, 0)]),
            "wave": dict(head=(0, -0.89), hip=(0, -0.42), sh=(0, -0.72), hands=[(-0.2, -0.46), (0.3, -1.0)], feet=[(-0.14, 0), (0.14, 0)]),
        }[pose]
        tp = lambda p: (x + f * p[0] * h, y + p[1] * h)
        hd, hip, sh = tp(P["head"]), tp(P["hip"]), tp(P["sh"])
        self.circle(hd[0], hd[1], r, **kw)
        neck = (hd[0] + (sh[0] - hd[0]) * 0.45, hd[1] + r * 1.05)
        self.path([neck, sh, hip], **kw)
        hands = [tp(p) for p in P["hands"]]
        for hp in hands:
            el = ((sh[0] + hp[0]) / 2 + f * h * 0.02, (sh[1] + hp[1]) / 2 + h * 0.03)
            self.path([sh, el, hp], **kw)
        for fp in P["feet"]:
            fp = tp(fp)
            kn = ((hip[0] + fp[0]) / 2 + f * h * 0.04, (hip[1] + fp[1]) / 2)
            self.path([hip, kn, fp, (fp[0] + f * h * 0.06, fp[1])], **kw)
        return dict(head=hd, hands=hands, hip=hip)

    def limbs_on(self, box, kind="arms_up", **kw):
        """Give an object arms/legs (personification without a face, cf. 0237).
        box = (x0,y0,x1,y1) of the object. Returns hand positions."""
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        hands = []
        if kind == "arms_up":
            for sx, ex in ((x0 + w * 0.08, x0 - w * 0.35), (x1 - w * 0.08, x1 + w * 0.35)):
                p = [(sx, y0 + h * 0.45), (ex, y0 + h * 0.15), (ex + (sx - ex) * 0.55, y0 - h * 0.3)]
                self.path(p, **kw); hands.append(p[-1])
        elif kind == "walk":
            for sx, d in ((x0 + w * 0.3, -1), (x1 - w * 0.3, 1)):
                self.path([(sx, y1), (sx + d * w * 0.08, y1 + h * 0.18), (sx + d * w * 0.2, y1 + h * 0.2)], **kw)
            for sx, ex in ((x0, x0 - w * 0.3), (x1, x1 + w * 0.3)):
                p = [(sx, y0 + h * 0.5), (ex, y0 + h * 0.7)]
                self.path(p, **kw); hands.append(p[-1])
        return hands

    def hand(self, x, y, s, facing=1, **kw):
        """Minimal pinching hand reaching down (lifting a tea bag, 0235). Fingertip at (x,y)."""
        f = facing
        self.path([(x - f * s * 0.9, y - s * 1.6), (x - f * s * 0.35, y - s * 0.55), (x, y)], **kw)
        self.path([(x - f * s * 0.2, y - s * 1.75), (x + f * s * 0.25, y - s * 0.5), (x + f * s * 0.1, y + s * 0.05)], **kw)

    def cloud(self, cx, cy, w, **kw):
        # upper envelope of a few overlapping bumps -> one continuous scalloped line
        bumps = [(-0.32, 0.02, 0.15), (-0.13, -0.08, 0.19), (0.1, -0.1, 0.2), (0.31, 0.0, 0.15)]
        base = cy + w * 0.08
        xs = np.linspace(cx - w * 0.47, cx + w * 0.46, 120)
        ys = np.full_like(xs, base)
        for bx, by, br in bumps:
            dx = xs - (cx + bx * w)
            ok = np.abs(dx) < br * w
            top = cy + by * w - np.sqrt(np.clip((br * w) ** 2 - dx ** 2, 0, None))
            ys = np.where(ok, np.minimum(ys, top), ys)
        pts = np.stack([xs, ys], 1)
        self.path(np.vstack([pts, [(cx + w * 0.44, base), (cx - w * 0.45, base)]]), closed=True, **kw)

    def tree(self, x, y, h, **kw):
        """0322-style: a cloud canopy and two trunk lines."""
        self.cloud(x, y - h * 0.66, h * 0.8, **kw)
        self.path([(x - h * 0.055, y), (x - h * 0.03, y - h * 0.58)], **kw)
        self.path([(x + h * 0.06, y), (x + h * 0.035, y - h * 0.58)], **kw)
        self.path([(x + h * 0.03, y - h * 0.4), (x + h * 0.12, y - h * 0.5)], **kw)   # one small branch
        self.path([(x - h * 0.3, y), (x + h * 0.35, y)], **kw)

    def sun(self, cx, cy, r, rays=8, **kw):
        self.circle(cx, cy, r, **kw)
        for i in range(rays):
            a = i * 2 * math.pi / rays + 0.2
            self.path([(cx + math.cos(a) * r * 1.4, cy + math.sin(a) * r * 1.4),
                       (cx + math.cos(a) * r * 1.85, cy + math.sin(a) * r * 1.85)], **kw)

    def moon(self, cx, cy, r, fill=None, **kw):
        if fill:
            self.fill(self._ellipse(cx, cy, r, r), fill, offset=(r * 0.12, r * 0.08))
        self.circle(cx, cy, r, **kw)

    def leaf(self, cx, cy, L, angle=0.0, fill=None, **kw):
        t = np.linspace(0, 1, 24)
        up = np.stack([t * L, np.sin(t * math.pi) * L * 0.28], 1)
        lo = np.stack([t[::-1] * L, -np.sin(t[::-1] * math.pi) * L * 0.24], 1)
        pts = np.vstack([up, lo]) - [L / 2, 0]
        c, s = math.cos(angle), math.sin(angle)
        pts = pts @ np.array([[c, s], [-s, c]]) + [cx, cy]
        if fill:
            self.fill(pts, fill, offset=(L * 0.05, L * 0.04))
        self.path(pts, closed=True, **kw)
        a, b = pts[0], pts[23]
        self.path([a, a + (b - a) * 0.85], **kw)

    def teabag(self, x, y, s, **kw):
        """Tea bag hanging from a string whose top is (x, y)."""
        self.path([(x, y), (x + s * 0.03, y + s * 0.9)], **kw)
        self.rect(x - s * 0.18, y + s * 0.9, x + s * 0.2, y + s * 1.05, **kw)
        self.path([(x - s * 0.32, y + s * 1.18), (x - s * 0.34, y + s * 1.75), (x + s * 0.36, y + s * 1.76), (x + s * 0.33, y + s * 1.16)], closed=True, **kw)

    def scribble(self, cx, cy, r, turns=2.5, **kw):
        """Loose motion swirl (0259)."""
        t = np.linspace(0, turns * 2 * math.pi, int(60 * turns))
        rr = r * (0.5 + 0.5 * t / t[-1])
        self.path(np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t) * 0.7], 1), **kw)

    # ------------------------------------------------ produce
    def fruit(self, cx, cy, r, kind="round", fill=None, **kw):
        """Simplified fruit: only the silhouette needed to recognise it (no face).
        kinds: round (orange/peach/apple), persimmon, grape, lemon, strawberry, mango, leaf_tea.
        Returns bbox."""
        off = (r * 0.12, r * 0.08)
        if kind == "grape":
            cs = [(0, -0.55), (-0.42, -0.5), (0.42, -0.5), (-0.22, -0.12), (0.24, -0.12), (0, 0.28), (-0.4, 0.2), (0.38, 0.22), (0, 0.66)]
            rr = r * 0.3
            for i, (u, v) in enumerate(cs):
                if fill:
                    self.fill(self._ellipse(cx + u * r, cy + v * r, rr, rr, 0, 2 * math.pi, 28), fill, offset=off)
            for u, v in cs:
                self.circle(cx + u * r, cy + v * r, rr, **kw)
            self.path([(cx, cy - r * 0.85), (cx + r * 0.08, cy - r * 1.2)], **kw)
            self.leaf(cx + r * 0.38, cy - r * 1.05, r * 0.6, -0.4, **kw)
            return (cx - r * 0.8, cy - r * 1.3, cx + r * 0.8, cy + r)
        if kind == "lemon":
            t = np.linspace(0, 2 * math.pi, 64)
            pts = np.stack([cx + r * 1.15 * np.cos(t) * (1 - 0.18 * np.cos(t) ** 8), cy + r * 0.8 * np.sin(t)], 1)
        elif kind == "strawberry":
            t = np.linspace(0, 2 * math.pi, 64)
            pts = np.stack([cx + r * 0.9 * np.sin(t) * (0.55 + 0.45 * (1 - np.cos(t)) / 2), cy - r * 0.9 * np.cos(t) * 0.95], 1)
        elif kind == "mango":
            t = np.linspace(0, 2 * math.pi, 64)
            pts = np.stack([cx + r * 0.8 * np.cos(t) + r * 0.18 * np.sin(2 * t), cy + r * 1.05 * np.sin(t)], 1)
        elif kind == "persimmon":
            t = np.linspace(0, 2 * math.pi, 64)
            pts = np.stack([cx + r * 1.05 * np.cos(t), cy + r * 0.85 * np.sin(t) + r * 0.06 * np.cos(4 * t)], 1)
        else:
            pts = self._ellipse(cx, cy, r, r * 0.95, -math.pi / 2, 1.5 * math.pi)
        if fill:
            self.fill(pts, fill, offset=off)
        self.path(pts, closed=True, **kw)
        top = float(pts[:, 1].min())
        if kind == "persimmon":
            for a in (-0.9, -0.3, 0.3, 0.9):
                self.path([(cx, top + r * 0.08), (cx + math.sin(a) * r * 0.42, top + r * 0.08 + math.cos(a) * r * 0.18)], **kw)
            self.path([(cx, top + r * 0.05), (cx + r * 0.04, top - r * 0.2)], **kw)
        elif kind == "strawberry":
            for u, v in ((-0.3, -0.1), (0.25, -0.2), (0, 0.25), (-0.2, 0.5), (0.3, 0.35), (0.05, -0.45)):
                self.path([(cx + u * r, cy + v * r), (cx + u * r + 1, cy + v * r + r * 0.07)], **kw)
            self.path([(cx - r * 0.5, top + r * 0.12), (cx, top - r * 0.1), (cx + r * 0.5, top + r * 0.12)], **kw)
        elif kind in ("round", "mango", "lemon"):
            self.path([(cx, top + r * 0.1), (cx + r * 0.08, top - r * 0.25)], **kw)
            if kind != "lemon":
                self.leaf(cx + r * 0.35, top - r * 0.2, r * 0.6, -0.5, **kw)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        return (float(x0), float(y0 - r * 0.4), float(x1), float(y1))

    def cup_tipped(self, cx, base_y, h, angle=1.25, fill=None, **kw):
        """A cup lying on its side (0322 'fallen cup'): rotate the body around its base."""
        tw, bw = h * 0.36, h * 0.27
        body = np.array([(-tw, -h), (-bw, 0), (bw, 0), (tw, -h)], float)
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        q = body @ R.T + [cx, base_y - bw * abs(s) * 0.2]
        if fill:
            self.fill(q, fill, alpha=1.0, offset=(h * 0.03, 0))
        self.path(q, **kw)
        lid = np.array([(-tw * 1.08, -h), (tw * 1.08, -h)]) @ R.T + [cx, base_y - bw * abs(s) * 0.2]
        self.path(lid, **kw)
        x0, y0 = q.min(0); x1, y1 = q.max(0)
        return (float(x0), float(y0), float(x1), float(y1))

    def brush_block(self, x0, y0, x1, y1, color, alpha=0.95):
        """Soft brushed colour field (0236): rectangle with wobbly, slightly feathered edges."""
        pts = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
        self.fill(pts, color, alpha=alpha, amp=0.012 * max(x1 - x0, y1 - y0))

    # ------------------------------------------------ output
    def render(self, ss: int = 3) -> Image.Image:
        W, H = self.width * ss, self.height * ss
        rgba = np.zeros((H, W, 4), np.float32)
        def over(mask, color, alpha):
            c = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
            a = mask.astype(np.float32) / 255 * alpha
            rgba[..., :3] = rgba[..., :3] * (1 - a[..., None]) + c * a[..., None]
            rgba[..., 3] = rgba[..., 3] * (1 - a) + a
        for pts, color, alpha in self.fills:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round(pts * ss * 4).astype(np.int32)], 255, cv2.LINE_AA, 2)
            over(m, color, alpha)
        by_color: dict = {}
        for pts, w, color in self.strokes:
            by_color.setdefault(color, []).append((pts, w))
        for color, items in by_color.items():
            m = np.zeros((H, W), np.uint8)
            for pts, w in items:
                cv2.polylines(m, [np.round(pts * ss * 4).astype(np.int32)], False, 255,
                              max(1, int(round(w * ss))), cv2.LINE_AA, 2)
            over(m, color, 1.0)
        out = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA").resize((self.width, self.height), Image.LANCZOS)

    def svg_elements(self) -> str:
        out = []
        for pts, color, alpha in self.fills:
            d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z"
            out.append(f'<path d="{d}" fill="{color}" fill-opacity="{alpha}"/>')
        for pts, w, color in self.strokes:
            d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts[::2])
            out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w:.1f}" stroke-linecap="round" stroke-linejoin="round"/>')
        return "\n".join(out)
