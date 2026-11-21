"""Parametric layout strategies (references/composition-system.md 2).

A strategy is a set of relative boxes (title / subject / info) plus alignment, not a
pixel template: boxes are jittered by seed inside safe ranges and adapted to the canvas
aspect (tall / square / wide). Text boxes and the subject box are disjoint by
construction; validate.py checks the *actual* ink boxes afterwards.
"""
from __future__ import annotations

import numpy as np

TALL = {
    "subject_center": dict(title=(0.10, 0.07, 0.90, 0.24), subject=(0.16, 0.31, 0.84, 0.84), info=(0.10, 0.89, 0.90, 0.95), align="center"),
    "subject_offset": dict(title=(0.08, 0.08, 0.64, 0.30), subject=(0.24, 0.37, 0.92, 0.86), info=(0.08, 0.90, 0.70, 0.95), align="left"),
    "negative_space": dict(title=(0.09, 0.09, 0.54, 0.30), subject=(0.40, 0.55, 0.90, 0.86), info=(0.09, 0.90, 0.62, 0.95), align="left"),
    "lettering_lead": dict(title=(0.08, 0.14, 0.88, 0.54), subject=(0.46, 0.58, 0.90, 0.86), info=(0.08, 0.90, 0.70, 0.95), align="left"),
    "vertical_title": dict(title=(0.72, 0.08, 0.90, 0.62), subject=(0.08, 0.34, 0.66, 0.86), info=(0.08, 0.90, 0.66, 0.95), align="right", vertical=True),
    "color_field": dict(title=(0.10, 0.07, 0.90, 0.24), subject=(0.12, 0.31, 0.88, 0.85), info=(0.10, 0.89, 0.90, 0.95), align="center"),
}
WIDE = {
    "subject_center": dict(title=(0.06, 0.16, 0.42, 0.60), subject=(0.48, 0.10, 0.94, 0.88), info=(0.06, 0.78, 0.42, 0.88), align="left"),
    "subject_offset": dict(title=(0.06, 0.12, 0.40, 0.52), subject=(0.46, 0.14, 0.94, 0.90), info=(0.06, 0.80, 0.40, 0.90), align="left"),
    "negative_space": dict(title=(0.06, 0.12, 0.36, 0.42), subject=(0.60, 0.44, 0.94, 0.90), info=(0.06, 0.82, 0.40, 0.90), align="left"),
    "lettering_lead": dict(title=(0.06, 0.14, 0.64, 0.70), subject=(0.70, 0.46, 0.94, 0.88), info=(0.06, 0.80, 0.60, 0.90), align="left"),
    "vertical_title": dict(title=(0.80, 0.10, 0.94, 0.80), subject=(0.10, 0.12, 0.74, 0.90), info=(0.10, 0.92, 0.74, 0.97), align="right", vertical=True),
    "color_field": dict(title=(0.06, 0.16, 0.42, 0.60), subject=(0.48, 0.10, 0.94, 0.88), info=(0.06, 0.78, 0.42, 0.88), align="left"),
}


def solve(strategy: str, W: int, H: int, seed: int = 0, jitter: float = 0.02) -> dict:
    table = WIDE if W / H > 1.25 else TALL
    if strategy not in table:
        raise ValueError(f"unknown layout strategy {strategy!r}; choose from {sorted(table)}")
    spec = table[strategy]
    rng = np.random.default_rng(seed + 7)
    out = dict(strategy=strategy, align=spec["align"], vertical=spec.get("vertical", False))
    # one shared shift keeps the boxes' relation (and their gaps) intact
    sx, sy = rng.uniform(-jitter, jitter, 2)
    for k in ("title", "subject", "info"):
        x0, y0, x1, y1 = spec[k]
        g = rng.uniform(-jitter / 2, jitter / 2, 2)
        x0, x1 = x0 + sx + g[0], x1 + sx - g[0]
        y0, y1 = y0 + (sy if k != "info" else 0) + g[1], y1 + (sy if k != "info" else 0) - g[1]
        x0, x1 = max(0.05, x0), min(0.95, x1)
        y0, y1 = max(0.04, y0), min(0.96, y1)
        out[k] = (round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H))
    return out


def place(w: int, h: int, box, align: str = "center", valign: str = "top") -> tuple[int, int]:
    x0, y0, x1, y1 = box
    x = {"left": x0, "center": (x0 + x1 - w) // 2, "right": x1 - w}[align]
    y = {"top": y0, "middle": (y0 + y1 - h) // 2, "bottom": y1 - h}[valign]
    return int(x), int(y)
