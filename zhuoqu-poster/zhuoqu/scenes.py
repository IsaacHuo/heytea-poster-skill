"""Procedural scene recipes: each one stages a *story* (references/creative-strategy.md),
not a decoration. A recipe draws into a Drawing inside the subject box and returns
photo placements plus anchor boxes.

The relation between elements is computed precisely (hands touch the cup edge, the
fruit sits between the raised hands) while the line itself wobbles - "precise relation,
casual form" is what makes simple doodles look finished (illustration-system.md 2).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


def _mix(c1: str, c2: str, t: float) -> str:
    """Blend hex colour c1 toward c2 by t (0 = c1, 1 = c2)."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def _cup_edge_x(cx, base, h, y, side=-1):
    tw, bw = h * 0.36, h * 0.27
    k = (base - y) / h
    return cx + side * (bw + (tw - bw) * k)


def _subject_mark(d, cx, cy, r, ctx, fill=True):
    """Draw the product's subject (fruit / leaf / moon) as a simplified silhouette."""
    kind, kw, acc = ctx["fruit_kind"], ctx.get("subject_keyword") or "", ctx["accents"]
    col = acc[0] if (fill and acc) else None
    if kind:
        return d.fruit(cx, cy, r, kind, fill=col)
    if any(t in kw for t in ("茶", "龙井", "茉莉", "抹茶")):
        d.leaf(cx, cy, r * 2.0, -0.5, fill=col)
        return (cx - r, cy - r * 0.6, cx + r, cy + r * 0.6)
    d.moon(cx, cy, r, fill=col)
    return (cx - r, cy - r, cx + r, cy + r)


def push_giant_cup(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh * 0.92, bw / 1.02)
    ground = y1
    cx = x1 - h * 0.38
    acc = ctx["accents"]
    d.cup(cx, ground, h, straw=True, fill=acc[0] if acc else None)
    fh = h * 0.42
    hand_y = ground - 0.61 * fh
    edge = _cup_edge_x(cx, ground, h, hand_y, -1)
    fx = edge - 0.46 * fh + d.pen * 0.4
    d.figure(fx, ground, fh, pose="push", facing=1)
    for i, yy in enumerate((0.55, 0.75)):
        a = fx - fh * (0.55 + 0.1 * i)
        d.line((a - fh * 0.25, ground - yy * fh), (a, ground - yy * fh))
    d.line((fx - fh * 0.5, ground + d.pen), (cx + h * 0.45, ground + d.pen * 0.5))
    return {}


def cup_lifts_fruit(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh * 0.46, bw * 0.6)
    cx, base = (x0 + x1) / 2, y1 - bh * 0.04
    tw = h * 0.36
    top = base - h
    d.cup(cx, base, h, lid=True)
    hands = d.limbs_on((cx - tw, top, cx + tw, base), "arms_up")
    hy = min(p[1] for p in hands)
    # the fruit must rest *in* the hands: hands sit on its lower flanks (~30deg below centre)
    half = abs(hands[1][0] - hands[0][0]) / 2
    r = min(half / 0.86, (hy - y0) / 1.5, bw * 0.3)
    _subject_mark(d, cx, hy - r * 0.5, r, ctx)
    for s in (-1, 1):   # tiny effort marks beside the cup
        d.line((cx + s * tw * 1.9, base - h * 0.15), (cx + s * tw * 2.15, base - h * 0.22))
    return {}


def fallen_cup_tree(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    th = min(bh * 0.88, bw * 0.62)
    tx = x0 + bw * 0.3
    d.tree(tx, y1, th)
    ch = th * 0.28
    acc = ctx["accents"]
    bx = d.cup_tipped(tx + th * 0.5, y1, ch, angle=1.45)
    if acc:   # the spill tells the story (cup knocked over while someone dozed)
        d.fill(d._ellipse(bx[2] + ch * 0.45, y1 - ch * 0.04, ch * 0.55, ch * 0.09), acc[0], alpha=0.9)
    d.line((bx[2] + ch * 0.1, bx[1] + ch * 0.05), (bx[2] + ch * 0.5, bx[1] - ch * 0.35))   # straw rolled out
    d.path([(tx + th * 0.3, y1), (min(x1, bx[2] + ch * 1.1), y1 + ch * 0.02)])   # same ground as the tree
    d.leaf(tx + th * 0.12, y1 - th * 0.22, th * 0.08, 0.9)
    return {}


def drip_into_cup(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh * 0.42, bw * 0.62)
    cx, base = (x0 + x1) / 2, y1
    acc = ctx["accents"]
    # the drink is the fruit, diluted: tint of the subject colour, not an unrelated secondary
    d.cup(cx, base, h, fill=_mix(acc[0], ctx["paper"], 0.42) if acc else None)
    r = min(bh * 0.13, bw * 0.17)
    fy = y0 + bh * 0.06 + r * 1.3
    bb = _subject_mark(d, cx, fy, r, ctx)
    top = base - h * 1.08
    span = top - bb[3]
    for k, t in enumerate((0.28, 0.58, 0.86)):   # three drops, growing as they fall
        dr = r * (0.07 + 0.035 * k)
        yy = bb[3] + span * t
        d.fill(d._ellipse(cx, yy, dr, dr * 1.35), acc[0] if acc else ctx["ink"], alpha=1.0, amp=0)
    return {}


def leap_on_cup(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh * 0.56, bw * 0.7)
    cx, base = x0 + bw * 0.5, y1
    acc = ctx["accents"]
    top = base - h
    fh = min(bh * 0.36, (top - y0) / 1.15)
    fx, fy = cx - h * 0.05, top - fh * 0.12
    if acc:
        # the burst sits *behind and below* the jump (where the energy came from), not on the body
        d.fill(d._ellipse(fx - fh * 0.42, fy - fh * 0.3, fh * 0.42, fh * 0.34), acc[0], alpha=0.95)
    d.cup(cx, base, h, straw=False)
    d.figure(fx, fy, fh, pose="leap", facing=1)
    for s in (-1, 1):
        d.line((fx + s * fh * 0.62, fy - fh * 1.05), (fx + s * fh * 0.78, fy - fh * 1.22))
    return {}


def color_field(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    acc = ctx["accents"] or ["#e9b44c", "#6f8f4e"]
    a2 = acc[1] if len(acc) > 1 else acc[0]
    d.brush_block(x0 + bw * 0.04, y0 + bh * 0.02, x1 - bw * 0.06, y0 + bh * 0.56, acc[0])
    d.brush_block(x0 + bw * 0.10, y0 + bh * 0.62, x1 - bw * 0.02, y1 - bh * 0.02, a2, alpha=0.9)
    ch = bh * 0.12
    d.cup(x1 - bw * 0.16, y1 - bh * 0.08, ch, width=d.pen * 0.9)   # ink line: paper-colour line vanished on light fields
    return {}


def lettering_lead(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh * 0.8, bw * 0.5)
    cx = x0 + bw * 0.62
    acc = ctx["accents"]
    d.cup(cx, y1, h, straw=True, fill=acc[0] if acc else None)
    fh = h * 0.95
    edge = _cup_edge_x(cx, y1, h, y1 - 0.72 * fh, -1)
    d.figure(edge - fh * 0.3, y1, fh, pose="lean", facing=-1)
    return {}


def _fit(img: Image.Image, box, fill: float = 1.0):
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * fill, (y1 - y0) * fill
    s = min(bw / img.width, bh / img.height)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    return img.resize((w, h), Image.LANCZOS)


def photo_frame(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ph = _fit(ctx["photo"], (x0, y0, x1 - bw * 0.14, y1 - bh * 0.04), 0.86)
    px = int(x0 + (bw * 0.86 - ph.width) / 2)
    py = int(y0 + (bh - ph.height) / 2)
    pad = 0.04 * min(ph.width, ph.height)
    d.rect(px - pad, py - pad, px + ph.width + pad, py + ph.height + pad)
    fh = min(bh * 0.28, ph.height * 0.45)
    fx = px + ph.width + pad + fh * 0.3
    d.figure(fx, py + ph.height + pad, fh, pose="lean", facing=1)
    return {"photos": [(ph, (px, py))]}


def photo_push(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ph = _fit(ctx["photo"], (x0 + bw * 0.3, y0, x1, y1), 1.0)
    px, py = int(x1 - ph.width), int(y1 - ph.height)
    fh = min(bh * 0.4, ph.height * 0.55, bw * 0.4)
    fx = px - 0.46 * fh + d.pen * 0.4
    d.figure(fx, y1, fh, pose="push", facing=1)
    for i, yy in enumerate((0.5, 0.72)):
        a = fx - fh * (0.55 + 0.1 * i)
        d.line((a - fh * 0.25, y1 - yy * fh), (a, y1 - yy * fh))
    d.line((fx - fh * 0.5, y1 + d.pen), (x1, y1 + d.pen * 0.5))
    return {"photos": [(ph, (px, py))]}


RECIPES = {f.__name__: f for f in (push_giant_cup, cup_lifts_fruit, fallen_cup_tree, drip_into_cup, leap_on_cup,
                                   color_field, lettering_lead, photo_frame, photo_push)}
