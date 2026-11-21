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


def _detail_ink(d, ctx):
    return _mix(ctx["accents"][0], d.ink, 0.46) if ctx["accents"] else d.ink


def _steam(d, x, y, h, color):
    for dx, dy, scale in [(-.12, 0, 1), (.12, -.04, .66)]:
        px, py = x + dx * h, y + dy * h
        s = scale * h
        d.path(_cubic((px, py), (px+s*.46, py-s*.32), (px-s*.42, py-s*.63), (px, py-s)),
               width=d.pen * .54, color=color, amp=1.0)


def _ticks(d, x, y, s, color, side=1):
    for dx, dy, vx, vy in [(0, 0, .35, -.55), (.29, .24, .57, -.28), (.35, .52, .48, .05)]:
        d.line((x + side*dx*s, y + dy*s), (x + side*(dx+vx)*s, y + (dy+vy)*s),
               width=d.pen * .54, color=color, amp=1.0)


def _subject_mark(d, cx, cy, r, ctx, fill=True):
    """Draw the product's subject (fruit / leaf / moon) as a simplified silhouette."""
    kind, kw, acc = ctx["fruit_kind"], ctx.get("subject_keyword") or "", ctx["accents"]
    col = acc[0] if (fill and acc) else None
    if any(name in ctx["theme"] for name in ("蘑菇", "菌菇", "牛肝菌")):
        cap = d._ellipse(cx, cy, r, r*.68, math.pi, 2*math.pi)
        if col:
            d.fill(cap, _mix(col, "#6c412e", .38), amp=2.0)
        d.path(cap, closed=True, overshoot=0)
        stem = [(cx-.31*r, cy), (cx-.28*r, cy+.56*r),
                (cx+.19*r, cy+.55*r), (cx+.28*r, cy)]
        d.path(stem)
        for k in (-.65, -.35, .35, .62):
            d.line((cx+k*r, cy-r*.13), (cx+(k+.12)*r, cy-r*.10), width=d.pen*.48, color=_detail_ink(d, ctx))
        return (cx-r, cy-r*.68, cx+r, cy+r*.56)
    if kind:
        return d.fruit(cx, cy, r, kind, fill=col)
    if any(t in kw for t in ("茶", "龙井", "茉莉", "抹茶")) or "茶事" in ctx["theme"]:
        col = col if kw else _mix("#76975b", ctx["paper"], .22)
        d.leaf(cx, cy, r * 2.0, -0.5, fill=col)
        return (cx - r, cy - r * 0.6, cx + r, cy + r * 0.6)
    d.moon(cx, cy, r, fill=col)
    return (cx - r, cy - r, cx + r, cy + r)


def push_giant_cup(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh * 0.67, bw / 1.30)
    ground = y1 - bh*.06
    cx = x0 + bw*.63
    acc = ctx["accents"]
    d.cup(cx, ground, h, straw=True, fill=acc[0] if acc else None)
    fh = h * 0.42
    hand_y = ground - 0.61 * fh
    edge = _cup_edge_x(cx, ground, h, hand_y, -1)
    fx = edge - 0.46 * fh + d.pen * 0.4
    d.figure(fx, ground, fh, pose="push", facing=1)
    detail = _detail_ink(d, ctx)
    if any(term in ctx["theme"] for term in ("茶会", "一桌", "校园")):
        d.figure(cx + h*.41, ground, fh*.68, pose="wave", facing=-1)
        _ticks(d, cx + h*.26, ground-h*.74, h*.12, detail)
    else:
        _ticks(d, fx-fh*.68, ground-fh*.7, fh*.31, detail, side=-1)
    d.line((fx - fh * 0.5, ground + d.pen), (cx + h * 0.46, ground + d.pen * 0.5), width=d.pen*.65)
    d.line((cx + h * 0.15, ground + d.pen*3), (cx + h * 0.33, ground + d.pen*2.8), color=detail, width=d.pen*.46)
    return {}


def cup_lifts_fruit(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh * 0.43, bw * 0.6)
    cx, base = (x0 + x1) / 2, y1 - bh * 0.08
    tw = h * 0.36
    top = base - h
    fill = _mix(ctx["accents"][0], ctx["paper"], .85) if ctx["accents"] else None
    d.cup(cx, base, h, lid=True, fill=fill)
    r = min(h*.45, bh*.19, bw*.2)
    fy = top - h*.22 - r*.86
    _subject_mark(d, cx, fy, r, ctx)
    for side in (-1, 1):
        hand = cx + side*r*.52, fy + r*.59
        d.path([(cx+side*tw*.84, top+h*.45), (cx+side*(tw+h*.23), top+h*.1), hand])
        d.circle(*hand, d.pen*1.16, width=d.pen*.60, amp=.6, overshoot=0)
    detail = _detail_ink(d, ctx)
    _ticks(d, cx+r*1.32, fy-r*.48, r*.31, detail)
    d.leaf(cx-r*1.53, fy-r*.13, r*.44, -.64, width=d.pen*.51, color=detail)
    d.line((cx-h*.35, base+d.pen*3), (cx+h*.28, base+d.pen*2.6), width=d.pen*.48, color=detail)
    return {}


def fallen_cup_tree(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    th = min(bh * 0.94, bw * 0.76)
    tx = x0 + bw * 0.36
    ground = y1-bh*.03
    d.tree(tx, ground, th)
    ch = th * 0.28
    acc = ctx["accents"]
    bx = d.cup_tipped(tx + th * 0.43, ground, ch, angle=1.45)
    if acc:   # the spill tells the story (cup knocked over while someone dozed)
        d.fill(d._ellipse(bx[2] + ch * 0.32, ground - ch * 0.04, ch * 0.47, ch * 0.09), acc[0], alpha=0.9)
    d.line((bx[2] + ch * 0.1, bx[1] + ch * 0.05), (bx[2] + ch * 0.5, bx[1] - ch * 0.35))   # straw rolled out
    d.line((tx + th*.14, ground), (min(x1, bx[2] + ch*.9), ground + ch*.02), width=d.pen*.65)
    detail = _detail_ink(d, ctx)
    for dx, dy, s, angle in [(-.15, -.73, .11, -.6), (.14, -.65, .10, .55), (.22, -.37, .11, 1.05)]:
        d.leaf(tx + th*dx, ground + th*dy, th*s, angle, width=d.pen*.48, color=detail)
    for dx in (-.26, -.19, .15):
        x = tx + dx*th
        d.line((x, ground-th*.02), (x-th*.025, ground-th*.07), color=detail, width=d.pen*.45)
    return {}


def drip_into_cup(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh * 0.42, bw * 0.62)
    cx, base = (x0 + x1) / 2, y1
    acc = ctx["accents"]
    # the drink is the fruit, diluted: tint of the subject colour, not an unrelated secondary
    d.cup(cx, base, h, lid=False, fill=_mix(acc[0], ctx["paper"], 0.42) if acc else None)
    rim_y = base - h
    d.path(d._ellipse(cx, rim_y, h*.36, h*.09), closed=True, overshoot=0)
    d.path(d._ellipse(cx, rim_y+h*.025, h*.26, h*.040, .15, math.pi-.15), width=d.pen*.48,
           color=_detail_ink(d, ctx))
    r = min(bh * 0.13, bw * 0.17)
    fy = y0 + bh * 0.06 + r * 1.3
    bb = _subject_mark(d, cx, fy, r, ctx)
    top = rim_y - h * 0.11
    span = top - bb[3]
    for k, t in enumerate((0.28, 0.58, 0.86)):   # three drops, growing as they fall
        dr = r * (0.07 + 0.035 * k)
        yy = bb[3] + span * t
        d.fill(d._ellipse(cx, yy, dr, dr * 1.35), acc[0] if acc else ctx["ink"], alpha=1.0, amp=0)
    _ticks(d, cx+r*1.3, fy+r*.10, r*.32, _detail_ink(d, ctx))
    d.line((cx-h*.37, base+d.pen*2.3), (cx+h*.40, base+d.pen*2.0), width=d.pen*.48, color=_detail_ink(d, ctx))
    return {}


def _cubic(start, control1, control2, end):
    t = np.linspace(0, 1, 25)[:, None]
    a, b, c, e = np.asarray([start, control1, control2, end])
    return (1-t)**3 * a + 3*(1-t)**2*t * b + 3*(1-t)*t**2 * c + t**3 * e


def moon_tea(d, box, ctx):
    """A crescent, a drip and an open teacup, with a few marks of night around them."""
    x0, y0, x1, y1 = box
    scale = min((x1 - x0) / 700, (y1 - y0) / 720)
    origin = ((x0 + x1) / 2, (y0 + y1 - 720 * scale) / 2)
    amber = ctx["accents"][0]
    paper = ctx["paper"]
    tea = _mix(amber, d.ink, 0.20)
    ochre = _mix(amber, d.ink, 0.38)
    pale = _mix(amber, paper, 0.75)

    def stroke(points, *, width=1.0, color=None, closed=False):
        d.path(np.asarray(points) * scale + origin, width=d.pen * width, color=color,
               closed=closed, amp=2.0 * scale, overshoot=0)

    def fill(points, color, amp=1.4):
        d.fill(np.asarray(points) * scale + origin, color, amp=amp * scale)

    def spark(x, y, size, color):
        shape = [(x, y-size), (x+size*.21, y-size*.22), (x+size*.72, y),
                 (x+size*.19, y+size*.19), (x, y+size*.82),
                 (x-size*.23, y+size*.22), (x-size*.65, y), (x-size*.2, y-size*.21)]
        stroke(shape, width=0.52, color=color, closed=True)

    # Keep the crescent's contour and the recessed tea surface easy to read.
    outer = d._ellipse(-106, 165, 148, 148, -.27 * math.pi, -1.72 * math.pi)
    inner = _cubic(outer[-1], (-215, 213), (-215, 138), outer[0])
    crescent = np.vstack([outer, inner[1:-1]])
    fill(crescent, amber, amp=2.5)
    stroke(crescent, width=0.86, closed=True)
    stroke(_cubic((-202, 114), (-235, 161), (-227, 206), (-205, 231)),
           width=0.56, color=_mix(amber, paper, 0.64))

    for x, y, size in [(-13, 308, 9), (2, 366, 12), (12, 429, 15)]:
        profile = [(x, y-size*1.4), (x+size*.64, y-size*.14), (x+size*.68, y+size*.37),
                   (x+size*.32, y+size*.78), (x-size*.32, y+size*.79),
                   (x-size*.67, y+size*.30), (x-size*.55, y-size*.18)]
        fill(profile, tea, amp=0.55)

    rim = d._ellipse(25, 512, 208, 57)
    cup = np.vstack([
        _cubic((-183, 512), (-160, 557), (-140, 626), (-50, 635)),
        _cubic((-50, 635), (-20, 638), (72, 638), (98, 634))[1:],
        _cubic((98, 634), (164, 627), (218, 554), (233, 512))[1:],
    ])
    fill(cup, pale)
    fill(d._ellipse(25, 524, 190, 36), tea)
    stroke(cup)
    stroke(rim, closed=True)
    stroke(_cubic((229, 521), (345, 486), (311, 567), (187, 579)), width=0.9)
    stroke(_cubic((-246, 629), (-222, 704), (246, 713), (282, 630)), width=0.65)
    stroke(_cubic((-247, 630), (-241, 622), (-221, 618), (-202, 617)), width=0.60)
    stroke(_cubic((254, 618), (268, 621), (277, 626), (282, 630)), width=0.60)
    stroke(_cubic((-34, 526), (-20, 543), (46, 544), (61, 526)), width=0.62, color=paper)
    stroke(_cubic((-62, 532), (-56, 543), (-40, 549), (-27, 552)), width=0.46, color=pale)

    # Two small clusters follow the diagonal from the moon to the cup.
    spark(106, 87, 27, ochre)
    spark(202, 173, 15, ochre)
    spark(-266, 284, 16, ochre)
    for x, y, r in [(157, 127, 4.5), (221, 211, 4), (-222, 310, 4.5), (-274, 320, 3)]:
        fill(d._ellipse(x, y, r, r * .86), ochre, amp=0.25)
    stroke(_cubic((122, 220), (163, 233), (183, 259), (183, 283)), width=0.42, color=ochre)
    stroke(_cubic((181, 298), (180, 302), (178, 306), (176, 310)), width=0.42, color=ochre)
    for i in range(3):
        x, y = 97 + 22*i, 686 - 2*i
        stroke([(x, y+8), (x+13, y-2)], width=0.41, color=ochre)
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
    if acc and not any(t in ctx["theme"] for t in ("暖", "冬", "热茶")):
        # the burst sits *behind and below* the jump (where the energy came from), not on the body
        d.fill(d._ellipse(fx - fh * 0.42, fy - fh * 0.3, fh * 0.42, fh * 0.34), acc[0], alpha=0.95)
    warm = any(t in ctx["theme"] for t in ("暖", "冬", "热茶"))
    fill = _mix("#b87943", ctx["paper"], .40) if warm else None
    d.cup(cx, base, h, straw=False, fill=fill)
    d.figure(fx, fy, fh, pose="leap", facing=1)
    for s in (-1, 1):
        d.line((fx + s * fh * 0.62, fy - fh * 1.05), (fx + s * fh * 0.78, fy - fh * 1.22))
    if warm:
        _steam(d, cx+h*.39, top-h*.12, h*.31, "#9b6841")
        d.path(d._ellipse(cx, base+h*.027, h*.44, h*.025, .16, math.pi-.16),
               width=d.pen*.51, color="#9b6841")
    return {}


def color_field(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    paper = ctx["paper"]
    acc = ctx["accents"] or ["#e9b44c"]
    cx, cy = x0 + bw*.49, y0 + bh*.48
    s = min(bw*.46, bh*.49)
    if "木瓜" in ctx["theme"]:
        cut = d._ellipse(cx, cy, s*.64, s*.84)
        d.fill(d._ellipse(cx-s*.08, cy+s*.055, s*.76, s*.90), "#7c8b58", amp=s*.008)
        d.fill(cut, "#e6a55e", amp=s*.007)
        d.path(cut, closed=True, overshoot=0)
        pit = d._ellipse(cx+s*.04, cy+s*.018, s*.25, s*.57)
        d.fill(pit, _mix("#e6a55e", paper, .58), amp=1.0)
        d.path(pit, closed=True, width=d.pen*.50, overshoot=0)
        for dx, dy, r in [(-.08,-.32,.042), (.083,-.24,.04), (-.073,-.08,.039),
                          (.144,.025,.038), (-.067,.13,.046), (.071,.24,.046), (-.021,.36,.035)]:
            d.fill(d._ellipse(cx+s*dx, cy+s*dy, s*r, s*r*1.12), "#604433", amp=.5)
        d.leaf(cx+s*.81, cy+s*.40, s*.32, -.5, color="#606b49", width=d.pen*.62)
        _ticks(d, cx+s*.78, cy-s*.35, s*.15, "#9e6231")
    elif any(t in ctx["theme"] for t in ("幼芽", "绿意", "茶叶", "茶芽", "抹茶")):
        green = "#78975c"
        d.fill(d._ellipse(cx, cy+s*.12, s*.72, s*.65), _mix(green, paper, .78), amp=s*.018)
        h, base = s*.67, cy+s*.70
        d.cup(cx, base, h, lid=False, fill=_mix(green, paper, .64))
        d.path(d._ellipse(cx, base-h, h*.36, h*.10), closed=True, overshoot=0)
        d.path(_cubic((cx, base-h), (cx-s*.09, cy-s*.22), (cx+s*.08, cy-s*.39), (cx, cy-s*.58)))
        d.leaf(cx-s*.17, cy-s*.42, s*.37, .64, fill=green, width=d.pen*.75)
        d.leaf(cx+s*.16, cy-s*.55, s*.35, -.64, fill=green, width=d.pen*.75)
        _ticks(d, cx+s*.47, cy-s*.50, s*.13, "#687845")
        d.leaf(cx-s*.59, cy+s*.29, s*.19, .65, width=d.pen*.50, color="#687845")
    else:
        d.fill(d._ellipse(cx+s*.07, cy+s*.05, s*.82, s*.72), _mix(acc[0], paper, .70), amp=s*.021)
        _subject_mark(d, cx-s*.09, cy-s*.10, s*.51, ctx)
        d.cup(cx+s*.48, cy+s*.78, s*.42, fill=_mix(acc[0], paper, .5), width=d.pen*.76)
        _ticks(d, cx+s*.56, cy-s*.45, s*.15, _detail_ink(d, ctx))
    return {}


def lettering_lead(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h = min(bh*.55, bw*.43)
    cx, base = x0+bw*.56, y1-bh*.12
    acc = ctx["accents"]
    detail = _detail_ink(d, ctx)
    theme = ctx["theme"]
    if "苹果" in theme:
        d.fruit(cx, base-h*.52, h*.55, "apple", fill=acc[0] if acc else None)
        d.path(_cubic((cx-h*.84, base+h*.08), (cx-h*.63, base-h*.09),
                      (cx+h*.57, base+h*.24), (cx+h*.76, base-h*.10)), width=d.pen*.58, color=detail)
        d.leaf(cx-h*.82, base-h*.74, h*.27, -.39, width=d.pen*.62, color=detail)
        d.path(_cubic((cx+h*.23, base-h*1.16), (cx+h*.38, base-h*1.24),
                      (cx+h*.64, base-h*1.12), (cx+h*.74, base-h*1.24)), width=d.pen*.46, color=detail)
    elif "牛奶" in theme:
        x, y = cx-h*.50, base-h*.70
        bottle = np.asarray([(-.26,.43),(-.25,-.21),(-.11,-.39),(-.10,-.51),(.14,-.51),
                             (.15,-.37),(.28,-.19),(.27,.38),(.10,.48),(-.18,.47)]) * h
        a = .72
        bottle = bottle @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + [x, y]
        d.fill(bottle, acc[0], amp=1.0)
        d.path(bottle, closed=True, overshoot=0)
        d.line(bottle[3], bottle[4], width=d.pen*.58, color=detail)
        d.path(d._ellipse(cx+h*.34, base-h*.25, h*.37, h*.12), closed=True, overshoot=0)
        d.path(_cubic((cx-h*.03, base-h*.25), (cx+h*.04, base+h*.17),
                      (cx+h*.64, base+h*.13), (cx+h*.71, base-h*.25)))
        mx, my = (bottle[3] + bottle[4]) / 2
        d.path(_cubic((mx+h*.075, my), (cx+h*.26, my+h*.10),
                      (cx+h*.13, base-h*.44), (cx+h*.30, base-h*.27)),
               color=detail, width=d.pen*.56)
        _ticks(d, x-h*.34, y-h*.32, h*.16, detail, side=-1)
    else:
        d.cup(cx, base, h, straw="导航" in theme, fill=acc[0] if acc else None)
        if "导航" in theme:
            d.arrow((cx-h*.92, base+h*.03), (cx+h*.90, base+h*.19),
                    head=h*.29, color=detail, width=d.pen*.73)
        else:
            _steam(d, cx, base-h*1.18, h*.39, detail)
            d.path(d._ellipse(cx, base+h*.10, h*.57, h*.095, .2, math.pi-.2),
                   color=detail, width=d.pen*.48)
            if "糯米" in theme:
                for u, v, a in [(.63,-.18,-.5), (.81,-.09,.30), (.69,.08,.73)]:
                    xx, yy = cx+h*u, base+h*v
                    grain = d._ellipse(0, 0, h*.10, h*.038)
                    grain = grain @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
                    d.path(grain+[xx, yy], closed=True, width=d.pen*.48, overshoot=0, color=detail)
            else:
                _ticks(d, cx+h*.58, base-h*.66, h*.15, detail)
    return {}


def _fit(img: Image.Image, box, fill: float = 1.0):
    img = img.convert("RGBA")
    foreground = img.getchannel("A").point(lambda a: 255 if a > 32 else 0).getbbox()
    if foreground is not None:
        img = img.crop(foreground)
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
    detail = _detail_ink(d, ctx)
    if ph.getchannel("A").getextrema()[0] == 255:
        d.rect(px - pad, py - pad, px + ph.width + pad, py + ph.height + pad, width=d.pen*.68)
    else:
        # A pair of open frame corners guides the eye without boxing in a cutout.
        s = min(ph.width, ph.height) * .20
        d.path([(px-pad, py+s), (px-pad, py-pad), (px+s, py-pad)],
               width=d.pen*.51, color=detail)
        x, y = px+ph.width+pad, py+ph.height+pad
        d.path([(x-s, y), (x, y), (x, y-s)], width=d.pen*.51, color=detail)
    if any(t in ctx["theme"] for t in ("山野", "山风", "奶盖")):
        lx, ly = px + ph.width*.94, py - pad*2
        _steam(d, lx, ly, min(bh*.12, ph.height*.18), detail)
        d.leaf(px+ph.width+pad*3, py+ph.height*.58, ph.width*.22, -.48, color=detail, width=d.pen*.50)
    else:
        fx = px + ph.width + pad + ph.width*.10
        r = min(bh*.077, ph.width*.17)
        fy = py + ph.height*.69
        if any(term in ctx["theme"] for term in ("芭乐", "番石榴")):
            # A cut guava reads by its green rind and speckled pink centre.
            skin = d._ellipse(fx, fy, r*.94, r)
            d.fill(skin, "#82945d", amp=r*.018)
            d.fill(d._ellipse(fx-r*.045, fy-r*.015, r*.76, r*.81), "#eea1af", amp=r*.017)
            d.path(skin, closed=True, width=d.pen*.72, overshoot=0)
            for u, v in [(-.15,-.33), (.28,-.12), (-.22,.09), (.11,.19), (-.04,.39)]:
                d.fill(d._ellipse(fx+r*u, fy+r*v, r*.04, r*.042), "#a47966", amp=.45)
        else:
            d.fruit(fx, fy, r, ctx["fruit_kind"] or "round",
                    fill=_mix(ctx["accents"][0], ctx["paper"], .20) if ctx["accents"] else None, width=d.pen*.78)
        _ticks(d, fx+r*1.24, py+ph.height*.60, r*.47, detail)
    return {"photos": [(ph, (px, py))]}


def photo_push(d, box, ctx):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ph = _fit(ctx["photo"], (x0 + bw*.35, y0, x1, y0+bh*.76), .9)
    ground = y1-bh*.06
    px, py = int(x0+bw*.76-ph.width/2), int(ground-ph.height)
    fh = min(bh*.35, ph.height*.48, bw*.35)
    alpha = np.asarray(ph.getchannel("A"))
    hand_row = round(ph.height - fh*.61)
    foreground = np.flatnonzero(alpha[min(ph.height-1, hand_row)] > 32)
    edge = px + (foreground[0] if foreground.size else 0)
    fx = edge - .46*fh + d.pen*.25
    detail = _detail_ink(d, ctx)
    if "奶盖" in ctx["theme"]:
        mx, my = px+ph.width*.48, py - fh*.28
        d.cloud(mx, my, ph.width*.74, width=d.pen*.75)
        _ticks(d, mx+ph.width*.44, my-fh*.24, fh*.20, detail)
        d.dotted((mx, my+fh*.11), (mx+fh*.035, py-d.pen*1.5), 3,
                 color=detail, width=d.pen*.49)
        fx = px-ph.width*.16
    else:
        d.figure(fx, ground, fh, pose="push", facing=1)
        _ticks(d, fx-fh*.55, ground-fh*.76, fh*.26, detail, side=-1)
        if any(t in ctx["theme"] for t in ("原料", "产地", "采摘")):
            lx, ly = px-ph.width*.49, py+ph.height*.43
            dx, dy = ph.width*.36, -ph.height*.23
            green = "#657846"
            d.line((lx, ly), (lx+dx, ly+dy), width=d.pen*.56, color=green)
            d.leaf(lx+dx*.25, ly+dy*.48, ph.width*.33, .72, fill="#abb48c", color=green, width=d.pen*.53)
            d.leaf(lx+dx*.92, ly+dy*.77, ph.width*.27, -.35, fill="#abb48c", color=green, width=d.pen*.52)
    d.line((fx - fh*.43, ground+d.pen), (px+ph.width, ground+d.pen*.7), width=d.pen*.60)
    return {"photos": [(ph, (px, py))]}


RECIPES = {f.__name__: f for f in (push_giant_cup, cup_lifts_fruit, fallen_cup_tree, drip_into_cup, moon_tea, leap_on_cup,
                                   color_field, lettering_lead, photo_frame, photo_push)}
