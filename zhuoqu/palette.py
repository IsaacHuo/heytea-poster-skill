"""Colour system derived from measured references (see references/visual-language.md 3).

Rule: paper-tinted background + near-black line + at most two subject colours.
Subject colours come from the product/ingredient itself, not from a fixed brand palette.
Paper and ink values are medians of measured backgrounds of A-line exemplars
(0234 #f7f7f3, 0236 #f2efe6, 0237 #f3f0e8, 0238 #f9f9f9; ink 0238 #060505 / 0322 #121110).
"""
from __future__ import annotations

import re

PAPERS = {"warm": "#f3f0e8", "neutral": "#f7f7f3", "cream": "#f2efe6", "white": "#f9f9f8", "pink": "#f6e4e4"}
INK = "#161514"
SOFT_INK = "#4a4743"

# keyword -> (main, secondary, fruit-kind for doodle)
SUBJECTS = [
    (r"柿", ("#e07a2e", "#6f8a4a", "persimmon")),
    (r"葡萄|巨峰|夏黑", ("#7a4c9c", "#8fae5a", "grape")),
    (r"芒果|芒", ("#f2ae2e", "#7da24a", "mango")),
    (r"桃", ("#f19a8e", "#8fb36b", "round")),
    (r"柠檬|青柠|柠", ("#efd34a", "#7da24a", "lemon")),
    (r"草莓|莓", ("#de4b52", "#6f9a4a", "strawberry")),
    (r"橙|橘|柑|桔", ("#f08a2b", "#6f9a4a", "round")),
    (r"苹果", ("#d9483f", "#7da24a", "round")),
    (r"西瓜|瓜", ("#e2574c", "#4f8a4a", "round")),
    (r"芭乐|番石榴", ("#f08aa0", "#8fb36b", "round")),
    (r"芋", ("#a58cc4", "#e9c46a", None)),
    (r"可可|巧克力|咖啡", ("#7a4f3e", "#e9d3a8", None)),
    (r"龙井|绿茶|抹茶|茶叶|茉莉", ("#6f8f4e", "#e6dca8", None)),
    (r"乌龙|红茶|岩茶|琥珀", ("#b8742f", "#efd9a7", None)),
    (r"奶|牦牛", ("#e9dcc2", "#c9a46a", None)),
    (r"樱", ("#f2a6b8", "#8fb36b", None)),
    (r"月|中秋", ("#f2d16b", "#2f6478", None)),
    (r"春节|新年|马年|过年", ("#d8402f", "#f2d16b", None)),
]
# short regex hits -> the word a copywriter would actually write
NAMES = {"柿": "柿子", "芒": "芒果", "柠": "柠檬", "莓": "草莓", "瓜": "西瓜", "芋": "香芋", "月": "月亮",
         "橘": "橘子", "橙": "橙子", "桃": "桃子", "樱": "樱花", "奶": "牛乳"}
SEASON = {"spring": ("#9bc47a", "#f2a6b8"), "summer": ("#4fa3c7", "#f2d64b"),
          "autumn": ("#d98a3a", "#8a6a3f"), "winter": ("#3f6f9a", "#e9e2d2")}


def subject_colors(text: str) -> tuple[str, str, str | None, str | None]:
    """Return (main, secondary, fruit_kind, matched_keyword) for the first matching subject."""
    for pat, (a, b, kind) in SUBJECTS:
        m = re.search(pat, text)
        if m:
            return a, b, kind, NAMES.get(m.group(0), m.group(0))
    return "#e9b44c", "#6f8f4e", None, None


def choose(text: str, mode: str = "single_accent", season: str | None = None) -> dict:
    """mode: single_accent | two_tone | mono_ink | soft_natural | high_contrast | color_field"""
    main, sec, kind, kw = subject_colors(text)
    if kw is None and season in SEASON:
        main, sec = SEASON[season]
    paper = PAPERS["warm"]
    if mode == "mono_ink":
        acc = []
    elif mode == "two_tone" or mode == "color_field":
        acc = [main, sec]
    elif mode == "high_contrast":
        paper, acc = main, ["#f7f3ea"]
    elif mode == "soft_natural":
        paper, acc = PAPERS["cream"], [_mix(main, "#f3f0e8", 0.25)]
    else:
        acc = [main]
    return dict(paper=paper, ink=INK, soft_ink=SOFT_INK, accents=acc, mode=mode, subject_keyword=kw, fruit_kind=kind)


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))


def luminance(h: str) -> float:
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted([luminance(a), luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)
