"""Regression set: the same text + poster cases after every change to the type engine,
layout or planner. Writes outputs/regression/<stamp>/{summary.json, text_sheet.png,
poster_sheet.png} and diffs the summary against the previous run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import paths

# (id, text, role, style, intensity, layout) - covers lengths, numbers, mixed script, rare chars
TEXT_CASES = [
    ("short", "秋", "expressive", "playful_handwritten", 0.6, "organic"),
    ("short2", "柿熟了", "expressive", "playful_handwritten", 0.55, "organic"),
    ("phrase", "今天喝点好的", "expressive", "playful_handwritten", 0.55, "organic"),
    ("long", "秋天的第一杯，慢慢喝不着急", "expressive", "playful_handwritten", 0.5, "organic"),
    ("long2", "把整个夏天的风都装进这一杯里", "expressive", "quiet_note", 0.45, "organic"),
    ("quiet", "慢慢喝，不着急", "expressive", "quiet_note", 0.3, "organic"),
    ("loud", "冲啊！下课了", "expressive", "playful_handwritten", 0.85, "stagger"),
    ("mixed", "龙井的 not to do list", "expressive", "playful_handwritten", 0.5, "organic"),
    ("mixed2", "new style tea 慢慢来", "expressive", "casual_pen", 0.5, "organic"),
    ("quote", "“柿”不简单", "expressive", "playful_handwritten", 0.55, "organic"),
    ("vertical", "月满杯满", "expressive", "playful_handwritten", 0.5, "vertical"),
    ("multiline", "小憩一下\n为呈现更好的喜茶", "expressive", "playful_handwritten", 0.5, "organic"),
    ("complex", "鬱金香馥郁", "expressive", "playful_handwritten", 0.5, "organic"),
    ("complex2", "龘靐齉爩", "expressive", "quiet_note", 0.4, "organic"),
    ("marker", "香芋超看热闹", "expressive", "marker_round", 0.55, "organic"),
    ("brush", "什组草台班子", "expressive", "brush_banzi", 0.5, "organic"),
    ("pen", "去茶底 纯享浓浓", "expressive", "casual_pen", 0.55, "organic"),
    ("hybrid", "多肉葡萄冻上新", "hybrid", "playful_handwritten", 0.55, "organic"),
    ("date", "11月20日-12月5日", "functional", "functional_clean", 0.5, "straight"),
    ("time", "14:00-17:00 图书馆前草坪", "functional", "functional_clean", 0.5, "straight"),
    ("price", "¥19 / 杯 限时", "functional", "functional_clean", 0.5, "straight"),
    ("ingredients", "配料：柿子果肉、乌龙茶汤、冰块", "functional", "functional_warm", 0.5, "straight"),
    ("disclaimer", "图片仅供参考，以门店实物为准", "functional", "functional_clean", 0.5, "straight"),
    ("en_func", "Open 10:00-22:00 Daily", "functional", "functional_clean", 0.5, "straight"),
]

POSTER_CASES = [
    ("fruit_tea", "为一杯新的柿子水果茶设计海报，秋天，自然", dict(info=["11月20日起 全国门店"]), None),
    ("campus", "为一个校园活动制作轻松、俏皮、有手绘感的宣传图 “下课去喝一杯”", dict(info=["12月5日 14:00-17:00", "图书馆前草坪"]), None),
    ("quiet_brand", "使用少量文字和大片留白，制作一张安静的龙井茶品牌风格海报", dict(), None),
    ("text_only", "纯文字海报，字是主角 “今天先发会儿呆”", dict(), "lettering_lead"),
    ("abstract", "把芒果的味道画成抽象色块，艺术一点", dict(info=["芒果季 6月1日起"]), "color_field"),
    ("illustration", "生成一张带有拙趣手写字的秋日插画", dict(), None),
    ("long_title", "葡萄上新海报 “把整个夏天的风都装进这一杯里”", dict(info=["¥19 / 杯"]), None),
    ("wide_banner", "横版 banner，柠檬茶，开心", dict(ratio="16:9", info=["7月1日-7月7日"]), None),
    ("photo", "将现有产品照片制作成一张具有鲜明手绘设计语言的广告 “刚摘的葡萄”", dict(photo="FIXTURE"), None),
    ("moon", "中秋节海报，温暖 “月满，杯满”", dict(ratio="9:16"), "cup_lifts_fruit"),
]


def _font(size):
    return ImageFont.truetype(str(paths.FONTS / "NotoSansSC[wght].ttf"), size)


def _sheet(items, out, cell=(520, 300), cols=4):
    rows = -(-len(items) // cols)
    sheet = Image.new("RGB", (cols * cell[0], rows * cell[1]), "#ffffff")
    d = ImageDraw.Draw(sheet)
    for k, (img, label, ok) in enumerate(items):
        x, y = (k % cols) * cell[0], (k // cols) * cell[1]
        im = img.convert("RGBA"); im.thumbnail((cell[0] - 20, cell[1] - 52))
        bg = Image.new("RGBA", im.size, "#f3f0e8"); bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), (x + 10, y + 8))
        d.text((x + 10, y + cell[1] - 40), label, fill="#1a7f37" if ok else "#c62828", font=_font(17))
    sheet.save(out)


def photo_fixture() -> Path:
    """Synthetic 'product photo' (gradient cup on a soft background) - license-free test input."""
    f = paths.ROOT / "tests" / "fixtures" / "synthetic_cup.png"
    if f.exists():
        return f
    f.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    W, H = 600, 800
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), np.float32) + np.array([226, 222, 214])
    img -= (yy / H)[..., None] * 18
    cup = (np.abs(xx - W / 2) < (110 + (H * 0.85 - yy) * -0.08 + 60)) & (yy > H * 0.22) & (yy < H * 0.86)
    shade = 0.75 + 0.25 * np.cos((xx - W / 2) / 140)
    img[cup] = (np.array([128, 76, 150])[None] * shade[cup][:, None])
    img[(yy > H * 0.18) & (yy < H * 0.23) & (np.abs(xx - W / 2) < 185)] = (240, 240, 236)
    Image.fromarray(img.clip(0, 255).astype("uint8")).save(f)
    return f


def run(posters: bool = True, seed: int = 7) -> dict:
    from .typography import render_text
    from .compose import make_poster
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = paths.OUTPUTS / "regression" / stamp
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(stamp=stamp, seed=seed, text=[], posters=[])
    items = []
    for cid, text, role, style, inten, lay in TEXT_CASES:
        try:
            r = render_text(text, role=role, style=style, intensity=inten, layout=lay, seed=seed, font_size=110)
            thr = r.params.get("legibility_threshold", 0.6) if role != "functional" else 0.999
            ok = r.legibility >= thr
            if r.ocr_unverifiable:
                ok = "unverified"   # not counted as pass; listed for human review
            r.image().save(out / f"text_{cid}.png")
            (out / f"text_{cid}.svg").write_text(r.svg(), encoding="utf-8")
            summary["text"].append(dict(id=cid, text=text, role=role, style=r.style, intensity=r.intensity, legibility=r.legibility,
                                        attempts=r.attempts, ocr=r.ocr_text, passed=ok if ok == "unverified" else bool(ok),
                                        warnings=r.warnings))
            items.append((r.image(), f"{cid} {r.legibility:.2f} {'UNVERIFIED' if ok == 'unverified' else 'OK' if ok else 'FAIL'}", ok is True))
        except Exception as e:  # noqa: BLE001
            summary["text"].append(dict(id=cid, text=text, passed=False, error=f"{type(e).__name__}: {e}"))
    _sheet(items, out / "text_sheet.png")
    if posters:
        pitems = []
        for cid, req, kw, concept in POSTER_CASES:
            kw = dict(kw)
            if kw.get("photo") == "FIXTURE":
                kw["photo"] = str(photo_fixture())
            try:
                res = make_poster(req, seed=seed, concept=concept, out_dir=out / f"poster_{cid}", record_history=False, **kw)
                rep = res["report"]
                summary["posters"].append(dict(id=cid, request=req, concept=res["plan"]["concept"]["concept"],
                                               title=res["plan"]["texts"]["title"]["text"], passed=rep["passed"],
                                               hard=rep["hard_failures"], warnings=rep["warnings"], metrics=rep["metrics"]))
                pitems.append((Image.open(res["poster"]), f"{cid} {res['plan']['concept']['concept']} {'PASS' if rep['passed'] else 'FAIL'}", rep["passed"]))
            except Exception as e:  # noqa: BLE001
                summary["posters"].append(dict(id=cid, request=req, passed=False, error=f"{type(e).__name__}: {e}"))
        if pitems:
            _sheet(pitems, out / "poster_sheet.png", cell=(420, 600), cols=5)
    prev = sorted(p for p in (paths.OUTPUTS / "regression").glob("*/summary.json") if p.parent != out)
    if prev:
        old = json.loads(prev[-1].read_text(encoding="utf-8"))
        o = {t["id"]: t.get("passed") for t in old.get("text", []) + old.get("posters", [])}
        summary["changed_vs_previous"] = [dict(id=t["id"], was=o.get(t["id"]), now=t.get("passed"))
                                          for t in summary["text"] + summary["posters"] if o.get(t["id"]) != t.get("passed")]
        summary["previous"] = prev[-1].parent.name
    tp = sum(t["passed"] is True for t in summary["text"]); pp = sum(p["passed"] is True for p in summary["posters"])
    tu = sum(t["passed"] == "unverified" for t in summary["text"])
    summary["totals"] = dict(text=f"{tp}/{len(summary['text'])}", text_unverified=tu, posters=f"{pp}/{len(summary['posters'])}")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    summary["out"] = str(out)
    return summary
