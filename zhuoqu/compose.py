"""End-to-end poster pipeline: plan -> layout -> subject layer -> text layers -> composite
-> validate -> (auto-repair once) -> save.

Outputs (outputs/<run>/): poster.png, title.svg, layers/*.png, plan.json, report.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from . import layout as L
from . import paths
from .generate import subject_layer
from .plan import make_plan
from .typography import render_text
from .validate import check


def _fit_text(text, box, role, base: dict, seed, color, vertical=False, max_size=None):
    """Render at a probe size, then re-render at the size that fits the box."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    kw = dict(role=role, style=base.get("style"), intensity=base.get("intensity", 0.5),
              layout="vertical" if vertical else base.get("layout", "organic"), seed=seed, color=color,
              overrides=base.get("overrides") or None, protect=base.get("protect") or None)
    probe = render_text(text, font_size=100, check_legibility=False, **kw)
    fs = 100 * min(bw / probe.width, bh / probe.height)
    if max_size:
        fs = min(fs, max_size)
    return render_text(text, font_size=max(18, fs), **kw)


def _paste(canvas_size, im: Image.Image, xy):
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    layer.paste(im, xy, im)
    return layer


def make_poster(request: str, title: Optional[str] = None, info: Optional[list[str]] = None, ratio: Optional[str] = None,
                photo: Optional[str] = None, seed: int = 0, concept: Optional[str] = None,
                text_instruction: Optional[str] = None, backend: Optional[str] = None,
                out_dir: Optional[str | Path] = None, use_retrieval: bool = True,
                record_history: bool = True) -> dict:
    t0 = time.time()
    plan = make_plan(request, title, info, ratio, photo, seed, concept, text_instruction, use_retrieval=use_retrieval,
                     novelty=record_history)
    W, H = plan["brief"]["size"]
    pal = plan["palette"]
    lay = L.solve(plan["layout"], W, H, seed)
    photo_im = Image.open(photo).convert("RGBA") if photo else None
    paper = pal["paper"]
    bg = Image.new("RGBA", (W, H), paper)
    gen = subject_layer(plan, lay["subject"], (W, H), photo_im, backend)
    subj = gen["image"]
    short = min(W, H)
    t = plan["texts"]
    title_color = pal["ink"]
    if plan["concept"]["concept"] == "color_field" and pal["accents"]:
        title_color = pal["ink"]
    vertical = lay.get("vertical", False)
    tbox = lay["title"]
    tspec = dict(t["title"])
    tmax, tval = 0.13 * short, "top"
    if plan["layout"] == "lettering_lead":
        # the words are the picture: big, wrapped into 2-3 short lines, sitting down on the drawing
        n = len(tspec["text"].replace(" ", ""))
        tspec["overrides"] = dict(tspec.get("overrides") or {}, max_chars_per_line=max(2, min(5, -(-n // 2))))
        tmax, tval = 0.24 * short, "bottom"
    tr = _fit_text(tspec["text"], tbox, tspec["role"], tspec, seed, title_color, vertical, max_size=tmax)
    timg = tr.image()
    tx, ty = L.place(timg.width, timg.height, tbox, lay["align"] if not vertical else "center", tval)
    layers = [("title", _paste((W, H), timg, (tx, ty)), t["title"], tr, title_color)]
    # subtitle sits under the title, smaller and quieter
    y_cursor = ty + timg.height
    if t.get("subtitle"):
        sb = (tbox[0], y_cursor + int(0.02 * H), tbox[2], y_cursor + int(0.08 * H))
        sr = _fit_text(t["subtitle"]["text"], sb, "expressive", dict(style="quiet_note", intensity=0.35), seed + 1,
                       pal["soft_ink"], max_size=0.045 * short)
        si = sr.image()
        sx, sy = L.place(si.width, si.height, sb, lay["align"] if not vertical else "left", "top")
        layers.append(("subtitle", _paste((W, H), si, (sx, sy)), t["subtitle"], sr, pal["soft_ink"]))
    # functional info: one quiet line (or stacked lines) in the info box
    ib = lay["info"]
    infos = t["info"]
    if infos:
        ih = (ib[3] - ib[1]) / max(1, len(infos))
        for k, it in enumerate(infos):
            box = (ib[0], int(ib[1] + k * ih), ib[2], int(ib[1] + (k + 1) * ih))
            fr = _fit_text(it["text"], box, "functional", dict(style="functional_clean"), seed, pal["soft_ink"],
                           max_size=0.03 * short)
            fi = fr.image()
            fx, fy = L.place(fi.width, fi.height, box, lay["align"] if lay["align"] != "right" else "left", "middle")
            layers.append((f"info_{k}", _paste((W, H), fi, (fx, fy)), it, fr, pal["soft_ink"]))
    poster = Image.alpha_composite(bg, subj)
    for _, lay_im, *_ in layers:
        poster = Image.alpha_composite(poster, lay_im)
    sub_alpha = np.asarray(subj)[..., 3]
    texts = [dict(text=spec["text"], role=spec["role"], alpha=np.asarray(im)[..., 3], color=col,
                  threshold=rr.params.get("legibility_threshold", 0.6)) for _, im, spec, rr, col in layers]
    report = check(poster, sub_alpha, texts, paper)
    report.update(backend=gen["backend"], backend_notes=gen["notes"], seconds=round(time.time() - t0, 1),
                  title_render=dict(style=tr.style, intensity=tr.intensity, attempts=tr.attempts,
                                    legibility=tr.legibility, warnings=tr.warnings))
    out = Path(out_dir) if out_dir else paths.OUTPUTS / "posters" / time.strftime("%Y%m%d-%H%M%S")
    (out / "layers").mkdir(parents=True, exist_ok=True)
    poster.convert("RGB").save(out / "poster.png")
    subj.save(out / "layers" / "subject.png")
    for name, im, spec, rr, _ in layers:
        im.save(out / "layers" / f"{name}.png")
    (out / "title.svg").write_text(tr.svg(), encoding="utf-8")
    safe_plan = json.loads(json.dumps(plan, ensure_ascii=False, default=str))
    safe_plan["layout_boxes"] = lay
    (out / "plan.json").write_text(json.dumps(safe_plan, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if record_history:   # feeds the planner novelty penalty; regression runs must not
      with (paths.OUTPUTS / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(concept=plan["concept"]["concept"], request=request, out=str(out),
                                passed=report["passed"]), ensure_ascii=False) + "\n")
    return dict(out=str(out), poster=str(out / "poster.png"), plan=plan, report=report)
