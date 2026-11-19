"""Quality checks. Hard gates: text accuracy (functional exact, expressive >= threshold),
text/subject collision, safe margins, contrast. Soft metrics (whitespace, edge density,
colour count) are compared with measured A-line reference ranges and only warn.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .analyze import measure_array
from .palette import contrast
from .typography import _norm, ocr_baseline, ocr_similarity

from .config import get

SOFT = dict(whitespace_min=get("validate", "whitespace_min", 0.55), edge_max=get("validate", "edge_max", 0.07),
            text_area_max=0.2)


def ink_box(alpha: np.ndarray, thr: int = 40):
    ys, xs = np.nonzero(alpha > thr)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _overlap(a, b):
    if a is None or b is None:
        return 0
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def check(poster: Image.Image, subject_alpha: np.ndarray, texts: list[dict], paper: str) -> dict:
    """texts: [{text, role, threshold, alpha (canvas-size uint8 mask), color}]"""
    W, H = poster.size
    hard, soft = [], []
    sub_mask = subject_alpha > 40
    margin = 0.04 * min(W, H)
    results = []
    for t in texts:
        a = t["alpha"]
        box = ink_box(a)
        r = dict(text=t["text"], role=t["role"], box=box)
        if box is None:
            hard.append(f"text '{t['text']}' rendered nothing"); results.append(r); continue
        # accuracy: OCR the poster crop (real background, real neighbours)
        x0, y0, x1, y1 = box
        pad = int(0.15 * (y1 - y0)) + 6
        crop = poster.crop((max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))).convert("RGBA")
        sim, got = ocr_similarity(crop, t["text"])
        r.update(ocr=got, similarity=sim)
        thr = 1.0 if t["role"] == "functional" else t.get("threshold", 0.6)
        if t["role"] == "functional" and _norm(got) != _norm(t["text"]):
            hard.append(f"functional text mismatch: want '{t['text']}' got '{got}'")
        elif sim < thr:
            b = ocr_baseline(t["text"])
            if b < thr:
                r["ocr_unverifiable"] = True
                soft.append(f"text '{t['text']}' is OCR-unverifiable (clean baseline {b}); needs human check")
            else:
                hard.append(f"expressive text '{t['text']}' OCR similarity {sim} < {thr}")
        coll = float(((a > 40) & sub_mask).sum()) / max(1, (a > 40).sum())
        r["collision"] = round(coll, 4)
        if coll > 0.02:
            hard.append(f"text '{t['text']}' overlaps illustration ({coll:.1%} of its ink)")
        if x0 < margin or y0 < margin or x1 > W - margin or y1 > H - margin:
            hard.append(f"text '{t['text']}' violates safe margin")
        c = contrast(t["color"], paper)
        r["contrast"] = round(c, 2)
        if c < (4.5 if t["role"] == "functional" else 3.0):
            hard.append(f"text '{t['text']}' contrast {c:.2f} too low")
        results.append(r)
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if _overlap(results[i].get("box"), results[j].get("box")) > 0:
                ai, aj = texts[i]["alpha"] > 40, texts[j]["alpha"] > 40
                if (ai & aj).sum() > 0:
                    hard.append(f"texts '{texts[i]['text']}' and '{texts[j]['text']}' collide")
    m = measure_array(np.asarray(poster.convert("RGB")))
    if m["whitespace_ratio"] < SOFT["whitespace_min"]:
        soft.append(f"whitespace {m['whitespace_ratio']} below A-line range (>= {SOFT['whitespace_min']})")
    if m["edge_density"] > SOFT["edge_max"]:
        soft.append(f"edge density {m['edge_density']} above A-line range (<= {SOFT['edge_max']}): busy")
    ink = sum((t["alpha"] > 40).sum() for t in texts) / (W * H)
    sub = float(sub_mask.mean())
    return dict(passed=not hard, hard_failures=hard, warnings=soft, texts=results,
                metrics=dict(whitespace=m["whitespace_ratio"], edge_density=m["edge_density"],
                             colorfulness=m["colorfulness"], text_ink_ratio=round(float(ink), 4),
                             subject_ink_ratio=round(sub, 4), centroid=(m["centroid_x"], m["centroid_y"])))
