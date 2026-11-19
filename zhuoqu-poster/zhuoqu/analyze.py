"""Deterministic image measurement + OCR + heuristic quality score.

Everything here is reproducible from pixels; it is *measured*, not judged.
Semantic labels (categories, techniques...) come from annotate.py and stay
review_required until a human approves them.

Usage:
    python -m zhuoqu.analyze [--ocr] [--force] [--ids heyt_0001,heyt_0002]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import paths
from .schema import Measured, Record, load, save

WORK_SIDE = 512          # measurements are done on a fixed-size copy -> resolution independent
BG_TOL = 28.0            # Lab distance treated as "same as background"

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def _load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"))


def _resize(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = WORK_SIDE / max(h, w)
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(round(x)) for x in c)


def border_background(rgb: np.ndarray) -> np.ndarray:
    """Median colour of a 4% border band - a robust estimate of paper/background."""
    h, w = rgb.shape[:2]
    b = max(2, int(min(h, w) * 0.04))
    band = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                           rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    return np.median(band, axis=0)


def dominant_colors(rgb: np.ndarray, k: int = 6, seed: int = 0) -> tuple[list[str], list[float]]:
    px = rgb.reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(px) > 20000:
        px = px[rng.choice(len(px), 20000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(float)
    order = np.argsort(-counts)
    w = counts[order] / counts.sum()
    return [_hex(centers[i]) for i in order], [round(float(x), 4) for x in w]


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric."""
    r, g, b = [rgb[..., i].astype(float) for i in range(3)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def measure_array(rgb_full: np.ndarray) -> dict:
    rgb = _resize(rgb_full)
    bg = border_background(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)[0, 0]
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    # whitespace = background-coloured AND locally flat (no edge within 3px)
    near_edge = cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0
    blank = (dist < BG_TOL) & ~near_edge
    ink = ~blank
    ys, xs = np.nonzero(ink)
    h, w = ink.shape
    cx = float(xs.mean() / w) if len(xs) else 0.5
    cy = float(ys.mean() / h) if len(ys) else 0.5
    dom, wts = dominant_colors(rgb)
    return dict(
        whitespace_ratio=round(float(blank.mean()), 4),
        edge_density=round(float((edges > 0).mean()), 4),
        centroid_x=round(cx, 4), centroid_y=round(cy, 4),
        colorfulness=round(colorfulness(rgb), 2),
        dominant_colors=dom, color_weights=wts,
        background_color=_hex(bg),
        blur_variance=round(float(cv2.Laplacian(cv2.cvtColor(rgb_full, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()), 2),
    )


def ocr_array(rgb: np.ndarray) -> tuple[str, float]:
    """Return (joined text, text-box area ratio). RapidOCR (PP-OCR ONNX) on CPU."""
    h, w = rgb.shape[:2]
    s = min(1.0, 1600 / max(h, w))
    small = cv2.resize(rgb, (round(w * s), round(h * s))) if s < 1 else rgb
    res, _ = _get_ocr()(small[..., ::-1])
    if not res:
        return "", 0.0
    area = 0.0
    texts = []
    for box, text, conf in res:
        if conf < 0.5:
            continue
        texts.append(text)
        area += cv2.contourArea(np.array(box, dtype=np.float32))
    return " | ".join(texts), round(min(1.0, area / (small.shape[0] * small.shape[1])), 4)


def quality(r: Record, m: dict) -> float:
    """Heuristic *technical* quality in [0,1] - resolution, sharpness, framing.
    It does not judge design merit; that is the human review's job."""
    short = min(r.width, r.height)
    res = min(1.0, short / 1000)
    sharp = min(1.0, np.log1p(m["blur_variance"]) / np.log1p(800))
    frame = {None: 1.0, "clean": 1.0, "photo_of_object": 0.85, "app_ui": 0.6, "composite": 0.5}[r.frame]
    return round(float(0.4 * res + 0.35 * sharp + 0.25 * frame), 3)


def run(ocr: bool = False, force: bool = False, ids: set[str] | None = None) -> dict:
    recs = load(paths.MANIFEST)
    done = fail = 0
    for r in recs:
        if r.review_status == "rejected" or (ids and r.id not in ids):
            continue
        if r.measured and not force and (not ocr or r.measured.ocr_text is not None):
            continue
        try:
            rgb = _load_rgb(paths.ROOT / r.local_path)
            m = measure_array(rgb)
            if ocr:
                m["ocr_text"], m["text_area_ratio"] = ocr_array(rgb)
            elif r.measured:
                m["ocr_text"], m["text_area_ratio"] = r.measured.ocr_text, r.measured.text_area_ratio
            r.measured = Measured(**m)
            r.quality_score = quality(r, m)
            if not r.color_palette:
                r.color_palette = [c for c, w in zip(m["dominant_colors"], m["color_weights"]) if w >= 0.03][:5]
            if r.annotator == "none":
                r.annotator = "measured"
            done += 1
        except Exception as e:  # keep going; report at the end
            fail += 1
            print(f"[analyze] {r.id}: {type(e).__name__}: {e}", file=sys.stderr)
        if done and done % 25 == 0:
            save(paths.MANIFEST, recs)   # checkpoint so an interrupted run resumes
    save(paths.MANIFEST, recs)
    return {"measured": done, "failed": fail}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ids")
    a = ap.parse_args(argv)
    print(run(a.ocr, a.force, set(a.ids.split(",")) if a.ids else None))


if __name__ == "__main__":
    main()
