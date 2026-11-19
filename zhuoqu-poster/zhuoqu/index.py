"""CLIP image index (open_clip ViT-B-32, laion2b). CPU-only, cached to data/indexes/.

Chinese queries are expanded to English tag phrases before CLIP text encoding
(the B-32 text tower is English-only); structured metadata scoring in retrieve.py
stays the primary signal, CLIP is an additive visual-semantic term.

Usage: python -m zhuoqu.index [--force]
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache

import numpy as np

from . import paths
from .schema import load, save

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"

ZH_EN = {
    "手写": "handwritten lettering", "标题": "title text", "极简": "minimal", "留白": "lots of white space",
    "拟人": "personified object with arms", "水果": "fruit", "手绘": "hand-drawn line doodle",
    "插画": "illustration", "摄影": "product photo", "照片": "photo", "产品": "drink cup product",
    "杯": "cup", "茶": "tea", "秋": "autumn", "中秋": "moon festival", "月": "moon", "小人": "stick figure",
    "简笔": "simple line drawing", "色块": "abstract color blocks", "地图": "hand-drawn map",
    "校园": "campus event", "活动": "event flyer", "春节": "chinese new year red", "夏": "summer",
    "葡萄": "grapes", "柿子": "persimmon", "芒果": "mango", "桃": "peach", "柠檬": "lemon",
}


@lru_cache(maxsize=1)
def _model():
    import open_clip, torch
    torch.set_num_threads(4)
    m, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    m.eval()
    return m, pre, open_clip.get_tokenizer(MODEL)


def to_english(q: str) -> str:
    words = [en for zh, en in ZH_EN.items() if zh in q]
    eng = " ".join(re.findall(r"[A-Za-z][A-Za-z ]+", q))
    return "a poster, " + ", ".join(words + ([eng] if eng else []))


def encode_text(qs: list[str]) -> np.ndarray:
    import torch
    m, _, tok = _model()
    with torch.no_grad():
        t = m.encode_text(tok([to_english(q) for q in qs])).float()
    t /= t.norm(dim=-1, keepdim=True)
    return t.numpy()


def encode_images(pil_images) -> np.ndarray:
    import torch
    m, pre, _ = _model()
    with torch.no_grad():
        x = torch.stack([pre(im.convert("RGB")) for im in pil_images])
        f = m.encode_image(x).float()
    f /= f.norm(dim=-1, keepdim=True)
    return f.numpy()


def build(force: bool = False) -> dict:
    from PIL import Image
    recs = load(paths.MANIFEST)
    paths.INDEXES.mkdir(parents=True, exist_ok=True)
    fe, fi = paths.INDEXES / "clip_image.npy", paths.INDEXES / "clip_ids.json"
    old = {}
    if fe.exists() and fi.exists() and not force:
        old = dict(zip(json.loads(fi.read_text(encoding="utf-8")), np.load(fe)))
    live = [r for r in recs if r.review_status != "rejected"]
    todo = [r for r in live if r.id not in old]
    failed = 0
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        ims, ok = [], []
        for r in batch:
            try:
                ims.append(Image.open(paths.ROOT / r.local_path)); ok.append(r)
            except Exception:
                r.embedding_status = "failed"; failed += 1
        if ims:
            for r, v in zip(ok, encode_images(ims)):
                old[r.id] = v
    ids = [r.id for r in live if r.id in old]
    for r in recs:
        if r.id in old and r.review_status != "rejected":
            r.embedding_status = "done"
    np.save(fe, np.stack([old[i] for i in ids]).astype(np.float32))
    fi.write_text(json.dumps(ids), encoding="utf-8")
    save(paths.MANIFEST, recs)
    return {"indexed": len(ids), "new": len(todo) - failed, "failed": failed, "model": f"{MODEL}/{PRETRAINED}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    print(build(ap.parse_args(argv).force))


if __name__ == "__main__":
    main()
