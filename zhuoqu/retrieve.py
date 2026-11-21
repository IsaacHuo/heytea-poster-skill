"""Reference retrieval: structured-metadata + text scoring, optional CLIP similarity,
and MMR-style diversity (no two refs from the same campaign/page unless unavoidable).

Usage: python -m zhuoqu.retrieve "找到水果拟人化的手绘作品" -k 5
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache

import numpy as np

from . import paths
from .schema import load

# query word -> structured tags it implies
LEXICON = {
    r"手写|手绘字|手写字": dict(typo=["expressive"]),
    r"标题|文字主导|字为主|以字": dict(cat=["lettering_dominant"]),
    r"极简|留白|负空间|简约|少量文字": dict(ws=True),
    r"拟人|长出|手臂": dict(tech=["personification"]),
    r"水果|果": dict(subj=["fruit"]),
    r"产品|饮品|杯": dict(subj=["drink"]),
    r"摄影|照片|实拍": dict(cat=["photo_plus_drawing", "documentary_photo_text"]),
    r"插画|手绘|简笔|涂鸦": dict(cat=["pure_illustration", "photo_plus_drawing"]),
    r"夸张|巨大|尺度": dict(tech=["action_exaggeration", "scale_contrast"]),
    r"幽默|反差|好笑|俏皮": dict(tech=["humorous_contrast"]),
    r"隐喻|生活|日常": dict(tech=["everyday_metaphor"]),
    r"双关|谐音": dict(tech=["visual_pun"]),
    r"色块|抽象": dict(tech=["nature_simplification"], cat=["graphic_dominant"]),
    r"人物|小人": dict(subj=["person"]),
    r"原料|茶叶|食材": dict(subj=["ingredient"]),
    r"门店|空间|围挡": dict(subj=["store"]),
    r"活动|校园|市集|地图": dict(typo=["functional"]),
}
CONTRAST_CAMPAIGNS = {"retro_guochao_2019_2020", "melon_kuakua_2020"}
STORE_CAMPAIGNS = {"store_jinsha_cq_2026", "store_one_avenue_sz_2026"}


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"[^\u4e00-\u9fffa-z0-9]", "", (s or "").lower())
    return {s[i:i + 2] for i in range(len(s) - 1)}


@lru_cache(maxsize=1)
def _pool():
    rs = [r for r in load(paths.MANIFEST)
          if r.review_status != "rejected" and not r.duplicate_of
          and r.brand_relation in ("official", "official_collab", "unverified")]
    return rs


@lru_cache(maxsize=1)
def _clip_index():
    f = paths.INDEXES / "clip_image.npy"
    ids = paths.INDEXES / "clip_ids.json"
    if not (f.exists() and ids.exists()):
        return None
    return np.load(f), json.loads(ids.read_text(encoding="utf-8"))


def _clip_text_scores(query: str):
    idx = _clip_index()
    if idx is None:
        return None
    try:
        from .index import encode_text
        q = encode_text([query])[0]
    except Exception:
        return None
    emb, ids = idx
    return dict(zip(ids, (emb @ q).tolist()))


def parse_query(q: str) -> dict:
    want = dict(cat=set(), typo=set(), tech=set(), subj=set(), ws=False)
    for pat, tags in LEXICON.items():
        if re.search(pat, q):
            for k, v in tags.items():
                if k == "ws":
                    want["ws"] = True
                else:
                    want[k].update(v)
    return want


def score_record(r, want, qgrams, clip=None, allow_contrast=False):
    s, why = 0.0, []
    for key, field in (("cat", r.visual_categories), ("typo", r.typography_roles),
                       ("tech", r.creative_techniques), ("subj", r.subject_tags)):
        hit = want[key] & set(field)
        if hit:
            s += 2.0 * len(hit); why.append(f"{key}:{','.join(sorted(hit))}")
    m = r.measured
    if want["ws"] and m and m.whitespace_ratio >= 0.6:
        s += 2.0; why.append(f"whitespace {m.whitespace_ratio:.2f}")
    text = " ".join(filter(None, [r.style_description, r.review_note, r.page_title, m.ocr_text if m else None]))
    tg = qgrams & _bigrams(text)
    if tg:
        s += 0.8 * min(4, len(tg)); why.append("text:" + "/".join(sorted(tg)[:4]))
    if clip is not None and r.id in clip:
        s += 6.0 * clip[r.id]; why.append(f"clip {clip[r.id]:.3f}")
    if r.campaign in CONTRAST_CAMPAIGNS and not allow_contrast:
        s -= 4.0; why.append("penalty: 2019-20 retro (contrast sample)")
    if r.campaign in STORE_CAMPAIGNS and not want["subj"] & {"store"}:
        s -= 1.5
    if r.frame in ("app_ui", "composite"):
        s -= 0.3
    s += 0.5 * (r.quality_score or 0)
    return s, why


def retrieve(query: str, k: int = 5, filters: dict | None = None, diversity: float = 1.5,
             exclude: set | None = None) -> list[dict]:
    want = parse_query(query)
    if filters:
        for key in ("cat", "typo", "tech", "subj"):
            want[key].update(filters.get(key, []))
        want["ws"] = want["ws"] or bool(filters.get("ws"))
    allow_contrast = bool(re.search(r"复古|国潮|2019|2020|土味", query))
    qg = _bigrams(query)
    clip = _clip_text_scores(query)
    scored = []
    for r in _pool():
        if exclude and r.id in exclude:
            continue
        s, why = score_record(r, want, qg, clip, allow_contrast)
        scored.append((s, why, r))
    scored.sort(key=lambda t: -t[0])
    chosen, camp_n, page_n = [], {}, {}
    cand = scored[: max(60, k * 10)]
    while cand and len(chosen) < k:
        best = max(cand, key=lambda t: t[0] - diversity * camp_n.get(t[2].campaign or t[2].page_url, 0)
                   - 0.8 * page_n.get(t[2].page_url, 0))
        cand.remove(best)
        s, why, r = best
        camp_n[r.campaign or r.page_url] = camp_n.get(r.campaign or r.page_url, 0) + 1
        page_n[r.page_url] = page_n.get(r.page_url, 0) + 1
        chosen.append(dict(id=r.id, score=round(s, 3), reasons=why, campaign=r.campaign,
                           brand_relation=r.brand_relation, source=r.source, source_url=r.source_url,
                           page_url=r.page_url, local_path=r.local_path,
                           note=(r.style_description or r.review_note or "")[:160]))
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)
    for x in retrieve(a.query, a.k):
        print(json.dumps(x, ensure_ascii=False))


if __name__ == "__main__":
    main()
