"""Human review: database stats, approve / reject / correct records.

Decisions are written to data/annotations/human_review.yaml (append-only, human wins over
agent annotations) and merged by annotate.apply(); the manifest is never edited by hand.
"""
from __future__ import annotations

import json
from collections import Counter

import yaml

from . import paths
from .annotate import apply
from .schema import load

HUMAN = paths.ANNOTATIONS / "human_review.yaml"


def stats() -> dict:
    rs = load(paths.MANIFEST)
    live = [r for r in rs if r.review_status != "rejected"]
    c = lambda f: dict(Counter(x for r in live for x in f(r)).most_common())  # noqa: E731
    return dict(total=len(rs), live=len(live), by_review=dict(Counter(r.review_status for r in rs)),
                brand=c(lambda r: [r.brand_relation]), campaign=c(lambda r: [r.campaign or "-"]),
                category=c(lambda r: r.visual_categories), technique=c(lambda r: r.creative_techniques),
                dated=sum(1 for r in live if r.published_at), measured=sum(1 for r in live if r.measured),
                embedded=sum(1 for r in live if r.embedding_status == "done"))


def decide(ids: list[str], decision: str, note: str | None = None, **fields) -> dict:
    if decision not in ("approve", "reject", "keep"):
        raise ValueError("decision must be approve | reject | keep")
    known = {r.id for r in load(paths.MANIFEST)}
    bad = [i for i in ids if i not in known]
    if bad:
        raise KeyError(f"unknown ids: {bad}")
    data = yaml.safe_load(HUMAN.read_text(encoding="utf-8")) if HUMAN.exists() else {}
    data = data or {}
    for i in ids:
        e = dict(data.get(i) or {}); e["decision"] = decision
        if note:
            e["reason" if decision == "reject" else "note"] = note
        e.update({k: v for k, v in fields.items() if v is not None})
        data[i] = e
    HUMAN.write_text("# Human review decisions (override agent annotations)\n" +
                     yaml.safe_dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8")
    return apply()


def show(rid: str) -> str:
    r = next((r for r in load(paths.MANIFEST) if r.id == rid), None)
    if r is None:
        raise KeyError(rid)
    return json.dumps(r.model_dump(mode="json"), ensure_ascii=False, indent=1)
