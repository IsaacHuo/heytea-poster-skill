"""Merge annotation YAML files (data/annotations/*.yaml) into the manifest.

Precedence: human_*.yaml > agent_*.yaml (later file wins within a tier). Agent decisions:
- reject  -> review_status=rejected, review_note="agent: <reason>" (a human can revert)
- keep / composite -> review_status stays pending; only a human file may set approved.
Unknown vocab values raise, so typos are caught instead of silently stored.
"""
from __future__ import annotations

import yaml

from . import paths
from .schema import Composition, Record, load, save

_MAP = {"cat": "visual_categories", "typo": "typography_roles", "subj": "subject_tags",
        "tech": "creative_techniques", "brand": "brand_relation", "note": "style_description"}


def _files():
    files = sorted(paths.ANNOTATIONS.glob("agent_*.yaml")) + sorted(paths.ANNOTATIONS.glob("human_*.yaml"))
    return [(f, f.name.startswith("human_")) for f in files]


def apply() -> dict:
    recs = {r.id: r for r in load(paths.MANIFEST)}
    stats = {"annotated": 0, "rejected": 0, "kept": 0, "composite": 0, "approved": 0, "unknown_ids": []}
    for f, human in _files():
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for rid, a in data.items():
            if rid not in recs:
                stats["unknown_ids"].append(rid)
                continue
            d = recs[rid].model_dump()
            dec = a.get("decision", "keep")
            for k, v in a.items():
                if k in _MAP:
                    d[_MAP[k]] = v
                elif k == "comp":
                    d["composition"] = {**d["composition"], **v}
                elif k in ("campaign", "frame", "quality_score", "copyright_status"):
                    d[k] = v
            if dec == "composite":
                d["frame"] = "composite"
            if dec == "reject":
                d["review_status"] = "rejected"
                d["review_note"] = ("human: " if human else "agent: ") + str(a.get("reason", ""))
                if "not HEYTEA" in str(a.get("reason", "")) or "competitor" in str(a.get("reason", "")):
                    d["brand_relation"] = "not_heytea" if "not HEYTEA" in str(a.get("reason")) else "imitation"
            elif human and dec == "approve":
                d["review_status"] = "approved"
            elif d["review_status"] == "rejected" and (d.get("review_note") or "").startswith("agent:"):
                d["review_status"], d["review_note"] = "pending", None
            if d["annotator"] != "human":
                d["annotator"] = "human" if human else "agent_assisted"
            recs[rid] = Record.model_validate(d)
            stats["annotated"] += 1
    for r in recs.values():
        if r.review_status == "rejected":
            stats["rejected"] += 1
        elif r.review_status == "approved":
            stats["approved"] += 1
        elif r.frame == "composite":
            stats["composite"] += 1
        else:
            stats["kept"] += 1
    save(paths.MANIFEST, recs.values())
    return stats
