"""Near-duplicate detection with perceptual hashes.

Exact duplicates are already rejected at collect time by sha256. Here we cluster by pHash
Hamming distance (default <= 6 of 64 bits) and keep the highest-resolution member; the others
get duplicate_of=<kept id> and review_status=rejected with a note. Nothing is deleted from disk.
"""
from __future__ import annotations

import imagehash

from . import paths
from .schema import load, save


def dedupe(threshold: int = 6) -> dict:
    recs = load(paths.MANIFEST)
    live = [r for r in recs if r.phash and r.duplicate_of is None]
    hashes = {r.id: imagehash.hex_to_hash(r.phash) for r in live}
    parent = {r.id: r.id for r in live}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if hashes[a.id] - hashes[b.id] <= threshold:
                parent[find(b.id)] = find(a.id)
    groups: dict[str, list] = {}
    for r in live:
        groups.setdefault(find(r.id), []).append(r)
    marked = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = max(members, key=lambda r: r.width * r.height)
        for r in members:
            if r is keep:
                continue
            r.duplicate_of = keep.id
            r.review_status = "rejected"
            r.review_note = f"near-duplicate of {keep.id} (phash<= {threshold})"
            marked += 1
    save(paths.MANIFEST, recs)
    return {"records": len(recs), "clusters": sum(1 for g in groups.values() if len(g) > 1), "marked_duplicates": marked}
