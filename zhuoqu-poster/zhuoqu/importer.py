"""Manual / authorised batch import (for sources the collector must not or cannot fetch:
login-walled social platforms, licensed press kits, your own photos).

Usage:
  python -m zhuoqu import <folder> [--source manual_import] [--page-url URL] [--brand official]
                          [--copyright licensed] [--published 2026-03-01] [--campaign NAME]
Optional <folder>/metadata.csv (utf-8) overrides per file; columns:
  filename,source,source_url,page_url,published_at,brand_relation,copyright_status,campaign,note
Nothing is guessed: missing dates stay null, provenance defaults to review_required.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import imagehash
from PIL import Image

from . import paths
from .collect import MAX_ASPECT, MIN_SHORT_SIDE
from .schema import Record, append, load, log_failure, next_id

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def import_folder(folder: str | Path, source: str = "manual_import", page_url: str = "", brand: str = "unverified",
                  copyright: str = "review_required", published: str | None = None, campaign: str | None = None) -> dict:
    paths.ensure_dirs()
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    meta = {}
    mf = folder / "metadata.csv"
    if mf.exists():
        with mf.open(encoding="utf-8-sig") as f:
            meta = {row["filename"]: row for row in csv.DictReader(f)}
    records = load(paths.MANIFEST)
    known_sha = {r.sha256: r.id for r in records}
    stats = dict(added=0, skipped=0, failed=0, ids=[])
    for fp in sorted(folder.iterdir()):
        if fp.suffix.lower() not in EXTS:
            continue
        m = meta.get(fp.name, {})
        try:
            raw = fp.read_bytes()
            im = Image.open(fp); im.load()
            w, h = im.size
            if min(w, h) < MIN_SHORT_SIDE or max(w, h) / min(w, h) > MAX_ASPECT:
                stats["skipped"] += 1; print(f"[skip] {fp.name}: size/aspect {w}x{h}"); continue
            sha = hashlib.sha256(raw).hexdigest()
            if sha in known_sha:
                stats["skipped"] += 1; print(f"[skip] {fp.name}: exact duplicate of {known_sha[sha]}"); continue
            rid = next_id(records)
            fmt = (im.format or "PNG").lower().replace("jpeg", "jpg")
            local = paths.ORIGINALS / f"{rid}.{fmt}"
            shutil.copyfile(fp, local)
            rgb = im.convert("RGB")
            thumb = paths.THUMBS / f"{rid}.jpg"
            t = rgb.copy(); t.thumbnail((512, 512)); t.save(thumb, quality=88)
            pub = m.get("published_at") or published
            rec = Record(id=rid, source=m.get("source") or source, source_url=m.get("source_url") or f"local:{fp.name}",
                         page_url=m.get("page_url") or page_url, published_at=date.fromisoformat(pub) if pub else None,
                         published_at_basis="supplied by importer" if pub else None,
                         campaign=m.get("campaign") or campaign, brand_relation=m.get("brand_relation") or brand,
                         copyright_status=m.get("copyright_status") or copyright,
                         local_path=str(local.relative_to(paths.ROOT)).replace("\\", "/"),
                         thumb_path=str(thumb.relative_to(paths.ROOT)).replace("\\", "/"), width=w, height=h, format=fmt,
                         sha256=sha, phash=str(imagehash.phash(rgb)), review_note=m.get("note") or None,
                         fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            append(paths.MANIFEST, rec); records.append(rec); known_sha[sha] = rid
            stats["added"] += 1; stats["ids"].append(rid)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            stats["failed"] += 1
            log_failure(paths.FAILED, kind="import", url=str(fp), error=f"{type(e).__name__}: {e}"[:300])
    return stats
