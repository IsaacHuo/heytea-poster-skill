"""Collect reference images from public pages listed in data/sources.yaml.

Properties:
- Idempotent / resumable: image URLs already in the manifest or in collect_state.json are skipped.
- Retries with exponential backoff; persistent failures go to data/failed.jsonl.
- Never logs in, never solves captchas, never spoofs cookies. Pages that need that are listed
  under `restricted` in sources.yaml instead.
- published_at is only filled from an explicit date found on the page; otherwise null.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from datetime import date, datetime, timezone
from urllib.parse import urljoin, urlparse

import imagehash
import requests
import yaml
from bs4 import BeautifulSoup
from PIL import Image

from . import paths
from .schema import Record, append, load, log_failure, next_id

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
MIN_SHORT_SIDE = 400
MAX_ASPECT = 3.2          # drop banners / long scroll screenshots
STATE = paths.DATA / "collect_state.json"
_BODY_SELECTORS = ["article", ".article-content", ".article_content", ".content-detail", ".detail-content",
                   ".rich_media_content", "#article", ".article", ".post-content", ".markdown-body", ".work", "main"]
_DATE_META = ["article:published_time", "og:release_date", "pubdate", "publishdate", "datePublished", "date"]


def _get(session: requests.Session, url: str, tries: int = 3, **kw) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = session.get(url, timeout=25, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - retried, then reported
            last = e
            time.sleep(1.5 * (2 ** i))
    raise RuntimeError(f"{type(last).__name__}: {last}")


def extract_date(soup: BeautifulSoup, html: str) -> tuple[date | None, str | None]:
    """Return (date, basis). Basis says where the date came from so it can be audited."""
    for key in _DATE_META:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key}) \
            or soup.find("meta", attrs={"itemprop": key})
        if tag and tag.get("content"):
            d = _parse_date(tag["content"])
            if d:
                return d, f"meta:{key}"
    m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
    if m and (d := _parse_date(m.group(1))):
        return d, "jsonld:datePublished"
    t = soup.find("time")
    if t and (d := _parse_date(t.get("datetime") or t.get_text(" "))):
        return d, "html:time"
    # Fallback: first explicit date in the visible header area (first 4000 chars of text).
    text = soup.get_text(" ")[:4000]
    if d := _parse_date(text):
        return d, "text:first_date_in_header"
    return None, None


def _parse_date(s: str) -> date | None:
    m = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", s)
    if not m:
        return None
    try:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    return d if date(2015, 1, 1) <= d <= date.today() else None


def extract_images(soup: BeautifulSoup, page_url: str, selector: str | None = None) -> list[str]:
    body = soup.select_one(selector) if selector else None
    for sel in ([] if body else _BODY_SELECTORS):
        body = soup.select_one(sel)
        if body and len(body.find_all("img")) >= 1:
            break
        body = None
    body = body or soup
    urls: list[str] = []
    for img in body.find_all("img"):
        src = img.get("data-src") or img.get("data-original") or img.get("data-lazy-src") or img.get("src") or ""
        if src.startswith("data:") or not src:
            continue
        u = urljoin(page_url, src.strip())
        if re.search(r"(logo|avatar|icon|qrcode|erweima|emoji|sprite)", u, re.I):
            continue
        if u not in urls:
            urls.append(u)
    return urls


def _load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"skipped": {}}


def _save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=0), encoding="utf-8")


def collect(limit_pages: int | None = None, delay: float = 0.6, verbose: bool = True) -> dict:
    paths.ensure_dirs()
    cfg = yaml.safe_load(paths.SOURCES.read_text(encoding="utf-8"))
    records = load(paths.MANIFEST)
    known_urls = {r.source_url for r in records}
    known_sha = {r.sha256: r.id for r in records}
    state = _load_state()
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
    stats = {"pages": 0, "page_errors": 0, "candidates": 0, "added": 0, "skipped": 0, "exact_dupes": 0, "img_errors": 0}

    for page in cfg.get("pages", [])[: limit_pages or None]:
        purl = page["url"]
        try:
            r = _get(s, purl)
        except Exception as e:  # noqa: BLE001
            stats["page_errors"] += 1
            log_failure(paths.FAILED, kind="page", url=purl, error=str(e), at=_now())
            continue
        r.encoding = r.apparent_encoding if (r.encoding or "").lower() in ("iso-8859-1", "") else r.encoding
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.get_text(strip=True) if soup.title else None) or None
        pub, basis = extract_date(soup, html)
        imgs = extract_images(soup, purl, page.get("selector"))
        if page.get("max_images"):
            imgs = imgs[: int(page["max_images"])]
        stats["pages"] += 1
        stats["candidates"] += len(imgs)
        if verbose:
            print(f"[page] {len(imgs):3d} imgs  date={pub} ({basis})  {purl}")
        for iu in imgs:
            if iu in known_urls or iu in state["skipped"]:
                continue
            try:
                ir = _get(s, iu, headers={"Referer": purl})
                raw = ir.content
                im = Image.open(io.BytesIO(raw))
                im.load()
            except Exception as e:  # noqa: BLE001
                stats["img_errors"] += 1
                log_failure(paths.FAILED, kind="image", url=iu, page=purl, error=str(e)[:200], at=_now())
                continue
            w, h = im.size
            reason = None
            if min(w, h) < MIN_SHORT_SIDE:
                reason = f"too_small {w}x{h}"
            elif max(w, h) / min(w, h) > MAX_ASPECT:
                reason = f"extreme_aspect {w}x{h}"
            sha = hashlib.sha256(raw).hexdigest()
            if not reason and sha in known_sha:
                reason = f"exact_duplicate_of {known_sha[sha]}"
                stats["exact_dupes"] += 1
            if reason:
                state["skipped"][iu] = reason
                stats["skipped"] += 1
                continue
            rid = next_id(records)
            fmt = (im.format or "PNG").lower().replace("jpeg", "jpg")
            if fmt not in ("jpg", "png", "webp", "gif"):
                fmt = "png"
            local = paths.ORIGINALS / f"{rid}.{fmt}"
            local.write_bytes(raw)
            rgb = im.convert("RGB")
            thumb = paths.THUMBS / f"{rid}.jpg"
            t = rgb.copy()
            t.thumbnail((512, 512))
            t.save(thumb, quality=88)
            rec = Record(
                id=rid, source=page["source"], source_url=iu, page_url=purl, page_title=title,
                published_at=pub, published_at_basis=(f"article date via {basis}" if pub else None),
                campaign=page.get("campaign"), local_path=str(local.relative_to(paths.ROOT)).replace("\\", "/"),
                thumb_path=str(thumb.relative_to(paths.ROOT)).replace("\\", "/"), width=w, height=h, format=fmt,
                sha256=sha, phash=str(imagehash.phash(rgb)), fetched_at=_now(),
            )
            append(paths.MANIFEST, rec)
            records.append(rec)
            known_urls.add(iu)
            known_sha[sha] = rid
            stats["added"] += 1
            time.sleep(delay)
        _save_state(state)
    return stats


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
