"""Image generation adapter.

Backends (selected by ZQ_IMAGE_BACKEND or argument):
- procedural (default, always available): doodle.py + scenes.py recipes, deterministic by seed.
- openai_compatible: POST {ZQ_IMAGE_BASE_URL}/images/generations with ZQ_IMAGE_API_KEY and
  ZQ_IMAGE_MODEL; the result is used as the subject illustration *without text* -
  all text is still rendered by typography.py.
If a remote backend is requested but not configured or fails, we fall back to procedural
and record the reason in the result (never silently).
"""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

from .doodle import Drawing
from .scenes import RECIPES


def available() -> dict:
    ok = bool(os.environ.get("ZQ_IMAGE_API_KEY") and os.environ.get("ZQ_IMAGE_BASE_URL"))
    return {"procedural": True, "openai_compatible": ok}


def _remote(prompt: str, size: tuple[int, int]) -> Image.Image:
    import requests
    base = os.environ["ZQ_IMAGE_BASE_URL"].rstrip("/")
    r = requests.post(f"{base}/images/generations", timeout=180,
                      headers={"Authorization": f"Bearer {os.environ['ZQ_IMAGE_API_KEY']}"},
                      json={"model": os.environ.get("ZQ_IMAGE_MODEL", "gpt-image-1"), "prompt": prompt,
                            "size": "1024x1024", "n": 1})
    r.raise_for_status()
    d = r.json()["data"][0]
    if "b64_json" in d:
        return Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))).convert("RGBA")
    return Image.open(io.BytesIO(requests.get(d["url"], timeout=120).content)).convert("RGBA")


def subject_layer(plan: dict, box, size: tuple[int, int], photo: Image.Image | None = None,
                  backend: str | None = None) -> dict:
    """Return {'image': RGBA canvas-size layer, 'backend': used, 'notes': [...]}."""
    W, H = size
    from .config import get
    backend = backend or os.environ.get("ZQ_IMAGE_BACKEND") or get("image", "backend", "procedural")
    notes = []
    if backend == "openai_compatible":
        if not available()["openai_compatible"]:
            notes.append("openai_compatible requested but ZQ_IMAGE_API_KEY/ZQ_IMAGE_BASE_URL not set -> procedural")
        else:
            try:
                im = _remote(plan["image_prompt"], size)
                x0, y0, x1, y1 = box
                s = min((x1 - x0) / im.width, (y1 - y0) / im.height)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                layer = Image.new("RGBA", size, (0, 0, 0, 0))
                layer.paste(im, ((x0 + x1 - im.width) // 2, (y0 + y1 - im.height) // 2))
                return dict(image=layer, backend=backend, notes=notes, photos=[])
            except Exception as e:
                notes.append(f"remote generation failed ({type(e).__name__}: {e}) -> procedural")
    short = min(W, H)
    d = Drawing(W, H, seed=plan["seed"], pen=max(3.0, 0.0055 * short), ink=plan["palette"]["ink"])
    ctx = dict(plan["palette"]); ctx["photo"] = photo
    res = RECIPES[plan["concept"]["concept"]](d, box, ctx) or {}
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    for ph, (px, py) in res.get("photos", []):
        layer.paste(ph.convert("RGBA"), (px, py), ph.convert("RGBA"))
    ink = d.render()
    layer = Image.alpha_composite(layer, ink)
    return dict(image=layer, backend="procedural", notes=notes, photos=res.get("photos", []), drawing=d)
