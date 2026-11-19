"""Load configs/default.yaml once; callers pass explicit arguments to override."""
from __future__ import annotations

from functools import lru_cache

import yaml

from . import paths


@lru_cache(maxsize=1)
def load() -> dict:
    f = paths.CONFIGS / "default.yaml"
    return yaml.safe_load(f.read_text(encoding="utf-8")) if f.exists() else {}


def get(section: str, key: str, default=None):
    return (load().get(section) or {}).get(key, default)
