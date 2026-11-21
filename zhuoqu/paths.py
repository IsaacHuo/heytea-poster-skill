"""Central project paths. Everything is relative to the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ORIGINALS = DATA / "originals"
THUMBS = DATA / "thumbnails"
ANNOTATIONS = DATA / "annotations"
INDEXES = DATA / "indexes"
MANIFEST = DATA / "manifest.jsonl"
FAILED = DATA / "failed.jsonl"
SOURCES = DATA / "sources.yaml"
CONFIGS = ROOT / "configs"
FONTS = ROOT / "assets" / "fonts"
OUTPUTS = ROOT / "outputs"


def ensure_dirs() -> None:
    for p in (DATA, ORIGINALS, THUMBS, ANNOTATIONS, INDEXES, OUTPUTS):
        p.mkdir(parents=True, exist_ok=True)
