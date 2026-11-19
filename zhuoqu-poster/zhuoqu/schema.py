"""Reference image record schema (pydantic v2).

Rules:
- Unknown facts are null or an explicit pending/review status; never guessed.
- published_at is only set when the source page states a date (see collect.extract_date);
  it records the *article* publication date, which is an upper bound for the artwork date.
- Controlled vocabularies are Literal types so a typo fails validation.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Source = Literal[
    "official_weibo", "official_xiaohongshu", "official_wechat", "official_site",
    "design_media", "news_media", "designer_portfolio", "community_post", "manual_import",
]
Copyright = Literal["review_required", "research_only", "licensed", "public_domain", "rejected"]
Review = Literal["pending", "approved", "rejected"]
Embedding = Literal["pending", "done", "failed"]
VisualCategory = Literal[
    "photo_plus_drawing", "pure_illustration", "lettering_dominant",
    "documentary_photo_text", "graphic_dominant", "mixed", "non_poster",
]
TypographyRole = Literal["functional", "expressive", "hybrid", "none"]
CreativeTechnique = Literal[
    "personification", "humorous_contrast", "everyday_metaphor", "action_exaggeration",
    "scale_contrast", "visual_pun", "emotion", "unusual_combination", "nature_simplification",
    "collage", "none",
]
Subject = Literal[
    "drink", "fruit", "ingredient", "person", "animal", "landscape", "everyday_object",
    "abstract_shape", "packaging", "store", "character_ip",
]
BrandRelation = Literal["official", "official_collab", "imitation", "designer_portfolio", "unverified", "not_heytea"]
Frame = Literal["clean", "app_ui", "composite", "photo_of_object"]
Annotator = Literal["none", "measured", "vision_model", "agent_assisted", "human"]


class Measured(BaseModel):
    """Deterministic, reproducible image measurements (analyze.measure)."""
    model_config = ConfigDict(extra="forbid")
    whitespace_ratio: float = Field(ge=0, le=1)
    edge_density: float = Field(ge=0, le=1)
    centroid_x: float = Field(ge=0, le=1)
    centroid_y: float = Field(ge=0, le=1)
    colorfulness: float = Field(ge=0)
    dominant_colors: list[str]
    color_weights: list[float]
    background_color: str
    blur_variance: float = Field(ge=0)
    text_area_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    ocr_text: Optional[str] = None


class Composition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layout: Optional[str] = None            # e.g. subject_center, subject_offset, negative_space
    subject_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    visual_center: Optional[str] = None     # e.g. "center", "lower-left"
    whitespace: Optional[str] = None        # "large" | "medium" | "small"
    alignment: Optional[str] = None
    density: Optional[str] = None           # "sparse" | "medium" | "dense"
    text_image_hierarchy: Optional[str] = None


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^heyt_\d{4}$")
    source: Source
    source_url: str
    page_url: str
    page_title: Optional[str] = None
    published_at: Optional[date] = None
    published_at_basis: Optional[str] = None
    campaign: Optional[str] = None
    brand_relation: BrandRelation = "unverified"
    frame: Optional[Frame] = None
    local_path: str
    thumb_path: Optional[str] = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phash: Optional[str] = None
    duplicate_of: Optional[str] = None
    copyright_status: Copyright = "review_required"
    visual_categories: list[VisualCategory] = []
    typography_roles: list[TypographyRole] = []
    composition: Composition = Composition()
    illustration_features: list[str] = []
    color_palette: list[str] = []
    creative_techniques: list[CreativeTechnique] = []
    subject_tags: list[Subject] = []
    style_description: str = ""
    measured: Optional[Measured] = None
    annotator: Annotator = "none"
    embedding_status: Embedding = "pending"
    quality_score: Optional[float] = Field(default=None, ge=0, le=1)
    review_status: Review = "pending"
    review_note: Optional[str] = None
    fetched_at: str

    @field_validator("color_palette")
    @classmethod
    def _hex(cls, v: list[str]) -> list[str]:
        for c in v:
            if not (len(c) == 7 and c.startswith("#")):
                raise ValueError(f"color must be #rrggbb: {c}")
            int(c[1:], 16)
        return v


def load(path: Path) -> list[Record]:
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(Record.model_validate_json(line))
            except Exception as e:  # surface the line number, not a bare traceback
                raise ValueError(f"{path}:{n}: {e}") from e
    return out


def save(path: Path, records: Iterable[Record]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    tmp.replace(path)


def append(path: Path, record: Record) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def next_id(records: list[Record]) -> str:
    n = max((int(r.id[5:]) for r in records), default=0) + 1
    return f"heyt_{n:04d}"


def log_failure(path: Path, **kw) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kw, ensure_ascii=False) + "\n")
