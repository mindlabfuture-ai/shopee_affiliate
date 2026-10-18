"""Product spec + on-disk layout (data/products/<slug>/...)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from . import config
from .imaging import list_images


class Feature(BaseModel):
    name: str
    description: str
    show_in_video: bool = True  # must be visible at least once


class ProductSpec(BaseModel):
    slug: str
    name: str
    category: str = "dress"
    colorway: str
    color_hex: str | None = None  # strongly recommended; otherwise derived from listing photos
    length: str = ""
    fabric: str = ""
    features: list[Feature] = Field(default_factory=list)
    required_shots: list[str] = Field(default_factory=lambda: ["front", "back", "full_length"])
    avoid: list[str] = Field(default_factory=list)
    setting_ideas: list[str] = Field(default_factory=list)
    hook_ideas: list[str] = Field(default_factory=list)
    language: str = "Taglish"
    target_duration_s: int = 10
    notes: str = ""
    listing_url: str | None = None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "product"


def product_dir(slug: str) -> Path:
    if slugify(slug) != slug:
        raise ValueError(f"invalid slug: {slug!r}")
    return config.DATA_DIR / "products" / slug


def listing_dir(slug: str) -> Path:
    return product_dir(slug) / "listing"


def save_spec(spec: ProductSpec) -> Path:
    d = product_dir(spec.slug)
    listing_dir(spec.slug).mkdir(parents=True, exist_ok=True)
    p = d / "product.json"
    p.write_text(spec.model_dump_json(indent=2))
    return p


def load_spec(slug: str) -> ProductSpec:
    return ProductSpec.model_validate_json((product_dir(slug) / "product.json").read_text())


def list_slugs() -> list[str]:
    root = config.DATA_DIR / "products"
    return sorted(p.name for p in root.iterdir() if (p / "product.json").exists()) if root.exists() else []


def listing_images(slug: str) -> list[Path]:
    return list_images(listing_dir(slug))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default
