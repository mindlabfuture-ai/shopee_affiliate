"""Optional market research: what other sellers' listings and buyers say about similar garments.

Research is context for the planner and for you. It never supplies product assets: competitor photos and
clips are not downloaded or used (CLAUDE.md rule 2), and review text is never quoted as a testimonial (rule 5).
"""
from __future__ import annotations

from ..spec import product_dir, read_json

RESEARCH_FILE = "research.json"
RAW_FILE = "research_raw.json"


def load_research(slug: str) -> dict | None:
    """Saved research for a product, or None."""
    return read_json(product_dir(slug) / RESEARCH_FILE)
