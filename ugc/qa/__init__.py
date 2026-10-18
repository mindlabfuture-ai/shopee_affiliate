"""QA entry points: run_video_qa / run_image_qa."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config
from ..imaging import Frame, hex_to_lab, load_frames, read_image, derive_garment_lab
from ..rules import RISKY_PHRASES
from ..spec import ProductSpec
from . import color, technical, vision
from .report import Check, Report


def _target_lab(spec: ProductSpec, reference: list[np.ndarray]) -> tuple[np.ndarray, str]:
    if spec.color_hex:
        return hex_to_lab(spec.color_hex), "(from spec) "
    return derive_garment_lab(reference), "(derived from listing photos; set color_hex to be exact) "


def lint_caption(text: str) -> list[Check]:
    hits = [why for rx, why in RISKY_PHRASES if rx.search(text)]
    return [Check("caption", "warn" if hits else "pass", "; ".join(hits) or "no risky wording")]


def _run(spec: ProductSpec, listing: list[Path], frames: list[Frame], target: str, tech: list[Check],
         cuts: list[float], use_vision: bool, client, caption: str | None, meta: dict) -> Report:
    reference = [read_image(p) for p in listing]
    checks = list(tech)
    if not reference:
        checks.append(Check("listing_photos", "fail", "no listing photos for this product; upload some first"))
    else:
        lab, src = _target_lab(spec, reference)
        checks += color.check_color(frames, lab, cuts, src)
        if use_vision:
            from .. import llm
            client = client or llm.get_client()
            r = vision.review(client, spec, reference, vision.pick_frames(frames, cuts))
            checks += vision.to_checks(spec, r)
        else:
            checks.append(Check("vision", "skip", "vision review skipped; human review of every frame required"))
    if caption:
        checks += lint_caption(caption)
    meta |= {"checked_at": datetime.now(timezone.utc).isoformat(), "claude_model": config.CLAUDE_MODEL if use_vision else None}
    return Report(spec.slug, target, checks, meta)


def run_video_qa(spec: ProductSpec, listing: list[Path], video: Path, use_vision: bool = True,
                 client=None, caption: str | None = None) -> Report:
    tech, info = technical.check_technical(video)
    frames = load_frames(video)
    sha = hashlib.sha256(video.read_bytes()).hexdigest()
    return _run(spec, listing, frames, video.name, tech, info["cuts"], use_vision, client, caption,
                {"video_sha256": sha})


def run_image_qa(spec: ProductSpec, listing: list[Path], image: Path, use_vision: bool = True,
                 client=None) -> Report:
    frames = [(0.0, read_image(image))]
    return _run(spec, listing, frames, image.name, [], [], use_vision, client, None,
                {"image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
