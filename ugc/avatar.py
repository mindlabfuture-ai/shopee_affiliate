"""Permanent avatar: one base photo -> a four-view identity sheet (close-up, front, left, right).

The face is locked in every prompt; only the hairstyle may change, and only if the user asks. Each view is
checked against the base photo by Claude (descriptive feature comparison, no identification of anyone).
The approved views are then passed as identity references when product avatar images are generated.

Assumes the avatar is your own AI-generated character, not a photo of a real person (platform rules and
CLAUDE.md rule 5: no real people's likeness without permission).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from . import config, llm
from .imaging import read_image, to_jpeg_b64

VIEW_ORDER = ("front", "closeup", "left", "right")  # reference priority and generation order
DISPLAY_ORDER = ("closeup", "front", "left", "right")
DEFAULT_HAIRSTYLE = "long sleek straight black hair with a soft side part"

FACE_LOCK = (
    "Keep the exact same woman as in the reference photo: identical face shape, eyes, eyebrows, nose, lips, "
    "jawline, skin tone and any moles or marks. Do not beautify, slim, age or restyle her face or body. "
    "Photorealistic, natural skin texture, soft natural light."
)

VIEWS = {
    "closeup": "Close-up portrait, head and shoulders, facing the camera directly, relaxed face with a faint "
               "natural smile, softly blurred neutral background.",
    "front": "Standing front view from head to knees, facing the camera straight on, shoulders square, arms "
             "relaxed at her sides, softly blurred neutral background. Same outfit as the reference photo.",
    "left": "Standing view from head to knees, body and head turned to show her left side in a three-quarter "
            "profile (about 75 degrees), her face turned toward the left edge of the image, softly blurred "
            "neutral background. Same outfit as the reference photo.",
    "right": "Standing view from head to knees, body and head turned to show her right side in a three-quarter "
             "profile (about 75 degrees), her face turned toward the right edge of the image, softly blurred "
             "neutral background. Same outfit as the reference photo.",
}


def avatar_dir() -> Path:
    return config.DATA_DIR / "avatar"


def sheet_dir() -> Path:
    return avatar_dir() / "sheet"


def base_path() -> Path:
    return avatar_dir() / "base.png"


def view_path(view: str) -> Path:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}")
    return sheet_dir() / f"{view}.png"


def view_prompt(view: str, hairstyle: str | None) -> str:
    hair = (f"Hairstyle: {hairstyle.strip()}." if hairstyle and hairstyle.strip()
            else "Keep the same hairstyle and hair color as the reference photo.")
    return f"{FACE_LOCK} {hair} {VIEWS[view]}"


def reference_files() -> list[Path]:
    """Identity references for product shoots: the approved sheet, else a legacy single avatar image."""
    sheet = [view_path(v) for v in VIEW_ORDER if view_path(v).exists()]
    legacy = config.DATA_DIR / "avatar_reference.png"
    return sheet or ([legacy] if legacy.exists() else [])


def save_base(img: np.ndarray) -> Path:
    avatar_dir().mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(base_path()), img)
    return base_path()


# ---------- your own images, removal ----------
def save_view(view: str, img: np.ndarray) -> None:
    """Put an image you made elsewhere straight into a slot. It is not re-checked: you chose it.
    The first image saved also becomes the base photo, so it can serve as the reference for missing views."""
    sheet_dir().mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(view_path(view)), img)
    (sheet_dir() / f"{view}.json").write_text(json.dumps(
        {"verdict": "provided", "summary": "uploaded by you, not generated or checked"}, indent=2))
    if not base_path().exists():
        save_base(img)


def remove_view(view: str) -> None:
    view_path(view).unlink(missing_ok=True)
    (sheet_dir() / f"{view}.json").unlink(missing_ok=True)


def clear_all() -> int:
    """Delete the whole permanent avatar (base, views, history, settings) and the old single-image avatar."""
    n = sum(1 for p in avatar_dir().rglob("*") if p.is_file()) if avatar_dir().exists() else 0
    shutil.rmtree(avatar_dir(), ignore_errors=True)
    legacy = config.DATA_DIR / "avatar_reference.png"
    if legacy.exists():
        legacy.unlink()
        n += 1
    return n


# ---------- consistency check ----------
class AttributeCheck(BaseModel):
    attribute: str
    status: Literal["same", "changed", "not_visible"]
    note: str


class FaceReview(BaseModel):
    attributes: list[AttributeCheck]
    hair_as_requested: bool
    summary: str


SYSTEM = """\
You compare two portraits of the same fictional character: REFERENCE (the original) and CANDIDATE (a new view
generated from it). Judge only whether the candidate keeps the same facial design. You are not identifying
anyone and must not name anyone. Compare these attributes: face shape and jawline, eyes and eyebrows, nose,
lips, skin tone, moles or marks, body build. Use not_visible when the angle hides it; use changed only for a
clear difference. Hair may differ only if the requested hairstyle says so; report whether it matches."""

CORE = ("face shape", "eyes", "nose", "lips")


def review_view(client, base: Path, candidate: Path, view: str, hairstyle: str | None) -> FaceReview:
    content = [
        {"type": "text", "text": f"View: {view}. Requested hairstyle: {hairstyle or 'unchanged from the reference'}."},
        {"type": "text", "text": "REFERENCE"}, llm.image_block(to_jpeg_b64(read_image(base), 1024)),
        {"type": "text", "text": "CANDIDATE"}, llm.image_block(to_jpeg_b64(read_image(candidate), 1024)),
    ]
    return llm.parse(client, FaceReview, SYSTEM, content, max_tokens=4000)


def verdict(r: FaceReview) -> str:
    changed = [a for a in r.attributes if a.status == "changed"]
    if any(any(k in a.attribute.lower() for k in CORE) for a in changed):
        return "drifted"
    return "check" if changed or not r.hair_as_requested else "consistent"


# ---------- generation ----------
def generate_view(prov, view: str, hairstyle: str | None, client=None) -> dict:
    """Generate one view, archive the one it replaces, store it as sheet/<view>.png and check it."""
    if not base_path().exists():
        raise RuntimeError("upload the base avatar photo first")
    refs = [base_path()] + ([view_path("front")] if view != "front" and view_path("front").exists() else [])
    staging = avatar_dir() / "staging"
    shutil.rmtree(staging, ignore_errors=True)
    out = prov.generate_avatar_images(view_prompt(view, hairstyle), refs, None, 1, staging, f"{view}")[0]
    sheet_dir().mkdir(parents=True, exist_ok=True)
    dest = view_path(view)
    if dest.exists():
        hist = sheet_dir() / "history"
        hist.mkdir(exist_ok=True)
        shutil.move(str(dest), hist / f"{view}_{int(time.time())}.png")
    shutil.move(str(out), dest)
    shutil.rmtree(staging, ignore_errors=True)
    check: dict = {"verdict": "unchecked", "summary": "no ANTHROPIC_API_KEY, so the face was not checked"}
    try:
        if client is not None:
            r = review_view(client, base_path(), dest, view, hairstyle)
            check = {"verdict": verdict(r), "summary": r.summary, "hair_as_requested": r.hair_as_requested,
                     "attributes": [a.model_dump() for a in r.attributes]}
    except Exception as e:  # a failed check must never lose the generated image
        check = {"verdict": "unchecked", "summary": f"check failed: {e}"}
    (sheet_dir() / f"{view}.json").write_text(json.dumps(check, indent=2))
    return check
