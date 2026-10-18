"""Reverse-engineer a short reference clip into a motion script (choreography + camera), so the planner can make
the avatar copy the moves.

Only movement and camera are kept. The model is told to ignore who is in the clip, their looks, clothing, the
place, brands and any text, and no pixels from the clip ever reach the video generator, so nothing of the person
in the reference (face, outfit) can transfer. Use clips you own or have the right to use."""
from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel

from . import llm
from .imaging import load_frames, to_jpeg_b64
from .qa import technical
from .rules import ACTIONS, CAMERA_MOVES

MAX_REF_SECONDS = 30.0
MAX_REF_BYTES = 100 * 1024 * 1024


class MotionBeat(BaseModel):
    start_s: float
    end_s: float
    action: str  # a key of rules.ACTIONS, or "other"
    action_detail: str  # exactly what the body does, movement only
    camera: str  # a key of rules.CAMERA_MOVES, or "other"
    camera_detail: str
    start_state: str  # body pose and framing at the start of the beat
    end_state: str  # body pose and framing at the end of the beat


class MotionReference(BaseModel):
    pacing: str  # e.g. "slow and relaxed, one move every 2-3 seconds"
    camera_summary: str
    beats: list[MotionBeat]
    notes: str  # anything a planner must keep in mind, e.g. "fast hair toss at 3 s"


SYSTEM = f"""\
You turn a short reference video into a motion script that another person can recreate. Describe ONLY the
choreography and the camera: what the body does, in what order, how fast, and how the camera moves.
Never describe or mention who the person is, their face, body, hair, clothing, accessories, the place, brands or
any text or logos in the video. Never name anyone. Say "she" or "the subject".

Split the clip into 3 to 8 beats in time order, covering the whole duration without gaps. For each beat pick the
nearest `action` from this library (use "other" only if nothing fits, and then explain in action_detail):
{", ".join(ACTIONS)}
and the nearest `camera` from: {", ".join(CAMERA_MOVES)} (or "other").
start_state and end_state describe pose and framing in plain words (for example "standing facing the camera, arms
relaxed, full-length framing"); each beat's start_state must repeat the previous beat's end_state. Be literal
about speed and direction; the frames are sampled, so infer motion from change between them."""


def analyze(client, video: Path, max_frames: int = 16) -> dict:
    """Returns the motion script as a plain dict (also adds duration_s and detected cut times)."""
    if not (shutil.which("ffprobe") and shutil.which("ffmpeg")):
        raise RuntimeError("ffmpeg/ffprobe are not installed")
    meta = technical.probe(video)
    duration = float(meta["format"].get("duration") or 0)
    if duration <= 0:
        raise ValueError("cannot read the reference clip")
    if duration > MAX_REF_SECONDS:
        raise ValueError(f"the reference clip is {duration:.0f}s; keep it under {MAX_REF_SECONDS:.0f}s")
    cuts = technical.scene_cuts(video)
    frames = load_frames(video, max_frames=max_frames, height=640)
    content: list[dict] = [{"type": "text", "text": (
        f"Reference clip: {duration:.1f} s, {len(frames)} sampled frames"
        + (f", camera cuts at {', '.join(f'{c:.1f}s' for c in cuts)}" if cuts else ", no camera cuts") + ".")}]
    for t, img in frames:
        content += [{"type": "text", "text": f"FRAME @ {t:.1f}s"}, llm.image_block(to_jpeg_b64(img, 640))]
    ref = llm.parse(client, MotionReference, SYSTEM, content, max_tokens=6000).model_dump()
    beats = []
    for b in sorted(ref["beats"], key=lambda b: b["start_s"]):
        b["start_s"] = round(min(max(b["start_s"], 0.0), duration), 2)
        b["end_s"] = round(min(max(b["end_s"], b["start_s"]), duration), 2)
        if b["action"] not in ACTIONS:
            b["action"] = "other"
        if b["camera"] not in CAMERA_MOVES:
            b["camera"] = "other"
        beats.append(b)
    return {**ref, "beats": beats, "duration_s": round(duration, 1), "cuts": [round(c, 2) for c in cuts]}


def to_prompt_block(ref: dict) -> str:
    """The motion script as text for the planner."""
    lines = [f"MOTION REFERENCE ({ref['duration_s']} s; pacing: {ref['pacing']}; camera: {ref['camera_summary']})"]
    for i, b in enumerate(ref["beats"], 1):
        lines.append(f"{i}. {b['start_s']:.1f}-{b['end_s']:.1f}s | action: {b['action']} ({b['action_detail']}) | "
                     f"camera: {b['camera']} ({b['camera_detail']}) | starts: {b['start_state']} | ends: {b['end_state']}")
    if ref.get("cuts"):
        lines.append("The reference has camera cuts at " + ", ".join(f"{c:.1f}s" for c in ref["cuts"])
                     + "; ours must stay one continuous take, so express those moments as camera moves instead.")
    if ref.get("notes"):
        lines.append(f"Notes: {ref['notes']}")
    return "\n".join(lines)
