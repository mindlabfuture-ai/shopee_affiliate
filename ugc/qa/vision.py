"""Claude vision review: listing photos vs sampled frames, feature by feature."""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from .. import llm
from ..imaging import Frame, to_jpeg_b64
from ..spec import ProductSpec
from .report import Check

SYSTEM = """\
You are a strict QA reviewer for fashion e-commerce videos. Compare REFERENCE listing photos (the real
product) with FRAMES from an AI-generated video. Marketplace rules forbid any change to the product's
appearance, so judge only garment fidelity, not video aesthetics.

For each listed feature decide:
- match: visible in the frames and consistent with the reference
- mismatch: visible but different (color, print, stitching, neckline, straps, length, cut/fullness, pockets, hardware)
- not_visible: never clearly visible in the supplied frames
Be literal. Do not assume a feature matches when you cannot see it; use not_visible. Report frame-level
problems (garment morphing, color shift, warped print, extra or missing parts) by frame label.
"""


class FeatureCheck(BaseModel):
    feature: str
    status: Literal["match", "mismatch", "not_visible"]
    evidence: str


class FrameIssue(BaseModel):
    frame_label: str
    issue: str


class VisionReview(BaseModel):
    features: list[FeatureCheck]
    frame_issues: list[FrameIssue]
    color_assessment: str
    fit_and_length_assessment: str
    summary: str


def pick_frames(frames: list[Frame], cuts: list[float] | None, n: int = 8) -> list[Frame]:
    idx = set(np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int).tolist())
    for c in (cuts or [])[:2]:  # frames either side of the biggest joins
        times = np.array([t for t, _ in frames])
        after = int(np.argmax(times >= c)) if (times >= c).any() else len(frames) - 1
        idx.update({max(after - 1, 0), after})
    return [frames[i] for i in sorted(idx)]


def review(client, spec: ProductSpec, reference: list[np.ndarray], frames: list[Frame]) -> VisionReview:
    feats = "\n".join(f"- {f.name}: {f.description}" for f in spec.features) or "- (none listed)"
    spec_text = (f"Product: {spec.name}\nColor: {spec.colorway} {spec.color_hex or ''}\nLength: {spec.length}\n"
                 f"Fabric: {spec.fabric}\nFeatures to check:\n{feats}\nDo-not-show: {', '.join(spec.avoid) or 'none'}")
    content: list[dict] = [{"type": "text", "text": spec_text + "\n\nREFERENCE listing photos follow."}]
    for i, img in enumerate(reference[:4], 1):
        content += [{"type": "text", "text": f"REFERENCE {i}"}, llm.image_block(to_jpeg_b64(img, 1024))]
    content.append({"type": "text", "text": "FRAMES follow."})
    for t, img in frames:
        content += [{"type": "text", "text": f"FRAME @ {t:.1f}s"}, llm.image_block(to_jpeg_b64(img, 768))]
    return llm.parse(client, VisionReview, SYSTEM, content)


def to_checks(spec: ProductSpec, r: VisionReview) -> list[Check]:
    must_show = {f.name for f in spec.features if f.show_in_video}
    mismatches = [f for f in r.features if f.status == "mismatch"]
    hidden = [f for f in r.features if f.status == "not_visible" and f.feature in must_show]
    out: list[Check] = []
    if mismatches:
        out.append(Check("vision_features", "fail", "; ".join(f"{m.feature}: {m.evidence}" for m in mismatches),
                         {"features": [f.model_dump() for f in r.features]}))
    elif hidden:
        out.append(Check("vision_features", "warn", "not shown: " + ", ".join(f.feature for f in hidden),
                         {"features": [f.model_dump() for f in r.features]}))
    else:
        out.append(Check("vision_features", "pass", "all listed features match or were not required",
                         {"features": [f.model_dump() for f in r.features]}))
    out.append(Check("vision_frames", "warn" if r.frame_issues else "pass",
                     "; ".join(f"{i.frame_label}: {i.issue}" for i in r.frame_issues) or "no frame-level issues",
                     {"issues": [i.model_dump() for i in r.frame_issues]}))
    out.append(Check("vision_notes", "info", f"{r.summary} | color: {r.color_assessment} | fit/length: {r.fit_and_length_assessment}"))
    return out
