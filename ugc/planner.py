"""Prompt-planner agent (Claude Sonnet): reads the product + listing photos and writes the avatar-image
prompt, per-clip video prompts and captions. Code, not the model, guarantees the garment-lock clause,
audio-off and clip lengths; a lint pass sends violations back to the model for revision."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from . import config, llm
from .imaging import read_image, to_jpeg_b64
from .rules import ALTER_RE, DISCLOSURE_CHECKLIST, POSES, RISKY_PHRASES, RULES_TEXT, SHOT_TAGS, VOICE_RE
from .spec import ProductSpec

NEGATIVE_PROMPT = (
    "changing garment color, altered print or pattern, different neckline, warped or morphing clothing, "
    "extra or missing straps, changed hem length, glowing or sparkling fabric, text, subtitles, watermark, "
    "logo, speech, dialogue, lip sync"
)


class ClipPlan(BaseModel):
    duration_s: int  # 4, 6 or 8
    shots: list[str]
    motion_prompt: str  # scene, camera and model motion only; never describes changes to the garment


class PromptPlan(BaseModel):
    setting: str
    avatar_image_prompt: str  # scene, pose, lighting, framing for the avatar still; garment is locked by code
    clips: list[ClipPlan]
    hook_captions: list[str]
    tiktok_caption: str
    shopee_caption: str
    risks: list[str]


SYSTEM = f"""\
You plan AI UGC videos for a Philippine ladies' fashion affiliate (TikTok Shop and Shopee). Buyers are
meticulous: the garment in the video must look exactly like the listing.

{RULES_TEXT}
You write ONLY: the setting, the avatar still's pose/scene/lighting/framing, per-clip motion and camera
prompts, and captions. The garment description is injected by code, so never describe changes to it.

Format: vertical 9:16, silent, one fictional avatar. Each clip is 4, 6 or 8 seconds; clips total about the
target duration. Cover every required shot across the clips using these tags: {', '.join(SHOT_TAGS)}.
Pose library (pick and combine; change pose roughly every 1-2 seconds, keep motion gentle so the garment
stays stable): {json.dumps(POSES)}.
Rules for motion prompts: describe the person's movement and the camera only. No speech, no music, no
text overlays, no brand names. Prefer soft, bright, blurred backgrounds that do not compete with the
garment. Use fabric_swish only for flowing fabric. If the listing shows pockets, include one natural
pocket pose. Include one full-length shot where the hem is visible.
Captions: truthful, matching the listing. Hooks: short ({{language}}), no exaggerated-result claims, no
'sexy'. Shopee and TikTok captions should include the AI-generated disclosure note.
`risks`: list anything about this product that could cause garment drift or a policy problem
(sheer fabric, fine prints, complex back, etc.).
"""


@dataclass
class Issue:
    level: str  # error | warn
    msg: str


def lock_clause(spec: ProductSpec) -> str:
    feats = "; ".join(f"{f.name}: {f.description}" for f in spec.features)
    parts = [
        f"The garment is the {spec.name}, {spec.colorway}" + (f" ({spec.color_hex})" if spec.color_hex else ""),
        f"length: {spec.length}" if spec.length else "",
        f"fabric: {spec.fabric}" if spec.fabric else "",
        f"details that must stay exactly as in the reference photo: {feats}" if feats else "",
    ]
    base = ", ".join(p for p in parts if p) + "."
    avoid = f" Never show: {', '.join(spec.avoid)}." if spec.avoid else ""
    return (
        f"{base} Reproduce the garment EXACTLY as in the reference photo in every frame. Do not alter its color, size, "
        "material, shape, print or pattern, stitching, neckline, straps, hem length or proportions. Do not add "
        f"accessories, logos, glow or lighting effects to it.{avoid}"
    )


def lint(spec: ProductSpec, plan: PromptPlan) -> list[Issue]:
    issues: list[Issue] = []
    total = sum(c.duration_s for c in plan.clips)
    for i, c in enumerate(plan.clips, 1):
        if c.duration_s not in config.CLIP_SECONDS:
            issues.append(Issue("error", f"clip {i}: duration must be one of {config.CLIP_SECONDS}, got {c.duration_s}"))
        for tag in c.shots:
            if tag not in SHOT_TAGS:
                issues.append(Issue("error", f"clip {i}: unknown shot tag {tag!r}"))
    if abs(total - spec.target_duration_s) > 2:
        issues.append(Issue("error", f"clips total {total}s, target is {spec.target_duration_s}s (+/-2)"))
    covered = {t for c in plan.clips for t in c.shots}
    missing = [s for s in spec.required_shots if s not in covered]
    if missing:
        issues.append(Issue("error", f"required shots not covered: {missing}"))

    texts = {"avatar_image_prompt": plan.avatar_image_prompt,
             **{f"clip {i} motion": c.motion_prompt for i, c in enumerate(plan.clips, 1)}}
    for name, t in texts.items():
        if m := ALTER_RE.search(t):
            issues.append(Issue("error", f"{name}: asks to change the garment ('{m.group(0)}')"))
        if m := VOICE_RE.search(t):
            issues.append(Issue("error", f"{name}: mentions speech/audio ('{m.group(0)}'); videos are silent"))
    captions = {**{f"hook {i}": h for i, h in enumerate(plan.hook_captions, 1)},
                "tiktok_caption": plan.tiktok_caption, "shopee_caption": plan.shopee_caption}
    for name, t in captions.items():
        for rx, why in RISKY_PHRASES:
            if rx.search(t):
                issues.append(Issue("error", f"{name}: {why}"))
    if not plan.hook_captions:
        issues.append(Issue("error", "need at least one hook caption"))
    for i, h in enumerate(plan.hook_captions, 1):
        if len(h.split()) > 10:
            issues.append(Issue("warn", f"hook {i} is long ({len(h.split())} words)"))
    for name in ("tiktok_caption", "shopee_caption"):
        if "ai" not in captions[name].lower():
            issues.append(Issue("warn", f"{name} has no AI-generated note"))
    return issues


def assemble(spec: ProductSpec, plan: PromptPlan, issues: list[Issue]) -> dict:
    lock = lock_clause(spec)
    return {
        "product": spec.slug,
        "model": config.CLAUDE_MODEL,
        "garment_lock": lock,
        "avatar_image": {"endpoint": config.IMAGE_ENDPOINT, "prompt": f"{lock} {plan.avatar_image_prompt}"},
        "clips": [
            {"index": i, "duration_s": c.duration_s, "shots": c.shots,
             "endpoint": config.VIDEO_ENDPOINT,
             "prompt": f"{lock} {c.motion_prompt}",
             "negative_prompt": NEGATIVE_PROMPT,
             "params": {"aspect_ratio": "9:16", "resolution": config.VIDEO_RESOLUTION,
                        "duration": f"{c.duration_s}s", "generate_audio": False}}
            for i, c in enumerate(plan.clips, 1)
        ],
        "setting": plan.setting,
        "hook_captions": plan.hook_captions,
        "tiktok_caption": plan.tiktok_caption,
        "shopee_caption": plan.shopee_caption,
        "risks": plan.risks,
        "lint": [{"level": i.level, "msg": i.msg} for i in issues],
        "disclosure_checklist": DISCLOSURE_CHECKLIST,
    }


def make_plan(spec: ProductSpec, listing: list[Path], client=None, max_revisions: int = 2) -> dict:
    client = client or llm.get_client()
    content: list[dict] = [{"type": "text", "text": (
        "Product spec (JSON):\n" + spec.model_dump_json(indent=2) +
        "\n\nListing photos follow. Plan the video." + (f"\nWrite hooks in {spec.language}." if spec.language else ""))}]
    for i, p in enumerate(listing[:6], 1):
        content += [{"type": "text", "text": f"LISTING PHOTO {i}"}, llm.image_block(to_jpeg_b64(read_image(p), 1024))]

    system = SYSTEM.replace("{language}", spec.language)
    plan = llm.parse(client, PromptPlan, system, content)
    issues = lint(spec, plan)
    for _ in range(max_revisions):
        errors = [i for i in issues if i.level == "error"]
        if not errors:
            break
        feedback = ("Your previous plan violated these rules. Return a corrected full plan.\n"
                    + "\n".join(f"- {e.msg}" for e in errors) + "\n\nPrevious plan:\n" + plan.model_dump_json(indent=2))
        plan = llm.parse(client, PromptPlan, system, content + [{"type": "text", "text": feedback}])
        issues = lint(spec, plan)
    return assemble(spec, plan, issues)
