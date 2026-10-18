"""Prompt-planner agent (Claude Sonnet): reads the product + listing photos and writes several outdoor
storyboards (each with its avatar-image prompt and per-clip video prompts) plus captions. Code, not the
model, guarantees the garment-lock clause, audio-off and clip lengths; a lint pass sends violations back
to the model for revision."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from . import config, llm, motion
from .imaging import read_image, to_jpeg_b64
from .rules import (ACTIONS, ALTER_RE, CAMERA_MOVES, DISCLOSURE_CHECKLIST, RISKY_PHRASES, RULES_TEXT, SHOT_TAGS,
                    VOICE_RE)
from .spec import ProductSpec

MIN_STORYBOARDS = 3

NEGATIVE_PROMPT = (
    "changing garment color, altered print or pattern, different neckline, warped or morphing clothing, "
    "extra or missing straps, changed hem length, glowing or sparkling fabric, text, subtitles, watermark, "
    "logo, speech, dialogue, lip sync"
)


class ClipPlan(BaseModel):
    duration_s: int  # one of config.allowed_clip_seconds()
    shots: list[str]
    action: str  # a key of rules.ACTIONS
    camera: str  # a key of rules.CAMERA_MOVES
    start_state: str  # pose, position and framing at the first frame (the previous clip's end_state, word for word)
    end_state: str  # pose, position and framing at the last frame
    motion_prompt: str  # how she moves from start_state to end_state; never describes changes to the garment


class Storyboard(BaseModel):
    title: str
    arc: str  # the shoot as one continuous take, in a sentence, e.g. "settle, show the back, lean, look up"
    location: str  # a specific outdoor place, e.g. "tropical garden path"
    time_of_day: str
    mood: str
    avatar_image_prompt: str  # pose, outdoor scene, lighting, framing for the still; garment is locked by code
    clips: list[ClipPlan]


class PromptPlan(BaseModel):
    storyboards: list[Storyboard]
    hook_captions: list[str]
    tiktok_caption: str
    shopee_caption: str
    risks: list[str]


SYSTEM = f"""\
You plan AI UGC videos for a Philippine ladies' fashion affiliate (TikTok Shop and Shopee). Buyers are
meticulous: the garment in the video must look exactly like the listing.

{RULES_TEXT}
You write ONLY: storyboards, the avatar still's pose/scene/lighting/framing, per-clip motion and camera
prompts, and captions. The garment description is injected by code, so never describe changes to it.

STORYBOARDS (required): the outfit in the uploaded photos is featured in an outdoor fashion shoot. Write
{MIN_STORYBOARDS} to 4 clearly DIFFERENT storyboards, each its own complete shoot concept:
- a distinct, specific outdoor location per storyboard (examples: tropical garden path, city street with
  soft bokeh, cafe terrace, seaside promenade, park lawn, old-town stone lane, rooftop at golden hour);
  never repeat a location;
- vary time of day, light and mood (golden hour, bright midday shade, soft overcast, blue hour);
- choose locations that suit this garment's style and keep the garment stable: no rain, strong wind,
  water, mud or busy crowds that could distort or hide it; backgrounds soft and blurred, never competing
  with the garment;
- every storyboard covers all required shots and has its own avatar still prompt and clips;
- the same fictional avatar and the same outfit in every storyboard; only the shoot concept changes.

Format: vertical 9:16, silent. Each clip is {{clip_seconds}} seconds; a storyboard's clips total about the target
duration (use more clips when each is short). Shot tags: {', '.join(SHOT_TAGS)}.

CONTINUITY (required). A storyboard's clips are generated one after another, and every clip starts from the LAST
FRAME of the previous clip, so the storyboard must play as one continuous take:
- clips[0].start_state is the pose, framing and place of your avatar still prompt.
- Each clip's end_state is exactly how its last frame looks; the next clip's start_state must repeat it WORD FOR
  WORD. Same place, same light, no scene jumps, no cuts.
- motion_prompt describes how she gets from start_state to end_state, plus the camera. Nothing else changes.

VARIETY (required). The video must not be just walking. Give every clip an `action` from this library:
{json.dumps(ACTIONS)}
and a `camera` from this one: {json.dumps(CAMERA_MOVES)}
Use at least three different actions in a storyboard with three or more clips (two with two clips), never the
same action in consecutive clips, and `walk` at most once. Build a small arc (for example settle, turn to show the
back, hand in pocket, lean, look up, step toward the camera) and put it in `arc`. Alternate the camera moves too.
Keep every move slow and gentle so the garment stays stable: no fast spins, running, jumping, big arm swings or
hands covering the garment. Pick only actions that suit this garment (twirl only for flowing fabric; sit only if
the length and fabric allow it without bunching). Include one natural pocket moment if the listing shows pockets,
and one full-length shot where the hem is visible.
Rules for prompts: describe the person's movement and the camera only. No speech, no music, no text overlays,
no brand names.
Captions: truthful, matching the listing. Hooks: short ({{language}}), no exaggerated-result claims, no
'sexy'. Shopee and TikTok captions should include the AI-generated disclosure note.
`risks`: list anything about this product that could cause garment drift or a policy problem
(sheer fabric, fine prints, complex back, etc.).
"""


REFERENCE_RULES = """\

MOTION REFERENCE (supplied below). The user wants the avatar to COPY this choreography. Every storyboard must
reproduce its beats in the same order, at the same pace and with the same camera behaviour, adapted to this outfit
and to that storyboard's location (the places still differ). Make the clips mirror the beat timings: the clips of a
storyboard total about the reference duration; split or merge beats to fit the allowed clip lengths. Map each beat
to the nearest library action and camera, and put the specifics of the movement in motion_prompt. Copy only the
movement and camera, never anything about the person, clothing or place in the reference. The continuity rules
stay strict: one continuous take, start_state equal to the previous end_state. If the reference repeats an action
or walks more than once, follow the reference (the variety rules are relaxed)."""


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


def _norm(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


def _variety_and_continuity(tag: str, sb: Storyboard, relax_variety: bool = False) -> list[Issue]:
    out: list[Issue] = []
    variety = "warn" if relax_variety else "error"  # a motion reference may legitimately repeat or walk
    actions = [c.action for c in sb.clips]
    if len(set(actions)) < min(3, len(actions)):
        out.append(Issue(variety, f"{tag}: use at least {min(3, len(actions))} different actions, got {sorted(set(actions))}"))
    if actions.count("walk") > 1:
        out.append(Issue(variety, f"{tag}: 'walk' may be used at most once; the video must not be just walking"))
    for i in range(1, len(sb.clips)):
        if sb.clips[i].action == sb.clips[i - 1].action:
            out.append(Issue(variety, f"{tag}: clips {i} and {i + 1} repeat the action {sb.clips[i].action!r}"))
        if _norm(sb.clips[i].start_state) != _norm(sb.clips[i - 1].end_state):
            out.append(Issue("error", f"{tag}: clip {i + 1} start_state must repeat clip {i} end_state word for word "
                                      "(each clip starts from the previous clip's last frame)"))
    if len(sb.clips) > 1 and len({c.camera for c in sb.clips}) == 1:
        out.append(Issue("warn", f"{tag}: every clip uses the camera {sb.clips[0].camera!r}; vary it"))
    return out


def lint(spec: ProductSpec, plan: PromptPlan, reference: dict | None = None,
         clip_limit: float | None = None) -> list[Issue]:
    issues: list[Issue] = []
    target = round(reference["duration_s"]) if reference else spec.target_duration_s
    sbs = plan.storyboards
    if len(sbs) < MIN_STORYBOARDS:
        issues.append(Issue("error", f"need at least {MIN_STORYBOARDS} different outdoor storyboards, got {len(sbs)}"))
    locs = [s.location.strip().lower() for s in sbs]
    if len(set(locs)) < len(locs):
        issues.append(Issue("error", "storyboards must use different outdoor locations"))
    if len(sbs) > 1 and len({s.time_of_day.strip().lower() for s in sbs}) == 1:
        issues.append(Issue("warn", "all storyboards share one time of day; vary the light"))

    texts: dict[str, str] = {}
    for n, sb in enumerate(sbs, 1):
        tag = f"storyboard {n}"
        total = sum(c.duration_s for c in sb.clips)
        if abs(total - target) > 2:
            issues.append(Issue("error", f"{tag}: clips total {total}s, target is {target}s (+/-2)"
                                         + (" to match the motion reference" if reference else "")))
        covered = {t for c in sb.clips for t in c.shots}
        missing = [s for s in spec.required_shots if s not in covered]
        if missing:
            issues.append(Issue("error", f"{tag}: required shots not covered: {missing}"))
        for i, c in enumerate(sb.clips, 1):
            if c.duration_s not in config.allowed_clip_seconds(clip_limit):
                issues.append(Issue("error", f"{tag} clip {i}: duration must be one of "
                                             f"{config.allowed_clip_seconds(clip_limit)}, got {c.duration_s}"))
            for t in c.shots:
                if t not in SHOT_TAGS:
                    issues.append(Issue("error", f"{tag} clip {i}: unknown shot tag {t!r}"))
            texts[f"{tag} clip {i} motion"] = c.motion_prompt
            texts[f"{tag} clip {i} start_state"] = c.start_state
            texts[f"{tag} clip {i} end_state"] = c.end_state
            if c.action not in ACTIONS:
                issues.append(Issue("error", f"{tag} clip {i}: unknown action {c.action!r}; use one of {sorted(ACTIONS)}"))
            if c.camera not in CAMERA_MOVES:
                issues.append(Issue("error", f"{tag} clip {i}: unknown camera {c.camera!r}; use one of {sorted(CAMERA_MOVES)}"))
            if not c.start_state.strip() or not c.end_state.strip():
                issues.append(Issue("error", f"{tag} clip {i}: start_state and end_state must both be filled in"))
        issues += _variety_and_continuity(tag, sb, relax_variety=bool(reference))
        texts[f"{tag} avatar_image_prompt"] = sb.avatar_image_prompt
        texts[f"{tag} location"] = f"{sb.location} {sb.mood}"

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


def clip_prompt(lock: str, c: ClipPlan) -> str:
    """Code builds the continuity wording, so it never depends on the model remembering it."""
    return (f"{lock} The first frame is the exact starting state: {c.start_state.strip().rstrip('.')}. "
            f"{c.motion_prompt.strip()} Camera: {CAMERA_MOVES.get(c.camera, c.camera)}. "
            f"End in this state: {c.end_state.strip().rstrip('.')}. One continuous take, no cuts; the place, lighting "
            "and the garment stay exactly the same throughout.")


def assemble(spec: ProductSpec, plan: PromptPlan, issues: list[Issue], reference: dict | None = None) -> dict:
    lock = lock_clause(spec)
    return {
        "product": spec.slug,
        "reference_used": bool(reference),
        "model": config.CLAUDE_MODEL,
        "garment_lock": lock,
        "storyboards": [
            {
                "index": n, "title": sb.title, "location": sb.location, "time_of_day": sb.time_of_day, "mood": sb.mood,
                "avatar_image": {"endpoint": config.IMAGE_ENDPOINT, "prompt": f"{lock} {sb.avatar_image_prompt}"},
                "arc": sb.arc,
                "clips": [
                    {"index": i, "duration_s": c.duration_s, "shots": c.shots, "action": c.action, "camera": c.camera,
                     "start_state": c.start_state, "end_state": c.end_state, "chained": i > 1,
                     "endpoint": config.VIDEO_ENDPOINT,
                     "prompt": clip_prompt(lock, c),
                     "negative_prompt": NEGATIVE_PROMPT,
                     "params": {"aspect_ratio": "9:16", "resolution": config.VIDEO_RESOLUTION,
                                "duration": f"{c.duration_s}s", "generate_audio": False}}
                    for i, c in enumerate(sb.clips, 1)
                ],
            }
            for n, sb in enumerate(plan.storyboards)
        ],
        "hook_captions": plan.hook_captions,
        "tiktok_caption": plan.tiktok_caption,
        "shopee_caption": plan.shopee_caption,
        "risks": plan.risks,
        "lint": [{"level": i.level, "msg": i.msg} for i in issues],
        "disclosure_checklist": DISCLOSURE_CHECKLIST,
    }


def make_plan(spec: ProductSpec, listing: list[Path], client=None, max_revisions: int = 2,
              reference: dict | None = None, clip_limit: float | None = None) -> dict:
    client = client or llm.get_client()
    content: list[dict] = [{"type": "text", "text": (
        "Product spec (JSON):\n" + spec.model_dump_json(indent=2) +
        "\n\nListing photos follow. Plan the outdoor fashion shoot storyboards for this outfit."
        + (f"\nWrite hooks in {spec.language}." if spec.language else ""))}]
    for i, p in enumerate(listing[:6], 1):
        content += [{"type": "text", "text": f"LISTING PHOTO {i}"}, llm.image_block(to_jpeg_b64(read_image(p), 1024))]

    secs = config.allowed_clip_seconds(clip_limit)
    if reference:
        content.append({"type": "text", "text": motion.to_prompt_block(reference)})
    system = ((SYSTEM + (REFERENCE_RULES if reference else "")).replace("{language}", spec.language)
              .replace("{clip_seconds}", ", ".join(map(str, secs[:-1])) + (" or " if len(secs) > 1 else "") + str(secs[-1])))
    plan = llm.parse(client, PromptPlan, system, content, max_tokens=12000)
    issues = lint(spec, plan, reference, clip_limit)
    for _ in range(max_revisions):
        errors = [i for i in issues if i.level == "error"]
        if not errors:
            break
        feedback = ("Your previous plan violated these rules. Return a corrected full plan.\n"
                    + "\n".join(f"- {e.msg}" for e in errors) + "\n\nPrevious plan:\n" + plan.model_dump_json(indent=2))
        plan = llm.parse(client, PromptPlan, system, content + [{"type": "text", "text": feedback}], max_tokens=12000)
        issues = lint(spec, plan, reference, clip_limit)
    return assemble(spec, plan, issues, reference)
