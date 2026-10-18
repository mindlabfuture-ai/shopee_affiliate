"""Platform compliance rules as data. Source of truth is CLAUDE.md; keep the two in sync."""
from __future__ import annotations

import re

RULES_TEXT = """\
Compliance rules (Shopee Video AI guidelines, TikTok Shop AI content policy):
- The garment must match the real listing exactly: color, size, material, shape, print/pattern, stitching,
  neckline, straps, hem length, proportions, bundle contents. No fake 3D, glow or lighting effects on it.
- AI may change only the scene, the model and the camera, never the garment.
- Realistic results only. No exaggerated before/after or body-change claims.
- No invented stories, fake reviews, fake experts, real people's likeness, voice or trademarks, or minors.
- No speech or AI voice. Videos are silent; the seller adds licensed music in the app.
- Captions must be truthful and match the listing.
"""

# Actions a storyboard can use (name -> what it looks like). Variety matters: walking is allowed once per storyboard.
ACTIONS = {
    "pose_hold": "settles into a relaxed pose and holds it, soft smile, tiny natural movements",
    "turn_over_shoulder": "slow turn away from the camera, glancing back over the shoulder",
    "hair_touch": "tucks hair behind the ear or gives it a gentle toss",
    "hand_on_hip": "one hand on hip, shifts weight to one leg",
    "pocket_hand": "one hand slips naturally into the side pocket",
    "lean": "leans a shoulder against a wall, railing or tree and relaxes",
    "sit": "sits down on a bench, low wall or step and crosses her ankles (only if the garment suits it)",
    "hold_prop": "lifts a plain handbag or a cup from beside her, no brand names, never covering the garment",
    "look_up_smile": "looks up toward the light and breaks into a natural smile",
    "twirl": "a slow half turn so a flowing skirt or hem swings out (flowing fabric only)",
    "step_and_pause": "takes one step toward the camera and pauses",
    "walk": "walks toward the camera for a few steps (at most once per storyboard)",
    "wave_gesture": "a small friendly gesture toward the camera",
}

CAMERA_MOVES = {
    "static": "locked camera",
    "slow_push_in": "slow push in toward her",
    "slow_pull_back": "slow pull back to reveal more of her and the place",
    "tilt_up": "starts low on the hem and tilts up slowly to her face (shows the full length)",
    "handheld_follow": "gentle handheld camera that drifts with her movement",
    "low_angle": "slightly low camera angle looking up, steady",
}

SHOT_TAGS = ("front", "side", "back", "full_length", "pocket", "detail_closeup", "walk")

# Moderation or exaggeration risks in captions/hooks. (regex, why)
RISKY_PHRASES = [
    (re.compile(r"\bsexy\b", re.I), "'sexy' wording can trigger moderation; prefer 'elegant' or 'open-back'"),
    (re.compile(r"\b(slimming|pampayat|instantly|instant|guaranteed|magic|whitening)\b", re.I),
     "exaggerated-result wording"),
    (re.compile(r"\b(doctor|dermatologist|professor)\b", re.I), "invented authority figure"),
]

# Prompt text that asks the model to change the garment.
ALTER_RE = re.compile(
    r"\b(change|alter|redesign|recolou?r|improve|enhance|upgrade|restyle|modify|make)\b[^.]{0,30}"
    r"\b(dress|skirt|top|garment|outfit|clothes|clothing|fabric|neckline|straps?)\b",
    re.I,
)

# Prompt text that asks for speech or audio.
VOICE_RE = re.compile(r"\b(speak\w*|say\w*|talk\w*|voice\w*|dialogue|narrat\w*|lip[- ]?sync\w*|music|song|sing\w*)\b", re.I)

DISCLOSURE_CHECKLIST = [
    "Shopee: turn ON the 'AI-generated Inform' toggle before publishing (missing it = 3 penalty points + removal).",
    "TikTok: turn ON the AI-generated content toggle and add a caption note (e.g. 'AI-generated video').",
    "Audio is stripped; add licensed music inside TikTok.",
    "Check the final video once against the listing photos yourself before posting.",
    "Check the brand's own affiliate terms: some ban AI content and revoke commission.",
]
