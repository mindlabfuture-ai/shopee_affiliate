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

# Pose library: name -> what it shows. The planner picks from these.
POSES = {
    "front_stand": "standing facing camera, relaxed, soft smile",
    "slow_turn": "slow quarter turn to show the side profile",
    "back_view": "turns to show the back, glances over the shoulder",
    "hand_on_hip": "one hand on hip, weight on one leg",
    "hair_touch": "touches hair lightly",
    "gesture": "open palm gesture toward the camera",
    "walk": "two or three slow steps toward the camera",
    "pocket_hand": "one hand resting naturally in the side pocket",
    "fabric_swish": "gentle sway so the skirt or hem moves (only for flowing fabric)",
    "detail_closeup": "slow push-in on a key construction detail, no change of the garment",
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
