"""Central settings. Everything price- or model-related can be overridden by env var."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("UGC_DATA_DIR", ROOT / "data"))

CLAUDE_MODEL = os.environ.get("UGC_CLAUDE_MODEL", "claude-sonnet-5-5")

# Lean stack. fal says the Veo 3.1 Lite endpoint is deprecated on 2026-11-17; swap via env, no code change.
IMAGE_ENDPOINT = os.environ.get("UGC_IMAGE_ENDPOINT", "fal-ai/nano-banana/edit")
VIDEO_ENDPOINT = os.environ.get("UGC_VIDEO_ENDPOINT", "fal-ai/veo3.1/lite/image-to-video")
VIDEO_RESOLUTION = os.environ.get("UGC_VIDEO_RESOLUTION", "720p")  # 1080p requires 8s clips on Veo Lite
CLIP_SECONDS = (4, 6, 8)  # durations Veo 3.1 accepts

# Estimated USD rates. From third-party price pages, NOT verified against fal's live pricing.
IMAGE_USD_EACH = float(os.environ.get("UGC_IMAGE_USD", "0.04"))
VIDEO_USD_PER_SEC = float(os.environ.get("UGC_VIDEO_USD_PER_SEC", "0.03"))  # audio off
DAILY_CAP_USD = float(os.environ.get("UGC_DAILY_CAP_USD", "15"))

# QA thresholds: starting points, calibrate on videos you have already accepted.
COLOR_WARN_DE, COLOR_FAIL_DE = 8.0, 15.0  # median CIELAB distance to the listing color
DRIFT_WARN_DE, DRIFT_FAIL_DE = 6.0, 12.0  # spread of garment color across frames
JUMP_WARN_DE, JUMP_FAIL_DE = 6.0, 12.0  # garment color change across a scene cut
NOT_FOUND_WARN, NOT_FOUND_FAIL = 0.25, 0.5  # share of frames where garment color is not found
MIN_DURATION_S, MAX_DURATION_S = 5.0, 30.0
