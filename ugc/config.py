"""Central settings. Everything price- or model-related can be overridden by env var."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("UGC_DATA_DIR", ROOT / "data"))

CLAUDE_MODEL = os.environ.get("UGC_CLAUDE_MODEL", "claude-sonnet-5-5")

PROVIDER = os.environ.get("UGC_PROVIDER", "fal")  # "fal" or "deapi"

# deAPI (https://docs.deapi.ai). Model slugs are discovered from /api/v2/models unless pinned here;
# run `python -m ugc deapi-models` to list them with their limits.
DEAPI_IMAGE_MODEL = os.environ.get("UGC_DEAPI_IMAGE_MODEL", "")
DEAPI_VIDEO_MODEL = os.environ.get("UGC_DEAPI_VIDEO_MODEL", "")
DEAPI_SIZE = os.environ.get("UGC_DEAPI_SIZE", "576x1024")  # WxH, 9:16; clamped to each model's limits
DEAPI_POLL_S = float(os.environ.get("UGC_DEAPI_POLL_S", "8"))  # Basic accounts allow only 200 status calls/day

# Lean stack. fal says the Veo 3.1 Lite endpoint is deprecated on 2026-11-17; swap via env, no code change.
IMAGE_ENDPOINT = os.environ.get("UGC_IMAGE_ENDPOINT", "fal-ai/nano-banana/edit")
VIDEO_ENDPOINT = os.environ.get("UGC_VIDEO_ENDPOINT", "fal-ai/veo3.1/lite/image-to-video")
VIDEO_RESOLUTION = os.environ.get("UGC_VIDEO_RESOLUTION", "720p")  # 1080p requires 8s clips on Veo Lite
# Clip lengths the planner may use. Veo 3.1 accepts 4, 6 or 8 s; longer-clip models (for example Seedance, up to ~30 s)
# can be allowed with e.g. UGC_CLIP_SECONDS=5,8,10,12. MAX_CLIP_S still caps them.
CLIP_SECONDS = tuple(sorted({int(x) for x in os.environ.get("UGC_CLIP_SECONDS", "4,6,8").split(",") if x.strip()}))
MAX_CLIP_S = int(os.environ.get("UGC_MAX_CLIP_S", "8"))  # set 4 for models capped at ~4 s (e.g. LTX-Video 13B on deAPI)


def allowed_clip_seconds(limit: float | None = None) -> tuple[int, ...]:
    """Clip lengths the planner may use. `limit` is the chosen model's real maximum in seconds (read live from the
    provider); the whole-second length that just fits is always allowed, so short-clip models still get a valid plan."""
    cap = MAX_CLIP_S if limit is None else min(MAX_CLIP_S, int(limit))
    extra = {cap} if limit is not None and cap >= 1 else set()
    return tuple(sorted({s for s in CLIP_SECONDS if s <= cap} | extra)) or (CLIP_SECONDS[0],)

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
