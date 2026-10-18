"""fal.ai provider: thin wrapper over ugc.fal_gen."""
from __future__ import annotations

import os
from pathlib import Path

from .. import config, costs, fal_gen


class FalProvider:
    name = "fal"

    def missing_env(self) -> list[str]:
        return [] if os.environ.get("FAL_KEY") else ["FAL_KEY"]

    def describe(self) -> dict:
        return {"provider": "fal", "image": config.IMAGE_ENDPOINT, "video": config.VIDEO_ENDPOINT,
                "resolution": config.VIDEO_RESOLUTION}

    def estimate_images(self, n: int) -> float:
        return costs.estimate_images(n)

    def estimate_video(self, seconds: float) -> float:
        return costs.estimate_video(seconds)

    def generate_avatar_images(self, prompt, garments, avatar_ref, n, out_dir, prefix, on_image=None) -> list[Path]:
        av = [avatar_ref] if isinstance(avatar_ref, Path) else list(avatar_ref or [])
        refs = list(garments) + av
        return fal_gen.generate_avatar_images(prompt, refs, n, out_dir, prefix=prefix, on_image=on_image)

    def generate_clip(self, prompt, negative_prompt, image, duration_s, out_path) -> Path:
        return fal_gen.generate_clip(prompt, negative_prompt, image, duration_s, out_path)
