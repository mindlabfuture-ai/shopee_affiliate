"""fal.ai generation: avatar stills (image edit) and video clips (image-to-video), plus joining."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from . import config, costs


class FalError(RuntimeError):
    pass


def _fal():
    if not os.environ.get("FAL_KEY"):
        raise FalError("FAL_KEY is not set")
    import fal_client
    return fal_client


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def generate_avatar_images(prompt: str, reference_images: list[Path], n: int, out_dir: Path,
                           prefix: str = "avatar") -> list[Path]:
    """reference_images: garment listing photos first, then (optionally) the saved avatar reference.
    Files are named <prefix>_NN.png; the app uses prefix avatar_s<storyboard index>."""
    fal = _fal()
    costs.guard(costs.estimate_images(n))
    urls = [fal.upload_file(str(p)) for p in reference_images]
    out: list[Path] = []
    for i in range(n):
        res = fal.subscribe(config.IMAGE_ENDPOINT, arguments={
            "prompt": prompt, "image_urls": urls, "num_images": 1, "aspect_ratio": "9:16", "output_format": "png"})
        costs.record("image", costs.estimate_images(1), config.IMAGE_ENDPOINT)
        stamp = len(list(out_dir.glob(f"{prefix}_*.png"))) + 1 if out_dir.exists() else 1
        out.append(_download(res["images"][0]["url"], out_dir / f"{prefix}_{stamp:02d}.png"))
    return out


def generate_clip(prompt: str, negative_prompt: str, image: Path, duration_s: int, out_path: Path,
                  seed: int | None = None) -> Path:
    fal = _fal()
    costs.guard(costs.estimate_video(duration_s))
    args = {"prompt": prompt, "negative_prompt": negative_prompt, "image_url": fal.upload_file(str(image)),
            "aspect_ratio": "9:16", "resolution": config.VIDEO_RESOLUTION, "duration": f"{duration_s}s",
            "generate_audio": False}
    if seed is not None:
        args["seed"] = seed
    res = fal.subscribe(config.VIDEO_ENDPOINT, arguments=args)
    costs.record("video", costs.estimate_video(duration_s), config.VIDEO_ENDPOINT)
    return _download(res["video"]["url"], out_path)


def join_clips(clips: list[Path], out_path: Path) -> Path:
    """Concatenate and strip audio, re-encoding so mismatched clips join cleanly."""
    if not shutil.which("ffmpeg"):
        raise FalError("ffmpeg is not installed")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "list.txt"
        lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out_path)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            raise FalError(f"ffmpeg join failed: {p.stderr[-400:]}")
    return out_path
