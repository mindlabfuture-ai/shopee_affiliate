"""Small ffmpeg helpers."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def last_frame(video: Path, out_png: Path) -> Path:
    """Save the final frame of a clip as a PNG. The next clip starts from it, so the join is seamless."""
    if not shutil.which("ffmpeg"):
        raise MediaError("ffmpeg is not installed")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.unlink(missing_ok=True)
    p = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-sseof", "-0.3", "-i", str(video),
                        "-update", "1", "-frames:v", "9999", "-q:v", "1", str(out_png)], capture_output=True, text=True)
    if p.returncode or not out_png.exists():
        raise MediaError(f"could not read the last frame of {video.name}: {p.stderr[-300:]}")
    return out_png
