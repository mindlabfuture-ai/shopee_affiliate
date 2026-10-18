"""Container-level checks via ffprobe/ffmpeg: format, audio, black frames, scene cuts."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .. import config
from .report import Check


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return p.stdout if cmd[0] == "ffprobe" else p.stderr


def probe(path: Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    return json.loads(out)


def volume(path: Path) -> tuple[float | None, float | None]:
    err = _run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"])
    mean = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", err)
    mx = re.search(r"max_volume: (-?[\d.]+|-inf) dB", err)
    f = lambda m: None if not m or m.group(1) == "-inf" else float(m.group(1))  # noqa: E731
    return f(mean), f(mx)


def black_segments(path: Path) -> list[tuple[float, float]]:
    err = _run(["ffmpeg", "-hide_banner", "-i", str(path), "-an", "-vf", "blackdetect=d=0.1:pix_th=0.10", "-f", "null", "-"])
    return [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)]


def scene_cuts(path: Path, threshold: float = 0.35) -> list[float]:
    err = _run(["ffmpeg", "-hide_banner", "-i", str(path), "-an", "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"])
    return [float(t) for t in re.findall(r"pts_time:([\d.]+)", err)]


def check_technical(path: Path) -> tuple[list[Check], dict]:
    """Returns (checks, info) where info holds duration and cut times for later steps."""
    if not (shutil.which("ffprobe") and shutil.which("ffmpeg")):
        return [Check("technical", "skip", "ffmpeg/ffprobe not installed")], {"duration": None, "cuts": []}

    meta = probe(path)
    v = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        return [Check("format", "fail", "no video stream")], {"duration": None, "cuts": []}
    w, h = int(v["width"]), int(v["height"])
    dur = float(meta["format"].get("duration") or v.get("duration") or 0)
    checks: list[Check] = []

    ratio_ok = abs(w / h - 9 / 16) < 0.02
    big_enough = h >= 1280
    checks.append(Check("format", "pass" if ratio_ok and big_enough else "warn",
                        f"{w}x{h}" + ("" if ratio_ok else ", not 9:16") + ("" if big_enough else ", under 1280 px tall"),
                        {"width": w, "height": h}))
    in_range = config.MIN_DURATION_S <= dur <= config.MAX_DURATION_S
    checks.append(Check("duration", "pass" if in_range else "warn", f"{dur:.1f}s", {"seconds": dur}))

    if a is None:
        checks.append(Check("audio", "pass", "no audio track"))
    else:
        mean, mx = volume(path)
        silent = mean is None or mean < -60
        if silent:
            checks.append(Check("audio", "pass", "audio track is silent"))
        else:
            checks.append(Check("audio", "warn",
                                f"audio present (mean {mean:.1f} dB). Strip it before upload: "
                                "ffmpeg -i in.mp4 -c:v copy -an out.mp4",
                                {"mean_db": mean, "max_db": mx}))

    blacks = black_segments(path)
    edge = [(s, e) for s, e in blacks if s < 0.3 or e > dur - 0.3]
    checks.append(Check("black_frames", "warn" if edge else "pass",
                        f"black at start/end: {edge}; trim it" if edge else "none at start/end", {"segments": blacks}))

    cuts = scene_cuts(path)
    checks.append(Check("scene_cuts", "info", f"{len(cuts)} cut(s) at {', '.join(f'{c:.1f}s' for c in cuts) or 'none'}",
                        {"cuts": cuts}))
    return checks, {"duration": dur, "cuts": cuts}
