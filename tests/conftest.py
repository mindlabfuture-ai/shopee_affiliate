import importlib
import shutil
import subprocess

import cv2
import numpy as np
import pytest

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UGC_DATA_DIR", str(tmp_path / "data"))
    from ugc import config
    importlib.reload(config)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data"


def make_frame(garment_bgr, bg_bgr=(200, 210, 215), size=(360, 640)):
    w, h = size
    img = np.full((h, w, 3), bg_bgr, np.uint8)
    img[int(h * 0.35) : int(h * 0.95), int(w * 0.3) : int(w * 0.7)] = garment_bgr
    return img


def write_video(path, frames, fps=12, tail_black=0, audio=False):
    h, w = frames[0].shape[:2]
    raw = path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    for _ in range(tail_black):
        vw.write(np.zeros_like(frames[0]))
    vw.release()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw)]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=10", "-shortest", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def write_image(path, img):
    cv2.imwrite(str(path), img)
    return path
