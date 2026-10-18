import cv2
import numpy as np
import pytest

from tests.conftest import make_frame, needs_ffmpeg, write_video
from ugc import media


@needs_ffmpeg
def test_last_frame_is_the_end_of_the_clip_not_the_start(tmp_path):
    red, blue = (47, 45, 179), (179, 45, 47)
    frames = [make_frame(red) for _ in range(24)] + [make_frame(blue) for _ in range(24)]
    clip = write_video(tmp_path / "c.mp4", frames)
    out = media.last_frame(clip, tmp_path / "out" / "last.png")
    img = cv2.imread(str(out))
    centre = img[int(img.shape[0] * 0.7), img.shape[1] // 2]  # inside the garment rectangle
    assert centre[0] > centre[2]  # blue channel (BGR index 0) beats red: it is the blue half
    assert img.shape[:2] == (640, 360)


@needs_ffmpeg
def test_last_frame_replaces_an_old_file(tmp_path):
    clip = write_video(tmp_path / "c.mp4", [make_frame((10, 10, 10)) for _ in range(12)])
    out = tmp_path / "last.png"
    out.write_bytes(b"stale")
    assert cv2.imread(str(media.last_frame(clip, out))) is not None


def test_last_frame_of_a_broken_file_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(media.MediaError):
        media.last_frame(bad, tmp_path / "x.png")
