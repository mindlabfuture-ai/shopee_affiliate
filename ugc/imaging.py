"""Image/video helpers: loading, frame sampling, CIELAB color math."""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
Frame = tuple[float, np.ndarray]  # (time in seconds, BGR image)


def list_images(directory: Path) -> list[Path]:
    d = Path(directory)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []


def read_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def trim_borders(img: np.ndarray, thresh: float = 12.0) -> np.ndarray:
    """Crop black bars (phone screenshots)."""
    g = img.mean(axis=2)
    rows = np.where(g.mean(axis=1) > thresh)[0]
    cols = np.where(g.mean(axis=0) > thresh)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def to_jpeg_b64(img: np.ndarray, max_side: int = 1024, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", resize_max(img, max_side), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.standard_b64encode(buf.tobytes()).decode()


def load_frames(path: Path, max_frames: int = 48, height: int = 960) -> list[Frame]:
    """Evenly spaced frames, decoded sequentially (robust for HEVC)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, -(-total // max_frames)) if total else 1
    frames: list[Frame] = []
    i = 0
    while cap.grab():
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                if f.shape[0] > height:
                    f = cv2.resize(f, (round(f.shape[1] * height / f.shape[0]), height), interpolation=cv2.INTER_AREA)
                frames.append((i / fps, f))
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return frames


def bgr_to_lab(img: np.ndarray) -> np.ndarray:
    """True CIELAB (L 0-100, a/b about -128..127) as float32."""
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def hex_to_lab(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return bgr_to_lab(np.array([[[b, g, r]]], dtype=np.uint8))[0, 0]


def lab_to_hex(lab: np.ndarray) -> str:
    bgr = cv2.cvtColor(np.array([[lab]], dtype=np.float32), cv2.COLOR_Lab2BGR)[0, 0]
    b, g, r = (int(round(float(np.clip(c, 0, 1)) * 255)) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def derive_garment_lab(images: list[np.ndarray]) -> np.ndarray:
    """Fallback: dominant non-background, non-hair color in the listing photos.

    Heuristic. Prefer setting color_hex in the product spec; the QA report prints what was derived.
    """
    px = []
    for img in images:
        small = resize_max(trim_borders(img), 320)
        h, w = small.shape[:2]
        lab = bgr_to_lab(small[int(h * 0.2) : int(h * 0.95), int(w * 0.1) : int(w * 0.9)]).reshape(-1, 3)
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        keep = ~(((lab[:, 0] > 78) & (chroma < 22)) | (lab[:, 0] < 12))  # light background, hair/black
        px.append(lab[keep])
    data = np.concatenate(px) if px else np.empty((0, 3), np.float32)
    if len(data) < 500:
        raise ValueError("not enough garment pixels in the listing photos to derive a color")
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(data.astype(np.float32), 5, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centers[np.bincount(labels.ravel(), minlength=5).argmax()]
