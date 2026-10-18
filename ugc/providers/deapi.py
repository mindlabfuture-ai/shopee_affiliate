"""deAPI provider (API v2, https://docs.deapi.ai).

Flow for every job: POST multipart -> data.request_id -> poll GET /api/v2/jobs/{id} until status is
done/error -> download data.result_url. Model limits (size, frames, fps, steps) are read from
GET /api/v2/models, not hard-coded, because they differ per model.

Prices are quoted with the /price endpoints (cached) and read in the account's credit unit, which this
module treats as USD. Basic accounts are limited to 200 status/price calls per day, hence the cache and
the long default poll interval.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from .. import config, costs
from ..imaging import resize_max

BASE = "https://api.deapi.ai"
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
PREFER_IMAGE = ("nano", "qwen")
PREFER_VIDEO = ("ltx",)
MAX_WAIT_S = 20 * 60
SUBMIT_RETRIES = 4  # Basic accounts hit the per-minute limit easily
REF_MAX_SIDE = 1280  # smaller uploads: big PNGs made the origin time out (504)
QUOTE_RETRY_AFTER_S = 300


class DeapiError(RuntimeError):
    pass


def _snap(value: int, step: int, lo: int, hi: int) -> int:
    v = min(max(int(value), lo), hi)
    v -= v % step
    return v + step if v < lo else v


class DeapiProvider:
    name = "deapi"

    def __init__(self, api_key: str | None = None, transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep, poll_s: float | None = None):
        self.key = api_key or os.environ.get("DEAPI_API_KEY")
        self.client = httpx.Client(base_url=BASE, timeout=120, transport=transport)  # no default auth header
        self.sleep = sleep
        self.poll_s = config.DEAPI_POLL_S if poll_s is None else poll_s
        self._models: dict[str, list[dict]] = {}
        self._quotes: dict[tuple, float] = {}
        self._quote_failed_until: dict[tuple, float] = {}
        self._chosen: dict[str, dict] = {}

    # ---------- plumbing ----------
    def missing_env(self) -> list[str]:
        return [] if self.key else ["DEAPI_API_KEY"]

    def _request(self, method: str, path_or_url: str, **kw) -> httpx.Response:
        if not self.key:
            raise DeapiError("DEAPI_API_KEY is not set")
        headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json"}
        r = self.client.request(method, path_or_url, headers=headers, **kw)
        if r.status_code >= 400:
            raise DeapiError(self._explain(r))
        return r

    @staticmethod
    def _explain(r: httpx.Response) -> str:
        try:
            body = r.json()
        except ValueError:
            body = {}
        detail = body.get("message") or r.text[:200]
        if body.get("errors"):
            detail += f" {body['errors']}"
        hint = {401: "check DEAPI_API_KEY", 403: "account suspended, or this model needs a Premium account (any top-up unlocks it); pin a free one with UGC_DEAPI_VIDEO_MODEL / UGC_DEAPI_IMAGE_MODEL",
                422: "the request does not fit this model's limits (see `python -m ugc deapi-models`)",
                429: f"rate limited ({r.headers.get('X-RateLimit-Type', '?')}); retry after "
                     f"{r.headers.get('Retry-After', '?')}s"}.get(r.status_code, "")
        return f"deAPI {r.request.method} {r.request.url.path} -> {r.status_code}: {detail}" + (f" ({hint})" if hint else "")

    # ---------- models ----------
    def models(self, inference_type: str) -> list[dict]:
        if inference_type not in self._models:
            r = self._request("GET", "/api/v2/models",
                              params={"filter[inference_types]": inference_type, "limit": 100})
            self._models[inference_type] = r.json().get("data", [])
        return self._models[inference_type]

    def _choose(self, kind: str, inference_type: str, pinned: str, prefer: tuple[str, ...]) -> dict:
        if kind in self._chosen:
            return self._chosen[kind]
        ms = self.models(inference_type)
        if not ms:
            raise DeapiError(f"no deAPI models for {inference_type}")
        if pinned:
            match = next((m for m in ms if m["slug"] == pinned), None)
            if not match:
                raise DeapiError(f"model {pinned!r} not found for {inference_type}; available: "
                                 + ", ".join(m["slug"] for m in ms))
            self._chosen[kind] = match
            return match
        usable = [m for m in ms if m.get("status") != "test_model" and "experimental" not in (m.get("tags") or [])] or ms
        if kind == "image":  # need several reference images (garment photo + avatar) when possible
            multi = [m for m in usable if int(((m.get("info") or {}).get("limits") or {}).get("max_input_images", 1)) >= 2]
            usable = multi or usable
        pick = next((m for kw in prefer for m in usable if kw in m["slug"].lower()), usable[0])
        self._chosen[kind] = pick
        return pick

    def image_model(self) -> dict:
        return self._choose("image", "img2img", config.DEAPI_IMAGE_MODEL, PREFER_IMAGE)

    def video_model(self) -> dict:
        return self._choose("video", "img2video", config.DEAPI_VIDEO_MODEL, PREFER_VIDEO)

    def describe(self) -> dict:  # no network: this runs on every page load
        img = self._chosen.get("image", {}).get("slug") or config.DEAPI_IMAGE_MODEL or "auto"
        vid = self._chosen.get("video", {}).get("slug") or config.DEAPI_VIDEO_MODEL or "auto"
        return {"provider": "deapi", "image": img, "video": vid, "resolution": config.DEAPI_SIZE}

    # ---------- parameters from live model limits ----------
    @staticmethod
    def _size(limits: dict) -> tuple[int, int]:
        """Target WxH scaled down (keeping the aspect ratio) to fit the model's max, then snapped to its step."""
        w, h = (int(x) for x in config.DEAPI_SIZE.lower().split("x"))
        step = int(limits.get("resolution_step") or limits.get("dimension_multiple") or 32)
        max_w, max_h = int(limits.get("max_width", 4096)), int(limits.get("max_height", 4096))
        scale = min(1.0, max_w / w, max_h / h)
        w, h = round(w * scale), round(h * scale)
        return (_snap(w, step, int(limits.get("min_width", 64)), max_w),
                _snap(h, step, int(limits.get("min_height", 64)), max_h))

    def _video_params(self, seconds: int) -> dict:
        m = self.video_model()
        info = m.get("info") or {}
        lim, dflt, feat = info.get("limits") or {}, info.get("defaults") or {}, info.get("features") or {}
        fps = int(dflt.get("fps") or 24)
        frames = min(max(int(seconds * fps), int(lim.get("min_frames", 1))), int(lim.get("max_frames", 10**6)))
        w, h = self._size(lim)
        p = {"model": m["slug"], "width": w, "height": h, "frames": frames, "fps": fps}
        if feat.get("supports_steps"):
            p["steps"] = int(dflt.get("steps") or lim.get("max_steps") or 10)
        if feat.get("supports_guidance"):
            p["guidance"] = float(dflt["guidance"] if dflt.get("guidance") is not None else lim.get("min_guidance", 3.0))
        p["_negative_ok"] = feat.get("supports_negative_prompt") is not False
        return p

    def _image_params(self) -> dict:
        m = self.image_model()
        info = m.get("info") or {}
        lim, dflt, feat = info.get("limits") or {}, info.get("defaults") or {}, info.get("features") or {}
        w, h = self._size(lim)
        p = {"model": m["slug"], "width": w, "height": h, "max_images": int(lim.get("max_input_images", 1))}
        if feat.get("supports_steps"):
            p["steps"] = int(dflt.get("steps") or lim.get("max_steps") or 10)
        if feat.get("supports_guidance"):
            p["guidance"] = float(dflt["guidance"] if dflt.get("guidance") is not None else lim.get("min_guidance", 3.0))
        return p

    # ---------- prices (quotes) ----------
    def _quote(self, key: tuple, path: str, body: dict) -> float | None:
        if key in self._quotes:
            return self._quotes[key]
        if time.time() < self._quote_failed_until.get(key, 0):
            return None
        try:
            price = float(self._request("POST", path, json=body).json()["data"]["price"])
        except (DeapiError, KeyError, ValueError, TypeError):
            self._quote_failed_until[key] = time.time() + QUOTE_RETRY_AFTER_S
            return None  # caller falls back to the configured estimate
        self._quotes[key] = price
        return price

    def estimate_images(self, n: int) -> float:
        try:
            p = self._image_params()
            q = self._quote(("img", p["model"], p["width"], p["height"]), "/api/v2/images/edits/price",
                            {k: v for k, v in {"prompt": "quote", "seed": 1, **p}.items() if k != "max_images"})
        except DeapiError:
            q = None
        return round(n * (q if q is not None else config.IMAGE_USD_EACH), 4)

    def estimate_video(self, seconds: float) -> float:
        try:
            p = self._video_params(8)
            body = {k: v for k, v in p.items() if k in ("model", "width", "height", "frames", "fps", "steps")}
            q = self._quote(("vid", p["model"], p["width"], p["height"], p["frames"]), "/api/v2/videos/animations/price", body)
            per_sec = q / (p["frames"] / p["fps"]) if q is not None else None
        except DeapiError:
            per_sec = None
        return round(seconds * (per_sec if per_sec is not None else config.VIDEO_USD_PER_SEC), 4)

    # ---------- jobs ----------
    def _submit(self, path: str, **kw) -> str:
        """POST a job. A per-minute 429 is retried after Retry-After; a daily 429 or any other error is not
        (a timeout may already have created a billable job, so it is never retried blindly)."""
        for attempt in range(SUBMIT_RETRIES + 1):
            if not self.key:
                raise DeapiError("DEAPI_API_KEY is not set")
            r = self.client.request("POST", path, headers={"Authorization": f"Bearer {self.key}", "Accept": "application/json"}, **kw)
            if r.status_code == 429 and r.headers.get("X-RateLimit-Type", "minute") != "daily" and attempt < SUBMIT_RETRIES:
                wait = min(max(float(r.headers.get("Retry-After", 20) or 20), 1.0), 90.0) + 1
                print(f"deapi: rate limited on {path}, waiting {wait:.0f}s (attempt {attempt + 1}/{SUBMIT_RETRIES})")
                self.sleep(wait)
                continue
            if r.status_code >= 400:
                raise DeapiError(self._explain(r))
            return r.json()["data"]["request_id"]
        raise DeapiError(f"deAPI kept rate limiting {path}")

    def _wait(self, request_id: str) -> dict:
        start = time.time()
        while True:
            try:
                data = self._request("GET", f"/api/v2/jobs/{request_id}").json()["data"]
            except DeapiError as e:
                if "429" in str(e) and "daily" not in str(e) and time.time() - start < MAX_WAIT_S:
                    self.sleep(max(self.poll_s, 10))  # per-minute limit: wait and retry
                    continue
                raise
            status = data.get("status")
            if status == "done":
                return data
            if status == "error":
                raise DeapiError(f"deAPI job failed: {data.get('error_code')} / {data.get('error_reason')} "
                                 f"(retryable={data.get('retryable')}, refunded={data.get('refunded')})")
            if time.time() - start > MAX_WAIT_S:
                raise DeapiError(f"deAPI job {request_id} still {status} after {MAX_WAIT_S // 60} min")
            self.sleep(self.poll_s)

    def _download(self, url: str) -> bytes:
        r = self.client.get(url, follow_redirects=True, timeout=300)  # absolute URL, no auth header sent
        r.raise_for_status()
        return r.content

    @staticmethod
    def _file_part(path: Path) -> tuple[str, bytes, str]:
        """Images are re-encoded as JPEG (max 1280 px): smaller uploads, and no 504s from the origin."""
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            ok, buf = cv2.imencode(".jpg", resize_max(img, REF_MAX_SIDE), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                return (path.stem + ".jpg", buf.tobytes(), "image/jpeg")
        return (path.name, path.read_bytes(), MIME.get(path.suffix.lower(), "application/octet-stream"))

    # ---------- generation ----------
    def generate_avatar_images(self, prompt, garments, avatar_ref, n, out_dir, prefix, on_image=None) -> list[Path]:
        costs.guard(self.estimate_images(n))
        p = self._image_params()
        limit = p.pop("max_images")
        refs = list(garments)[: max(1, limit - (1 if avatar_ref else 0))] + ([avatar_ref] if avatar_ref and limit > 1 else [])
        out_dir.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        for _ in range(n):
            fields = {"prompt": prompt, "seed": random.randint(1, 2**31 - 1), **{k: v for k, v in p.items()}}
            files = ([("images[]", self._file_part(r)) for r in refs] if len(refs) > 1
                     else [("image", self._file_part(refs[0]))])
            rid = self._submit("/api/v2/images/edits", data=fields, files=files)
            done = self._wait(rid)
            img = cv2.imdecode(np.frombuffer(self._download(done["result_url"]), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise DeapiError("deAPI returned an image that cannot be decoded")
            idx = len(list(out_dir.glob(f"{prefix}_*.png"))) + 1
            path = out_dir / f"{prefix}_{idx:02d}.png"
            cv2.imwrite(str(path), img)
            out.append(path)
            if on_image:
                on_image(path)
            costs.record("image", float((done.get("price") or {}).get("amount") or self.estimate_images(1)),
                         f"deapi {p['model']}")
        return out

    def generate_clip(self, prompt, negative_prompt, image, duration_s, out_path) -> Path:
        costs.guard(self.estimate_video(duration_s))
        p = self._video_params(duration_s)
        negative_ok = p.pop("_negative_ok")
        fields = {"prompt": prompt, "seed": random.randint(1, 2**31 - 1), **p}
        if negative_ok and negative_prompt:
            fields["negative_prompt"] = negative_prompt
        rid = self._submit("/api/v2/videos/animations", data=fields,
                           files=[("first_frame_image", self._file_part(image))])
        done = self._wait(rid)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self._download(done["result_url"]))
        actual = p["frames"] / p["fps"]
        if abs(actual - duration_s) > 0.5:  # model cap: clip is shorter than planned
            print(f"deapi: {p['model']} capped the clip at {actual:.1f}s (planned {duration_s}s)")
        costs.record("video", float((done.get("price") or {}).get("amount") or self.estimate_video(actual)),
                     f"deapi {p['model']}")
        return out_path
