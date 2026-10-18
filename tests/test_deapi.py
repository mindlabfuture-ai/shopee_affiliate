import json
import re

import cv2
import httpx
import numpy as np
import pytest

from tests.conftest import make_frame, write_image
from ugc import config, costs, providers
from ugc.providers.deapi import DeapiError, DeapiProvider

MODELS = {
    "img2img": [
        {"slug": "QwenImageEdit_Plus_NF4", "status": "standard_model", "tags": [],
         "info": {"limits": {"min_width": 256, "max_width": 1536, "min_height": 256, "max_height": 1536,
                             "resolution_step": 64, "max_input_images": 1}, "defaults": {}, "features": {}}},
        {"slug": "Nano_Banana_2", "status": "standard_model", "tags": [],
         "info": {"limits": {"min_width": 256, "max_width": 2048, "min_height": 256, "max_height": 2048,
                             "resolution_step": 64, "max_input_images": 3}, "defaults": {}, "features": {}}},
    ],
    "img2video": [
        {"slug": "Seedance_test", "status": "test_model", "tags": ["test"], "info": {}},
        {"slug": "Ltx_2_3_22B", "status": "standard_model", "tags": [],
         "info": {"limits": {"min_width": 256, "max_width": 1280, "min_height": 256, "max_height": 1280,
                             "min_frames": 25, "max_frames": 121, "resolution_step": 32, "max_steps": 20},
                  "defaults": {"fps": 24, "steps": 8},
                  "features": {"supports_steps": True, "supports_negative_prompt": True}}},
    ],
}
PNG = cv2.imencode(".png", make_frame((47, 45, 179)))[1].tobytes()


def fields(req):
    body = req.content.decode("latin-1")
    return {m.group(1): m.group(2) for m in re.finditer(r'name="([^"\[\]]+)"\r\n\r\n([^\r]*)\r\n', body)}


def parts(req):
    return re.findall(r'name="([^"]+)"', req.content.decode("latin-1"))


class Api:
    """Fake of the documented deAPI v2 surface."""

    def __init__(self, job_script=None, price=0.20, post_failures=None):
        self.requests, self.job_script, self.price = [], list(job_script or []), price
        self.posts = []
        self.post_failures = list(post_failures or [])  # [(status, headers)] returned before a POST succeeds
        self.post_attempts = 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if req.url.host != "api.deapi.ai":  # result download
            return httpx.Response(200, content=self.download)
        path = req.url.path
        if path == "/api/v2/models":
            return httpx.Response(200, json={"data": MODELS[req.url.params["filter[inference_types]"]]})
        if path.endswith("/price"):
            if self.price is None:
                return httpx.Response(422, json={"message": "nope", "errors": {}})
            return httpx.Response(200, json={"data": {"price": self.price, "is_estimated": False}})
        if req.method == "POST":
            self.post_attempts += 1
            if self.post_failures:
                status, headers = self.post_failures.pop(0)
                return httpx.Response(status, headers=headers, json={"message": "Too Many Attempts."})
            self.posts.append(req)
            return httpx.Response(200, json={"data": {"request_id": "rid-1"}})
        if path.startswith("/api/v2/jobs/"):
            step = self.job_script.pop(0) if self.job_script else {"status": "done", "result_url": "https://cdn.example/out",
                                                                   "price": {"amount": 0.07}}
            if step == "429d":
                return httpx.Response(429, headers={"X-RateLimit-Type": "daily", "Retry-After": "999"}, json={"message": "limit"})
            return httpx.Response(200, json={"data": step})
        return httpx.Response(404)

    download = PNG


def provider(api, key="k"):
    return DeapiProvider(api_key=key, transport=httpx.MockTransport(api), sleep=lambda s: None, poll_s=0)


def refs(tmp_path, n):
    return [write_image(tmp_path / f"g{i}.png", make_frame((47, 45, 179))) for i in range(n)]


def test_model_auto_selection_prefers_multi_image_and_ltx():
    p = provider(Api())
    assert p.image_model()["slug"] == "Nano_Banana_2"  # accepts several reference images
    assert p.video_model()["slug"] == "Ltx_2_3_22B"  # skips the test model


def test_pinned_unknown_model_lists_alternatives(monkeypatch):
    monkeypatch.setattr(config, "DEAPI_VIDEO_MODEL", "Nope")
    with pytest.raises(DeapiError, match="Ltx_2_3_22B"):
        provider(Api()).video_model()


def test_clip_respects_model_limits_and_returns_priced_video(tmp_path):
    api = Api(job_script=[{"status": "pending"}, {"status": "processing", "progress": 40},
                          {"status": "done", "result_url": "https://cdn.example/v.mp4", "price": {"amount": 0.07}}])
    api.download = b"fake-mp4"
    img = refs(tmp_path, 1)[0]
    out = provider(api).generate_clip("walk", "no morphing", img, 8, tmp_path / "clip.mp4")
    assert out.read_bytes() == b"fake-mp4"
    f = fields(api.posts[0])
    assert f["frames"] == "121"  # 8 s x 24 fps = 192, capped at the model's max_frames
    assert (f["width"], f["height"]) == ("576", "1024")  # 9:16, multiples of the 32 step
    assert f["steps"] == "8" and f["negative_prompt"] == "no morphing" and f["model"] == "Ltx_2_3_22B"
    assert "first_frame_image" in parts(api.posts[0])
    assert api.posts[0].url.path == "/api/v2/videos/animations"
    assert costs.today_total() == pytest.approx(0.07)  # actual price from the job, not the estimate


def test_auth_header_goes_to_api_but_never_to_download_host(tmp_path):
    api = Api()
    api.download = b"x"
    provider(api, key="secret").generate_clip("p", "", refs(tmp_path, 1)[0], 4, tmp_path / "c.mp4")
    for r in api.requests:
        if r.url.host == "api.deapi.ai":
            assert r.headers["authorization"] == "Bearer secret" and r.headers["accept"] == "application/json"
        else:
            assert "authorization" not in r.headers


def test_avatar_sends_multiple_references_and_writes_png(tmp_path):
    api = Api()
    ref = write_image(tmp_path / "ref.png", make_frame((10, 10, 10)))
    out = provider(api).generate_avatar_images("prompt", refs(tmp_path, 2), ref, 1, tmp_path / "av", "avatar_s0")
    assert parts(api.posts[0]).count("images[]") == 3 and api.posts[0].url.path == "/api/v2/images/edits"
    assert out[0].name == "avatar_s0_01.png" and cv2.imread(str(out[0])) is not None


def test_avatar_keeps_the_avatar_reference_when_the_model_caps_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEAPI_IMAGE_MODEL", "Nano_Banana_2")  # limit 3
    api = Api()
    ref = write_image(tmp_path / "ref.png", make_frame((10, 10, 10)))
    provider(api).generate_avatar_images("p", refs(tmp_path, 3), ref, 1, tmp_path / "av", "avatar_s0")
    assert parts(api.posts[0]).count("images[]") == 3  # 2 garment photos + the avatar, not 3 garments


def test_single_image_model_uses_image_field(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEAPI_IMAGE_MODEL", "QwenImageEdit_Plus_NF4")
    api = Api()
    provider(api).generate_avatar_images("p", refs(tmp_path, 3), tmp_path / "ref.png", 1, tmp_path / "av", "avatar_s0")
    assert "image" in parts(api.posts[0]) and "images[]" not in parts(api.posts[0])


def test_job_error_surfaces_reason_and_refund(tmp_path):
    api = Api(job_script=[{"status": "error", "error_code": "WORKER_ERROR", "error_reason": "OOM",
                           "retryable": True, "refunded": True}])
    with pytest.raises(DeapiError, match=r"WORKER_ERROR / OOM \(retryable=True, refunded=True\)"):
        provider(api).generate_clip("p", "", refs(tmp_path, 1)[0], 4, tmp_path / "c.mp4")


def test_daily_rate_limit_is_not_retried(tmp_path):
    with pytest.raises(DeapiError, match="rate limited \\(daily\\)"):
        provider(Api(job_script=["429d"])).generate_clip("p", "", refs(tmp_path, 1)[0], 4, tmp_path / "c.mp4")


def test_quote_is_per_second_and_cached():
    api = Api(price=0.30)  # for the 5.04 s (121 frames) quote
    p = provider(api)
    first = p.estimate_video(10)
    second = p.estimate_video(10)
    assert first == second == pytest.approx(10 * 0.30 / (121 / 24), rel=1e-3)
    assert sum(r.url.path.endswith("animations/price") for r in api.requests) == 1  # saves the 200/day quota


def test_estimate_falls_back_to_config_when_quote_fails():
    p = provider(Api(price=None))
    assert p.estimate_video(10) == pytest.approx(10 * config.VIDEO_USD_PER_SEC)
    assert p.estimate_images(2) == pytest.approx(2 * config.IMAGE_USD_EACH)


def test_cap_blocks_before_any_generation_request(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DAILY_CAP_USD", 0.01)
    api = Api(price=5.0)
    with pytest.raises(costs.CapExceeded):
        provider(api).generate_clip("p", "", refs(tmp_path, 1)[0], 4, tmp_path / "c.mp4")
    assert not api.posts


def test_missing_key(tmp_path):
    p = DeapiProvider(api_key="", transport=httpx.MockTransport(Api()))
    p.key = None
    assert p.missing_env() == ["DEAPI_API_KEY"]
    with pytest.raises(DeapiError, match="DEAPI_API_KEY"):
        p.models("img2img")


def test_state_reports_deapi_and_its_missing_key(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER", "deapi")
    monkeypatch.delenv("DEAPI_API_KEY", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    providers._instances.clear()
    from fastapi.testclient import TestClient
    from app import server
    st = TestClient(server.app).get("/api/state").json()
    assert st["config"]["provider"] == "deapi" and "DEAPI_API_KEY" in st["missing"] and st["generation_ready"] is False
    providers._instances.clear()


def test_size_keeps_9_16_when_the_model_caps_resolution(monkeypatch):
    # "LTX-Video 13B" in the deAPI dashboard: up to 768x768
    limits = {"min_width": 256, "max_width": 768, "min_height": 256, "max_height": 768, "resolution_step": 32}
    w, h = DeapiProvider._size(limits)
    assert h == 768 and w % 32 == 0 and w <= 768
    assert abs(w / h - 9 / 16) < 0.05  # clamping each side alone would have produced 576x768 (3:4)


def test_size_unchanged_when_within_limits():
    assert DeapiProvider._size({"max_width": 2048, "max_height": 2048, "resolution_step": 32}) == (576, 1024)


MINUTE = (429, {"X-RateLimit-Type": "minute", "Retry-After": "7"})


def test_submit_waits_out_the_per_minute_rate_limit(tmp_path):
    sleeps = []
    api = Api(post_failures=[MINUTE, MINUTE])
    api.download = b"mp4"
    p = DeapiProvider(api_key="k", transport=httpx.MockTransport(api), sleep=sleeps.append, poll_s=0)
    p.generate_clip("p", "", refs(tmp_path, 1)[0], 4, tmp_path / "c.mp4")
    assert api.post_attempts == 3 and (tmp_path / "c.mp4").read_bytes() == b"mp4"
    assert sleeps[:2] == [8.0, 8.0]  # Retry-After (7 s) plus a second


def test_submit_gives_up_after_repeated_minute_limits(tmp_path):
    api = Api(post_failures=[MINUTE] * 10)
    p = DeapiProvider(api_key="k", transport=httpx.MockTransport(api), sleep=lambda s: None, poll_s=0)
    with pytest.raises(DeapiError, match="rate limited"):
        p.generate_clip("p", "", refs(tmp_path, 1)[0], 4, tmp_path / "c.mp4")
    assert api.post_attempts == 5  # first try + 4 retries


def test_daily_limit_and_gateway_timeout_are_never_retried(tmp_path):
    for failure in ((429, {"X-RateLimit-Type": "daily", "Retry-After": "999"}), (504, {})):
        api = Api(post_failures=[failure, failure])
        with pytest.raises(DeapiError):
            provider(api).generate_clip("p", "", refs(tmp_path, 1)[0], 4, tmp_path / "c.mp4")
        assert api.post_attempts == 1  # a timed-out POST may already be a billable job


def test_reference_images_are_downscaled_to_jpeg(tmp_path):
    big = write_image(tmp_path / "big.png", np.full((2000, 3000, 3), 120, np.uint8))
    name, data, mime = DeapiProvider._file_part(big)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert name == "big.jpg" and mime == "image/jpeg" and max(img.shape[:2]) == 1280


def test_failed_quotes_are_not_retried_on_every_page_refresh():
    api = Api(price=None)
    p = provider(api)
    for _ in range(5):
        p.estimate_video(10)
        p.estimate_images(2)
    price_calls = sum(r.url.path.endswith("/price") for r in api.requests)
    assert price_calls == 2  # one failed attempt per endpoint, then backed off


def test_each_avatar_is_handed_back_as_soon_as_it_exists(tmp_path):
    seen = []
    out = provider(Api()).generate_avatar_images("p", refs(tmp_path, 1), None, 2, tmp_path / "av", "avatar_s0",
                                                 on_image=seen.append)
    assert seen == out and len(out) == 2
