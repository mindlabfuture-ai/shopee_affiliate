import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_frame, write_image
from ugc import config, costs


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    from app import server
    importlib.reload(server)
    return TestClient(server.app)


def png(tmp_path):
    return write_image(tmp_path / "p.png", make_frame((47, 45, 179)))


def make_product(client, tmp_path):
    with png(tmp_path).open("rb") as f:
        r = client.post("/api/products", data={"name": "Red Dress", "colorway": "deep red", "color_hex": "#b32d2f",
                        "features": "back: deep V open back"}, files=[("photos", ("p.png", f, "image/png"))])
    assert r.status_code == 200, r.text
    return r.json()["slug"]


def test_create_and_fetch_product(client, tmp_path):
    slug = make_product(client, tmp_path)
    st = client.get(f"/api/products/{slug}").json()
    assert st["spec"]["features"][0]["name"] == "back" and len(st["listing"]) == 1
    assert client.get(st["listing"][0]).status_code == 200


def test_rejects_bad_hex_and_non_image(client, tmp_path):
    assert client.post("/api/products", data={"name": "x", "colorway": "y", "color_hex": "red"}).status_code == 400
    r = client.post("/api/products", data={"name": "x", "colorway": "y"},
                    files=[("photos", ("a.exe", b"nope", "application/octet-stream"))])
    assert r.status_code == 400


def test_file_route_blocks_traversal(client, tmp_path):
    slug = make_product(client, tmp_path)
    assert client.get(f"/files/{slug}/..%2F..%2Fledger.jsonl").status_code == 404
    assert client.get(f"/files/{slug}/product.json").status_code == 200


def test_password_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    from app import server
    importlib.reload(server)
    c = TestClient(server.app)
    assert c.get("/api/state").status_code == 401
    assert c.post("/api/login", json={"password": "bad"}).status_code == 401
    assert c.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert c.get("/api/state").status_code == 200
    assert TestClient(server.app).get("/files/x/y").status_code == 401  # no cookie, no files


def test_cost_gate_requires_confirmation(client, tmp_path):
    slug = make_product(client, tmp_path)
    assert client.post(f"/api/products/{slug}/avatar", json={"n": 2}).status_code == 400  # needs plan, and confirm
    from ugc.spec import product_dir, write_json
    write_json(product_dir(slug) / "plan.json", {"storyboards": [{"index": 0, "avatar_image": {"prompt": "p"}, "clips": []}]})
    r = client.post(f"/api/products/{slug}/avatar", json={"n": 2, "confirm_usd": 0})
    assert r.status_code == 400 and "confirm" in r.json()["detail"]


def test_daily_cap(monkeypatch):
    monkeypatch.setattr(config, "DAILY_CAP_USD", 1.0)
    costs.record("video", 0.9)
    with pytest.raises(costs.CapExceeded):
        costs.guard(0.2)
    costs.guard(0.05)


def test_unknown_storyboard_is_rejected(client, tmp_path):
    slug = make_product(client, tmp_path)
    from ugc.spec import product_dir, write_json
    write_json(product_dir(slug) / "plan.json", {"storyboards": [{"index": 0, "avatar_image": {"prompt": "p"}, "clips": []}]})
    r = client.post(f"/api/products/{slug}/avatar", json={"n": 1, "storyboard": 7, "confirm_usd": 5})
    assert r.status_code == 400 and "storyboard" in r.json()["detail"]


# ---------- manual mode ----------
import time

import cv2

from tests.conftest import needs_ffmpeg, write_video


def wait(client, job, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise AssertionError("job timed out")


def test_avatar_upload_is_checked_and_listed(client, tmp_path):
    slug = make_product(client, tmp_path)
    with png(tmp_path).open("rb") as f:
        r = client.post(f"/api/products/{slug}/avatar/upload", data={"storyboard": "0"},
                        files=[("photos", ("mine.png", f, "image/png"))])
    assert r.status_code == 200
    j = wait(client, r.json()["job"])
    assert j["status"] == "done" and j["result"] == ["avatar_s0_01.png"]
    st = client.get(f"/api/products/{slug}").json()
    assert st["avatars"][0]["name"] == "avatar_s0_01.png"
    assert st["avatars"][0]["qa"]["overall"] in ("pass", "warn")  # same gate as generated stills
    assert client.post(f"/api/products/{slug}/approve", json={"avatar": "avatar_s0_01.png"}).status_code == 200


def test_avatar_upload_rejects_bad_files(client, tmp_path):
    slug = make_product(client, tmp_path)
    bad_type = client.post(f"/api/products/{slug}/avatar/upload", files=[("photos", ("a.gif", b"x", "image/gif"))])
    corrupt = client.post(f"/api/products/{slug}/avatar/upload", files=[("photos", ("a.png", b"not a png", "image/png"))])
    assert bad_type.status_code == 400 and corrupt.status_code == 400


@needs_ffmpeg
def test_clip_upload_joins_mixed_sizes_strips_audio_and_checks(client, tmp_path):
    slug = make_product(client, tmp_path)
    red = (47, 45, 179)
    a = write_video(tmp_path / "clip1.mp4", [make_frame(red) for _ in range(36)], audio=True)  # 360x640 with audio
    b = write_video(tmp_path / "clip2.mp4", [make_frame(red, size=(540, 960)) for _ in range(36)])  # different size
    files = [("videos", (p.name, p.open("rb"), "video/mp4")) for p in (a, b)]
    r = client.post(f"/api/products/{slug}/clips/upload", files=files)
    assert r.status_code == 200
    j = wait(client, r.json()["job"], timeout=120)
    assert j["status"] == "done", j
    names = {c["name"]: c for c in j["result"]["checks"]}
    assert names["audio"]["status"] == "pass"  # stripped by the join
    assert names["garment_color"]["status"] == "pass"
    st = client.get(f"/api/products/{slug}").json()
    assert st["final"] and len(st["clips"]) == 2
    cap = cv2.VideoCapture(str(config.DATA_DIR / "products" / slug / "final.mp4"))
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (360, 640)


def test_clip_upload_rejects_bad_type(client, tmp_path):
    slug = make_product(client, tmp_path)
    r = client.post(f"/api/products/{slug}/clips/upload", files=[("videos", ("x.exe", b"x", "application/octet-stream"))])
    assert r.status_code == 400


# ---------- production hardening ----------
def test_locked_when_password_required_but_missing(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setenv("UGC_REQUIRE_PASSWORD", "1")
    from app import server
    importlib.reload(server)
    c = TestClient(server.app)
    assert c.get("/health").json() == {"ok": True}  # healthcheck still answers
    assert c.get("/api/state").status_code == 503
    assert c.post("/api/login", json={"password": ""}).status_code == 503


def test_login_lockout_after_repeated_failures(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.setenv("UGC_SECURE_COOKIE", "1")
    from app import server
    importlib.reload(server)
    c = TestClient(server.app)
    codes = [c.post("/api/login", json={"password": "bad"}).status_code for _ in range(10)]
    assert codes[:8] == [401] * 8 and codes[8:] == [429, 429]
    assert c.post("/api/login", json={"password": "s3cret"}).status_code == 429  # even the right one while locked


def test_secure_cookie_flag(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.setenv("UGC_SECURE_COOKIE", "1")
    from app import server
    importlib.reload(server)
    r = TestClient(server.app).post("/api/login", json={"password": "s3cret"})
    cookie = r.headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie and "samesite=strict" in cookie


# ---------- failures that must stay visible, and resuming a half-finished video ----------
class FakeProv:
    name = "fake"

    def __init__(self, fail_on=None):
        self.calls, self.fail_on, self.starts = [], fail_on, []

    def missing_env(self): return []
    def describe(self): return {"provider": "fake", "image": "i", "video": "v", "resolution": "r"}
    def estimate_images(self, n): return 0.0
    def estimate_video(self, seconds): return 0.0

    def generate_clip(self, prompt, negative_prompt, image, duration_s, out_path):
        self.calls.append(out_path.name)
        self.starts.append(Path(image).name)
        if out_path.name == self.fail_on:
            raise RuntimeError("deAPI POST /api/v2/videos/animations -> 429: Too Many Attempts.")
        write_video(out_path, [make_frame((47, 45, 179)) for _ in range(12)])
        return out_path


@needs_ffmpeg
def test_video_failure_is_saved_and_the_retry_only_pays_for_the_missing_clip(client, tmp_path, monkeypatch):
    from ugc import providers
    from ugc.spec import product_dir, write_json
    fake = FakeProv(fail_on="clip_02.mp4")
    monkeypatch.setitem(providers._instances, config.PROVIDER, fake)
    slug = make_product(client, tmp_path)
    d = product_dir(slug)
    write_json(d / "plan.json", {"tiktok_caption": "AI-generated", "storyboards": [{"index": 0, "clips": [
        {"index": 1, "duration_s": 4, "prompt": "p", "negative_prompt": "n"},
        {"index": 2, "duration_s": 4, "prompt": "p", "negative_prompt": "n"}]}]})
    write_image(d / "avatars" / "avatar_s0_01.png", make_frame((47, 45, 179)))
    write_json(d / "approved.json", {"avatar": "avatar_s0_01.png"})

    r = client.post(f"/api/products/{slug}/video", json={"confirm_usd": 1})
    assert wait(client, r.json()["job"], timeout=120)["status"] == "error"
    st = client.get(f"/api/products/{slug}").json()
    assert "429" in st["last_error"]["error"] and st["final"] is None  # the page can still show what went wrong
    assert (d / "clips" / "clip_01.mp4").exists()  # the finished clip is kept

    fake.fail_on = None
    r = client.post(f"/api/products/{slug}/video", json={"confirm_usd": 1})
    assert wait(client, r.json()["job"], timeout=120)["status"] == "done"
    assert fake.calls == ["clip_01.mp4", "clip_02.mp4", "clip_02.mp4"]  # clip 1 was not generated (or paid for) again
    st = client.get(f"/api/products/{slug}").json()
    assert st["final"] and st["last_error"] is None


def test_wrong_password_says_how_many_attempts_are_left_and_how_long_a_lockout_lasts(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    from app import server
    importlib.reload(server)
    c = TestClient(server.app)
    first = c.post("/api/login", json={"password": "bad"})
    assert first.status_code == 401 and "7 attempts left" in first.json()["detail"]
    for _ in range(6):
        last = c.post("/api/login", json={"password": "bad"})
    assert "1 attempt left" in last.json()["detail"]
    final = c.post("/api/login", json={"password": "bad"})
    assert final.status_code == 401 and "Locked for 10 minutes" in final.json()["detail"]
    locked = c.post("/api/login", json={"password": "s3cret"})
    assert locked.status_code == 429 and "Try again in" in locked.json()["detail"]
    assert 1 <= int(locked.headers["retry-after"]) <= 600


def test_a_correct_password_clears_the_failed_attempts(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    from app import server
    importlib.reload(server)
    c = TestClient(server.app)
    for _ in range(5):
        c.post("/api/login", json={"password": "typo"})
    assert c.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert "7 attempts left" in c.post("/api/login", json={"password": "typo"}).json()["detail"]  # counter reset


def plan3(prompt="p"):
    clips = [{"index": i, "duration_s": 4, "prompt": f"{prompt}{i}", "negative_prompt": "n", "chained": i > 1}
             for i in (1, 2, 3)]
    return {"tiktok_caption": "AI-generated", "storyboards": [{"index": 0, "clips": clips}]}


def setup_chain(client, tmp_path, monkeypatch, fake):
    from ugc import providers
    from ugc.spec import product_dir, write_json
    monkeypatch.setitem(providers._instances, config.PROVIDER, fake)
    slug = make_product(client, tmp_path)
    d = product_dir(slug)
    write_json(d / "plan.json", plan3())
    write_image(d / "avatars" / "avatar_s0_01.png", make_frame((47, 45, 179)))
    write_json(d / "approved.json", {"avatar": "avatar_s0_01.png"})
    return slug, d


@needs_ffmpeg
def test_each_clip_starts_from_the_previous_clips_last_frame(client, tmp_path, monkeypatch):
    fake = FakeProv()
    slug, d = setup_chain(client, tmp_path, monkeypatch, fake)
    j = wait(client, client.post(f"/api/products/{slug}/video", json={"confirm_usd": 1}).json()["job"], timeout=180)
    assert j["status"] == "done", j
    assert fake.starts == ["avatar_s0_01.png", "start_02.png", "start_03.png"]  # only clip 1 starts from the avatar
    assert (d / "clips" / "start_02.png").exists() and (d / "clips" / "start_03.png").exists()


@needs_ffmpeg
def test_a_failed_chain_resumes_from_the_last_finished_clip(client, tmp_path, monkeypatch):
    fake = FakeProv(fail_on="clip_03.mp4")
    slug, d = setup_chain(client, tmp_path, monkeypatch, fake)
    assert wait(client, client.post(f"/api/products/{slug}/video", json={"confirm_usd": 1}).json()["job"],
                timeout=180)["status"] == "error"
    fake.fail_on, fake.calls, fake.starts = None, [], []
    j = wait(client, client.post(f"/api/products/{slug}/video", json={"confirm_usd": 1}).json()["job"], timeout=180)
    assert j["status"] == "done"
    assert fake.calls == ["clip_03.mp4"] and fake.starts == ["start_03.png"]  # chained from the clip kept on disk


@needs_ffmpeg
def test_changing_the_plan_discards_clips_made_for_the_old_one(client, tmp_path, monkeypatch):
    from ugc.spec import write_json
    fake = FakeProv()
    slug, d = setup_chain(client, tmp_path, monkeypatch, fake)
    wait(client, client.post(f"/api/products/{slug}/video", json={"confirm_usd": 1}).json()["job"], timeout=180)
    fake.calls = []
    write_json(d / "plan.json", plan3("new prompt "))  # a regenerated plan, same avatar and storyboard number
    j = wait(client, client.post(f"/api/products/{slug}/video", json={"confirm_usd": 1}).json()["job"], timeout=180)
    assert j["status"] == "done" and fake.calls == ["clip_01.mp4", "clip_02.mp4", "clip_03.mp4"]
