import importlib

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
