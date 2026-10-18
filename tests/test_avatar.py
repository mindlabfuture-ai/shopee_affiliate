import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_frame, write_image
from ugc import avatar, config, providers


def face_image(tone):
    return make_frame((tone, tone, tone))


class FakeProv:
    """Writes a distinct image per call, like an image-edit provider would, and records what it was given."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def missing_env(self): return []
    def describe(self): return {"provider": "fake", "image": "i", "video": "v", "resolution": "r"}
    def estimate_images(self, n): return round(0.04 * n, 4)
    def estimate_video(self, s): return 0.0

    def generate_avatar_images(self, prompt, garments, avatar_ref, n, out_dir, prefix, on_image=None):
        self.calls.append({"prompt": prompt, "garments": [Path(g).name for g in garments],
                           "avatar_ref": [Path(a).name for a in ([avatar_ref] if isinstance(avatar_ref, Path) else (avatar_ref or []))],
                           "prefix": prefix})
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{len(self.calls):02d}.png"
        write_image(path, face_image(60 + 20 * len(self.calls)))
        if on_image:
            on_image(path)
        return [path]


def review(*statuses, hair=True):
    names = ["face shape", "eyes", "nose", "lips", "skin tone"]
    return avatar.FaceReview(
        attributes=[avatar.AttributeCheck(attribute=n, status=s, note="") for n, s in zip(names, statuses)],
        hair_as_requested=hair, summary="s")


# ---------- prompts ----------
def test_every_view_locks_the_face_and_names_its_angle():
    for view in avatar.VIEWS:
        p = avatar.view_prompt(view, "a low bun")
        assert "identical face shape" in p and "Do not beautify" in p and "Hairstyle: a low bun." in p
    assert "Close-up" in avatar.view_prompt("closeup", None)
    assert "left side" in avatar.view_prompt("left", None) and "left edge" in avatar.view_prompt("left", None)
    assert "right side" in avatar.view_prompt("right", None) and "right edge" in avatar.view_prompt("right", None)


def test_blank_hairstyle_keeps_the_hair():
    for blank in (None, "", "   "):
        assert "Keep the same hairstyle and hair color" in avatar.view_prompt("front", blank)


def test_unknown_view_is_rejected():
    with pytest.raises(ValueError):
        avatar.view_path("back")


# ---------- consistency verdict ----------
def test_verdict_rules():
    assert avatar.verdict(review("same", "same", "same", "same", "same")) == "consistent"
    assert avatar.verdict(review("same", "not_visible", "same", "same", "same")) == "consistent"  # angle hides it
    assert avatar.verdict(review("changed", "same", "same", "same", "same")) == "drifted"  # face shape
    assert avatar.verdict(review("same", "same", "same", "changed", "same")) == "drifted"  # lips
    assert avatar.verdict(review("same", "same", "same", "same", "changed")) == "check"  # skin tone: human look
    assert avatar.verdict(review("same", "same", "same", "same", "same", hair=False)) == "check"


# ---------- generation ----------
class FakeClient:
    def __init__(self, result):
        self.result, self.calls = result, []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=self.result)


def test_generate_view_stores_archives_and_checks():
    prov = FakeProv()
    write_image(avatar.avatar_dir() / "base.png", face_image(100))
    client = FakeClient(review("same", "same", "same", "same", "same"))
    check = avatar.generate_view(prov, "front", "a low bun", client)
    assert check["verdict"] == "consistent" and avatar.view_path("front").exists()
    assert prov.calls[0]["garments"] == ["base.png"] and prov.calls[0]["avatar_ref"] == []
    first = cv2.imread(str(avatar.view_path("front"))).mean()

    avatar.generate_view(prov, "left", "a low bun", client)  # other views also see the approved front view
    assert prov.calls[1]["garments"] == ["base.png", "front.png"]

    avatar.generate_view(prov, "front", "a low bun", client)  # redo: old one is archived, not lost
    assert len(list((avatar.sheet_dir() / "history").glob("front_*.png"))) == 1
    assert cv2.imread(str(avatar.view_path("front"))).mean() != first
    assert json.loads((avatar.sheet_dir() / "front.json").read_text())["verdict"] == "consistent"


def test_generate_view_without_a_key_still_keeps_the_image():
    write_image(avatar.avatar_dir() / "base.png", face_image(100))
    check = avatar.generate_view(FakeProv(), "closeup", None, client=None)
    assert check["verdict"] == "unchecked" and avatar.view_path("closeup").exists()


def test_a_failed_check_never_loses_the_image():
    write_image(avatar.avatar_dir() / "base.png", face_image(100))
    client = SimpleNamespace(messages=SimpleNamespace(parse=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))))
    check = avatar.generate_view(FakeProv(), "front", None, client)
    assert check["verdict"] == "unchecked" and "boom" in check["summary"] and avatar.view_path("front").exists()


def test_needs_a_base_photo():
    with pytest.raises(RuntimeError, match="base avatar photo"):
        avatar.generate_view(FakeProv(), "front", None)


def test_reference_files_prefer_the_sheet_then_the_legacy_image():
    assert avatar.reference_files() == []
    write_image(config.DATA_DIR / "avatar_reference.png", face_image(90))
    assert [p.name for p in avatar.reference_files()] == ["avatar_reference.png"]
    for v in ("right", "closeup", "front"):
        write_image(avatar.view_path(v), face_image(80))
    assert [p.name for p in avatar.reference_files()] == ["front.png", "closeup.png", "right.png"]


# ---------- app ----------
@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app import server
    importlib.reload(server)
    prov = FakeProv()
    monkeypatch.setitem(providers._instances, config.PROVIDER, prov)
    c = TestClient(server.app)
    c.prov = prov
    return c


def wait(client, job, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job timed out")


def upload_base(client, tmp_path):
    with write_image(tmp_path / "me.png", face_image(100)).open("rb") as f:
        return client.post("/api/avatar/base", files={"photo": ("me.png", f, "image/png")})


def test_full_sheet_flow(client, tmp_path):
    assert client.get("/api/avatar").json()["base"] is None
    assert upload_base(client, tmp_path).json()["base"].startswith("/avatar-files/base.png")
    est = client.post("/api/avatar/sheet/estimate", json={}).json()
    assert est["images"] == 4

    assert client.post("/api/avatar/sheet", json={"hairstyle": "a low bun"}).status_code == 400  # must confirm cost
    r = client.post("/api/avatar/sheet", json={"hairstyle": "a low bun", "confirm_usd": est["usd"]})
    j = wait(client, r.json()["job"])
    assert j["status"] == "done" and set(j["result"]) == {"front", "closeup", "left", "right"}
    assert [c["prefix"] for c in client.prov.calls] == ["front", "closeup", "left", "right"]  # front first

    st = client.get("/api/avatar").json()
    assert all(st["views"][v] for v in ("closeup", "front", "left", "right"))
    assert st["hairstyle"] == "a low bun" and "a low bun" in st["prompts"]["left"]
    assert client.get(st["views"]["front"]["url"]).status_code == 200


def test_redo_one_view_costs_one_image(client, tmp_path):
    upload_base(client, tmp_path)
    est = client.post("/api/avatar/sheet/estimate", json={"views": ["left"]}).json()
    assert est["images"] == 1
    j = wait(client, client.post("/api/avatar/sheet", json={"views": ["left"], "confirm_usd": est["usd"]}).json()["job"])
    assert list(j["result"]) == ["left"] and len(client.prov.calls) == 1


def test_avatar_endpoints_reject_bad_input(client, tmp_path):
    assert client.post("/api/avatar/sheet", json={"confirm_usd": 9}).status_code == 400  # no base photo yet
    assert client.post("/api/avatar/base", files={"photo": ("a.gif", b"x", "image/gif")}).status_code == 400
    assert client.post("/api/avatar/base", files={"photo": ("a.png", b"not an image", "image/png")}).status_code == 400
    upload_base(client, tmp_path)
    assert client.post("/api/avatar/sheet", json={"views": ["back"], "confirm_usd": 9}).status_code == 400


def test_avatar_files_are_protected(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    from app import server
    importlib.reload(server)
    c = TestClient(server.app)
    assert c.get("/avatar-files/base.png").status_code == 401
    assert c.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert c.get("/avatar-files/..%2Fledger.jsonl").status_code == 404
    assert c.get("/avatar-files/settings.json").status_code == 404  # images only


def test_product_shoots_use_the_whole_sheet_as_identity_references(client, tmp_path):
    from ugc.spec import product_dir, write_json
    for v in ("front", "closeup", "left", "right"):
        write_image(avatar.view_path(v), face_image(80))
    with write_image(tmp_path / "p.png", make_frame((47, 45, 179))).open("rb") as f:
        slug = client.post("/api/products", data={"name": "Red Dress", "colorway": "red"},
                           files=[("photos", ("p.png", f, "image/png"))]).json()["slug"]
    write_json(product_dir(slug) / "plan.json", {"storyboards": [{"index": 0, "avatar_image": {"prompt": "p"}, "clips": []}]})
    j = wait(client, client.post(f"/api/products/{slug}/avatar", json={"n": 1, "confirm_usd": 1}).json()["job"])
    assert j["status"] == "done"
    assert client.prov.calls[-1]["avatar_ref"] == ["front.png", "closeup.png", "left.png", "right.png"]
    assert "keep her face" in client.prov.calls[-1]["prompt"]
