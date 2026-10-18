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
