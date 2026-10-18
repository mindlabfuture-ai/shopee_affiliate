import importlib
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_frame, needs_ffmpeg, write_image, write_video
from tests.test_planner import (BACK, HIP, SPEC, STAND, board, clip, good_plan, listing, FakeClient as PlanClient)
from ugc import config, llm, motion, planner
from ugc.motion import MotionBeat, MotionReference
from ugc.planner import ClipPlan


def beat(s, e, action="pose_hold", camera="static", start=STAND, end=HIP):
    return MotionBeat(start_s=s, end_s=e, action=action, action_detail="moves slowly", camera=camera,
                      camera_detail="steady", start_state=start, end_state=end)


def reference(*beats, notes="keep it relaxed"):
    return MotionReference(pacing="slow and relaxed", camera_summary="slow push in", beats=list(beats), notes=notes)


class Client:
    def __init__(self, result):
        self.result, self.calls = result, []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=self.result)


def ref_video(tmp_path, seconds=3, name="ref.mp4"):
    frames = [make_frame((20 + 8 * (i % 20), 40, 90)) for i in range(12 * seconds)]
    return write_video(tmp_path / name, frames, fps=12)


# ---------- analysis ----------
@needs_ffmpeg
def test_analysis_sends_labelled_frames_and_forbids_describing_the_person(tmp_path):
    client = Client(reference(beat(0, 1.5), beat(1.5, 3, "lean", "slow_push_in", HIP, BACK)))
    out = motion.analyze(client, ref_video(tmp_path))
    call = client.calls[0]
    kinds = [b["type"] for b in call["messages"][0]["content"]]
    assert kinds.count("image") >= 8 and any("FRAME @" in b.get("text", "") for b in call["messages"][0]["content"])
    system = " ".join(call["system"].split())
    for needle in ("ONLY the choreography and the camera", "Never describe or mention who the person is",
                   "Never name anyone"):
        assert needle in system
    assert out["duration_s"] == pytest.approx(3.0, abs=0.3) and len(out["beats"]) == 2


@needs_ffmpeg
def test_beats_are_cleaned_up(tmp_path):
    messy = reference(beat(2.5, 9.0, "moonwalk", "spin", HIP, BACK), beat(-1, 1.0))
    out = motion.analyze(Client(messy), ref_video(tmp_path))
    first, second = out["beats"]
    assert first["start_s"] == 0.0 and second["action"] == "other" and second["camera"] == "other"  # sorted, mapped
    assert second["end_s"] <= out["duration_s"]  # clamped to the clip


@needs_ffmpeg
def test_a_reference_over_30_seconds_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="keep it under 30s"):
        motion.analyze(Client(reference(beat(0, 1))), ref_video(tmp_path, seconds=31, name="long.mp4"))


def test_prompt_block_lists_beats_cuts_and_notes():
    ref = {**reference(beat(0, 2), beat(2, 4, "hair_touch", "tilt_up", HIP, BACK)).model_dump(),
           "duration_s": 4.0, "cuts": [2.0]}
    text = motion.to_prompt_block(ref)
    assert "MOTION REFERENCE (4.0 s" in text and "1. 0.0-2.0s" in text and "2. 2.0-4.0s | action: hair_touch" in text
    assert "camera cuts at 2.0s" in text and "one continuous take" in text and "Notes: keep it relaxed" in text


# ---------- the planner follows the reference ----------
REF = {**reference(beat(0, 4), beat(4, 8, "hair_touch", "slow_push_in", HIP, BACK)).model_dump(),
       "duration_s": 8.0, "cuts": []}


def walk_twice():
    cl = [clip(4, ["front", "full_length"], "walk", "static", STAND, HIP),
          clip(4, ["back", "side"], "turn_over_shoulder", "static", HIP, BACK),
          clip(4, ["side"], "walk", "tilt_up", BACK, "end pose")]
    return good_plan(storyboards=[board("a", clips=cl), board("b", "blue hour", clips=cl),
                                  board("c", "soft overcast", clips=cl)])


def test_a_reference_relaxes_variety_but_not_continuity():
    plain = [i for i in planner.lint(SPEC, walk_twice()) if i.level == "error"]
    assert any("at most once" in i.msg for i in plain)  # without a reference walking twice is an error
    with_ref = planner.lint(SPEC, walk_twice(), {**REF, "duration_s": 12.0})
    assert not [i for i in with_ref if i.level == "error"]
    assert any(i.level == "warn" and "at most once" in i.msg for i in with_ref)

    broken = walk_twice()
    broken.storyboards[0].clips[1] = clip(4, ["back"], "lean", "static", "somewhere else", BACK)
    assert any("word for word" in i.msg for i in planner.lint(SPEC, broken, {**REF, "duration_s": 12.0}) if i.level == "error")


def test_the_reference_sets_the_target_duration():
    assert not [i for i in planner.lint(SPEC, good_plan(), REF) if i.level == "error"]  # plan is 10 s, ref 8 s: within 2
    long_ref = {**REF, "duration_s": 20.0}
    msgs = [i.msg for i in planner.lint(SPEC, good_plan(), long_ref) if i.level == "error"]
    assert any("target is 20s" in m and "motion reference" in m for m in msgs)


def test_the_planner_gets_the_motion_script_and_the_copy_rules(tmp_path):
    client = PlanClient(good_plan())
    result = planner.make_plan(SPEC, listing(tmp_path), client=client, max_revisions=0, reference=REF)
    call = client.calls[0]
    system = " ".join(call["system"].split())
    assert "COPY this choreography" in system and "never anything about the person" in system
    texts = [b.get("text", "") for b in call["messages"][0]["content"]]
    assert any(t.startswith("MOTION REFERENCE (8.0 s") for t in texts)
    assert result["reference_used"] is True


def test_without_a_reference_nothing_changes(tmp_path):
    client = PlanClient(good_plan())
    result = planner.make_plan(SPEC, listing(tmp_path), client=client, max_revisions=0)
    assert "COPY this choreography" not in client.calls[0]["system"] and result["reference_used"] is False


# ---------- app ----------
@pytest.fixture
def app_client(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from app import server
    importlib.reload(server)
    return TestClient(server.app)


def make_product(c, tmp_path):
    with write_image(tmp_path / "p.png", make_frame((47, 45, 179))).open("rb") as f:
        return c.post("/api/products", data={"name": "Red Dress", "colorway": "red"},
                      files=[("photos", ("p.png", f, "image/png"))]).json()["slug"]


def wait(c, job, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/api/jobs/{job}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise AssertionError("timed out")


def upload(c, slug, path):
    with path.open("rb") as f:
        return c.post(f"/api/products/{slug}/reference", files={"video": (path.name, f, "video/mp4")})


@needs_ffmpeg
def test_reference_upload_analysis_state_and_plan(app_client, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: Client(reference(beat(0, 1.5), beat(1.5, 3, "lean", "static", HIP, BACK))))
    slug = make_product(app_client, tmp_path)
    j = wait(app_client, upload(app_client, slug, ref_video(tmp_path)).json()["job"])
    assert j["status"] == "done" and len(j["result"]["beats"]) == 2

    st = app_client.get(f"/api/products/{slug}").json()
    assert st["reference"]["url"].endswith("reference/clip.mp4") and len(st["reference"]["motion"]["beats"]) == 2
    assert app_client.get(st["reference"]["url"]).status_code == 200

    seen = {}
    monkeypatch.setattr(planner, "make_plan", lambda spec, listing, **kw: seen.update(kw) or {"storyboards": []})
    wait(app_client, app_client.post(f"/api/products/{slug}/plan").json()["job"])
    assert seen["reference"]["duration_s"] == pytest.approx(3.0, abs=0.3)  # the plan is made from the reference

    assert app_client.delete(f"/api/products/{slug}/reference").status_code == 200
    assert app_client.get(f"/api/products/{slug}").json()["reference"] is None


@needs_ffmpeg
def test_a_clip_that_cannot_be_analyzed_is_not_kept(app_client, tmp_path, monkeypatch):
    class Boom:
        messages = SimpleNamespace(parse=lambda **kw: (_ for _ in ()).throw(RuntimeError("model said no")))
    monkeypatch.setattr(llm, "get_client", lambda: Boom())
    slug = make_product(app_client, tmp_path)
    j = wait(app_client, upload(app_client, slug, ref_video(tmp_path)).json()["job"])
    assert j["status"] == "error" and "model said no" in j["error"]
    st = app_client.get(f"/api/products/{slug}").json()
    assert st["reference"] is None and "model said no" in st["last_error"]["error"]


def test_reference_upload_rejects_bad_requests(app_client, tmp_path, monkeypatch):
    slug = make_product(app_client, tmp_path)
    bad = app_client.post(f"/api/products/{slug}/reference", files={"video": ("x.exe", b"x", "application/octet-stream")})
    assert bad.status_code == 400
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    nokey = app_client.post(f"/api/products/{slug}/reference", files={"video": ("a.mp4", b"x", "video/mp4")})
    assert nokey.status_code == 400 and "ANTHROPIC_API_KEY" in nokey.json()["detail"]
