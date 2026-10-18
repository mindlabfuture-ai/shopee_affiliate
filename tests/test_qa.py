from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import make_frame, needs_ffmpeg, write_image, write_video
from ugc import qa
from ugc.qa import color, vision
from ugc.imaging import hex_to_lab, lab_to_hex
from ugc.spec import Feature, ProductSpec

RED = (47, 45, 179)  # BGR of #b32d2f
SPEC = ProductSpec(slug="red", name="red dress", colorway="deep red", color_hex="#b32d2f",
                   features=[Feature(name="back", description="deep V open back")])


def listing(tmp_path):
    return [write_image(tmp_path / "ref.png", make_frame(RED))]


def by_name(report):
    return {c.name: c for c in report.checks}


def test_hex_roundtrip():
    assert lab_to_hex(hex_to_lab("#b32d2f")) == "#b32d2f"


def test_color_pass_on_matching_frames():
    frames = [(i * 0.5, make_frame(RED)) for i in range(8)]
    checks = {c.name: c for c in color.check_color(frames, hex_to_lab("#b32d2f"))}
    assert checks["garment_color"].status == "pass"
    assert checks["color_stability"].status == "pass"


def test_color_flags_drift_and_missing_garment():
    shifted = (47, 80, 190)  # dE 15.7 from the listing: inside the match window, but clearly off
    frames = [(i * 0.5, make_frame(RED if i < 4 else shifted)) for i in range(8)]
    checks = {c.name: c for c in color.check_color(frames, hex_to_lab("#b32d2f"))}
    assert checks["color_stability"].status == "fail"
    far = (47, 100, 200)  # dE 26: outside the window, so it shows up as "garment color not found"
    frames = [(i * 0.5, make_frame(RED if i < 4 else far)) for i in range(8)]
    checks = {c.name: c for c in color.check_color(frames, hex_to_lab("#b32d2f"))}
    assert checks["garment_color_found"].status in ("warn", "fail")
    blue = [(i * 0.5, make_frame((200, 80, 30))) for i in range(8)]
    out = {c.name: c for c in color.check_color(blue, hex_to_lab("#b32d2f"))}
    assert out["garment_color_found"].status == "fail"


def test_join_jump_detected():
    frames = [(i * 0.5, make_frame(RED if i < 4 else (47, 90, 190))) for i in range(8)]
    out = {c.name: c for c in color.check_color(frames, hex_to_lab("#b32d2f"), cuts=[2.0])}
    assert out["join_color_jump"].status in ("warn", "fail")


@needs_ffmpeg
def test_video_qa_clean(tmp_path):
    v = write_video(tmp_path / "ok.mp4", [make_frame(RED) for _ in range(60)])
    r = qa.run_video_qa(SPEC, listing(tmp_path), v, use_vision=False)
    c = by_name(r)
    assert c["audio"].status == "pass" and c["black_frames"].status == "pass"
    assert c["garment_color"].status == "pass"
    assert r.meta["video_sha256"]


@needs_ffmpeg
def test_video_qa_flags_audio_and_black_tail(tmp_path):
    v = write_video(tmp_path / "bad.mp4", [make_frame(RED) for _ in range(60)], tail_black=18, audio=True)
    c = by_name(qa.run_video_qa(SPEC, listing(tmp_path), v, use_vision=False))
    assert c["audio"].status == "warn"
    assert c["black_frames"].status == "warn"


def test_image_qa(tmp_path):
    img = write_image(tmp_path / "avatar.png", make_frame(RED))
    r = qa.run_image_qa(SPEC, listing(tmp_path), img, use_vision=False)
    assert by_name(r)["garment_color"].status == "pass"


def test_missing_listing_fails(tmp_path):
    img = write_image(tmp_path / "avatar.png", make_frame(RED))
    assert qa.run_image_qa(SPEC, [], img, use_vision=False).overall == "fail"


def test_caption_lint():
    assert qa.lint_caption("Sexy back, instant slimming")[0].status == "warn"
    assert qa.lint_caption("Open-back midi dress. AI-generated video")[0].status == "pass"


def _review(statuses, issues=()):
    return vision.VisionReview(
        features=[vision.FeatureCheck(feature="back", status=s, evidence="e") for s in statuses],
        frame_issues=[vision.FrameIssue(frame_label="@1s", issue=i) for i in issues],
        color_assessment="ok", fit_and_length_assessment="ok", summary="s")


def test_vision_verdicts():
    assert vision.to_checks(SPEC, _review(["mismatch"]))[0].status == "fail"
    assert vision.to_checks(SPEC, _review(["not_visible"]))[0].status == "warn"
    assert vision.to_checks(SPEC, _review(["match"]))[0].status == "pass"
    assert vision.to_checks(SPEC, _review(["match"], ["warped print"]))[1].status == "warn"


def test_vision_call_sends_images_and_uses_parsed_output(tmp_path):
    captured = {}

    class Fake:
        class messages:
            @staticmethod
            def parse(**kw):
                captured.update(kw)
                return SimpleNamespace(stop_reason="end_turn", parsed_output=_review(["match"]))

    frames = [(i * 0.5, make_frame(RED)) for i in range(10)]
    r = qa.run_video_qa if False else None  # noqa: F841
    from ugc.imaging import read_image
    out = vision.review(Fake, SPEC, [read_image(p) for p in listing(tmp_path)], vision.pick_frames(frames, [2.0]))
    assert out.summary == "s"
    kinds = [b["type"] for b in captured["messages"][0]["content"]]
    assert kinds.count("image") >= 9
    assert captured["model"] == "claude-sonnet-5-5"
