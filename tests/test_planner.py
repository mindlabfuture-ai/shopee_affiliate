from types import SimpleNamespace

from tests.conftest import make_frame, write_image
from ugc import planner
from ugc.planner import ClipPlan, PromptPlan, Storyboard
from ugc.spec import Feature, ProductSpec

SPEC = ProductSpec(slug="red", name="red backless slip dress", colorway="deep red", color_hex="#b32d2f",
                   length="ankle-length maxi", target_duration_s=10, required_shots=["front", "back", "full_length"],
                   features=[Feature(name="back", description="deep V open back"),
                             Feature(name="neckline", description="gathered scoop neckline")])


def clips():
    return [ClipPlan(duration_s=4, shots=["front", "full_length"], motion_prompt="slow quarter turn, soft smile"),
            ClipPlan(duration_s=6, shots=["back", "walk"], motion_prompt="turns to show the back, glances over shoulder")]


def board(location, tod="golden hour", **over):
    base = dict(title=f"Shoot at {location}", location=location, time_of_day=tod, mood="warm and relaxed",
                avatar_image_prompt="standing front pose, soft natural light, full body", clips=clips())
    return Storyboard(**{**base, **over})


def good_plan(**over):
    base = dict(
        storyboards=[board("tropical garden path"), board("seaside promenade", "blue hour"),
                     board("old-town stone lane", "soft overcast")],
        hook_captions=["Elegant open-back dress"], tiktok_caption="Open-back maxi. AI-generated video",
        shopee_caption="Open-back maxi dress. AI-generated video", risks=["open back must match listing"])
    return PromptPlan(**{**base, **over})


class FakeClient:
    def __init__(self, *plans):
        self.plans, self.calls = list(plans), []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=self.plans.pop(0))


def listing(tmp_path):
    return [write_image(tmp_path / "ref.png", make_frame((47, 45, 179)))]


def test_lock_clause_contains_spec():
    lock = planner.lock_clause(SPEC)
    for s in ("red backless slip dress", "#b32d2f", "deep V open back", "EXACTLY", "ankle-length maxi"):
        assert s in lock


def test_lint_clean():
    assert not [i for i in planner.lint(SPEC, good_plan()) if i.level == "error"]


def test_lint_catches_violations():
    bad = good_plan(
        storyboards=[board("park", clips=[
            ClipPlan(duration_s=5, shots=["front"], motion_prompt="make the dress brighter and she says hello"),
            ClipPlan(duration_s=8, shots=["wat"], motion_prompt="walk")])],
        hook_captions=["Sexy back!"])
    msgs = " | ".join(i.msg for i in planner.lint(SPEC, bad))
    for s in ("duration must be", "unknown shot tag", "required shots not covered", "change the garment",
              "speech/audio", "sexy", "at least 3 different outdoor storyboards"):
        assert s in msgs


def test_lint_requires_distinct_outdoor_locations():
    dup = good_plan(storyboards=[board("park"), board("Park", "blue hour"), board("rooftop", "soft overcast")])
    assert "different outdoor locations" in " | ".join(i.msg for i in planner.lint(SPEC, dup))
    same_light = good_plan(storyboards=[board("a"), board("b"), board("c")])
    assert any(i.level == "warn" and "time of day" in i.msg for i in planner.lint(SPEC, same_light))


def test_plan_revises_after_lint_errors_and_locks_garment(tmp_path):
    bad = good_plan(hook_captions=["Sexy back"])
    client = FakeClient(bad, good_plan())
    result = planner.make_plan(SPEC, listing(tmp_path), client=client)
    assert len(client.calls) == 2
    assert "Sexy back" in client.calls[1]["messages"][0]["content"][-1]["text"]  # feedback sent to the model
    assert not [i for i in result["lint"] if i["level"] == "error"]
    assert len(result["storyboards"]) == 3
    for sb in result["storyboards"]:
        assert sb["avatar_image"]["prompt"].startswith(result["garment_lock"])
        for c in sb["clips"]:
            assert c["prompt"].startswith(result["garment_lock"])
            assert c["params"]["generate_audio"] is False and c["params"]["aspect_ratio"] == "9:16"
            assert c["params"]["duration"] == f"{c['duration_s']}s"


def test_plan_gives_up_after_max_revisions(tmp_path):
    bad = good_plan(hook_captions=["Sexy back"])
    client = FakeClient(bad, bad, bad)
    result = planner.make_plan(SPEC, listing(tmp_path), client=client, max_revisions=2)
    assert len(client.calls) == 3
    assert any(i["level"] == "error" for i in result["lint"])  # surfaced, not hidden


def test_short_clip_models_restrict_planned_durations(monkeypatch):
    from ugc import config
    monkeypatch.setattr(config, "MAX_CLIP_S", 4)
    assert config.allowed_clip_seconds() == (4,)
    msgs = " | ".join(i.msg for i in planner.lint(SPEC, good_plan()))  # good_plan has a 6 s clip
    assert "duration must be one of (4,)" in msgs
    three_short = [ClipPlan(duration_s=4, shots=["front", "full_length"], motion_prompt="slow turn"),
                   ClipPlan(duration_s=4, shots=["back"], motion_prompt="shows the back"),
                   ClipPlan(duration_s=4, shots=["walk"], motion_prompt="walks forward")]
    ok = good_plan(storyboards=[board("a", clips=three_short), board("b", "blue hour", clips=three_short),
                                board("c", "soft overcast", clips=three_short)])
    assert not [i for i in planner.lint(SPEC, ok) if i.level == "error"]


def test_system_prompt_names_the_allowed_clip_lengths(tmp_path, monkeypatch):
    from ugc import config
    monkeypatch.setattr(config, "MAX_CLIP_S", 4)
    client = FakeClient(good_plan())
    planner.make_plan(SPEC, listing(tmp_path), client=client, max_revisions=0)
    assert "Each clip is 4 seconds" in client.calls[0]["system"]
