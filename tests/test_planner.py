from types import SimpleNamespace

from tests.conftest import make_frame, write_image
from ugc import planner
from ugc.planner import ClipPlan, PromptPlan
from ugc.spec import Feature, ProductSpec

SPEC = ProductSpec(slug="red", name="red backless slip dress", colorway="deep red", color_hex="#b32d2f",
                   length="ankle-length maxi", target_duration_s=10, required_shots=["front", "back", "full_length"],
                   features=[Feature(name="back", description="deep V open back"),
                             Feature(name="neckline", description="gathered scoop neckline")])


def good_plan(**over):
    base = dict(
        setting="bright minimalist living room", avatar_image_prompt="standing front pose, soft window light, full body",
        clips=[ClipPlan(duration_s=4, shots=["front", "full_length"], motion_prompt="slow quarter turn, soft smile"),
               ClipPlan(duration_s=6, shots=["back", "walk"], motion_prompt="turns to show the back, glances over shoulder")],
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
        clips=[ClipPlan(duration_s=5, shots=["front"], motion_prompt="make the dress brighter and she says hello"),
               ClipPlan(duration_s=8, shots=["wat"], motion_prompt="walk")],
        hook_captions=["Sexy back!"])
    msgs = " | ".join(i.msg for i in planner.lint(SPEC, bad))
    for s in ("duration must be", "unknown shot tag", "required shots not covered", "change the garment",
              "speech/audio", "sexy"):
        assert s in msgs


def test_plan_revises_after_lint_errors_and_locks_garment(tmp_path):
    bad = good_plan(hook_captions=["Sexy back"])
    client = FakeClient(bad, good_plan())
    result = planner.make_plan(SPEC, listing(tmp_path), client=client)
    assert len(client.calls) == 2
    assert "Sexy back" in client.calls[1]["messages"][0]["content"][-1]["text"]  # feedback sent to the model
    assert not [i for i in result["lint"] if i["level"] == "error"]
    for c in result["clips"]:
        assert c["prompt"].startswith(result["garment_lock"])
        assert c["params"]["generate_audio"] is False and c["params"]["aspect_ratio"] == "9:16"
        assert c["params"]["duration"] == f"{c['duration_s']}s"
    assert result["avatar_image"]["prompt"].startswith(result["garment_lock"])


def test_plan_gives_up_after_max_revisions(tmp_path):
    bad = good_plan(hook_captions=["Sexy back"])
    client = FakeClient(bad, bad, bad)
    result = planner.make_plan(SPEC, listing(tmp_path), client=client, max_revisions=2)
    assert len(client.calls) == 3
    assert any(i["level"] == "error" for i in result["lint"])  # surfaced, not hidden
