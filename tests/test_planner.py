from types import SimpleNamespace

from tests.conftest import make_frame, write_image
from ugc import planner
from ugc.planner import ClipPlan, PromptPlan, Storyboard
from ugc.spec import Feature, ProductSpec

SPEC = ProductSpec(slug="red", name="red backless slip dress", colorway="deep red", color_hex="#b32d2f",
                   length="ankle-length maxi", target_duration_s=10, required_shots=["front", "back", "full_length"],
                   features=[Feature(name="back", description="deep V open back"),
                             Feature(name="neckline", description="gathered scoop neckline")])


STAND = "standing facing the camera on the path, relaxed"
HIP = "standing facing the camera, one hand on the hip"
BACK = "back to the camera, glancing over her shoulder"


def clip(duration, shots, action, camera, start, end, motion="slow, gentle movement"):
    return ClipPlan(duration_s=duration, shots=shots, action=action, camera=camera, start_state=start,
                    end_state=end, motion_prompt=motion)


def clips():
    return [clip(4, ["front", "full_length"], "pose_hold", "tilt_up", STAND, HIP),
            clip(6, ["back", "walk"], "turn_over_shoulder", "slow_pull_back", HIP, BACK)]


def board(location, tod="golden hour", **over):
    base = dict(title=f"Shoot at {location}", location=location, time_of_day=tod, mood="warm and relaxed",
                avatar_image_prompt="standing front pose, soft natural light, full body", clips=clips())
    base["arc"] = "settle, hand on hip, turn to show the back"
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
            clip(5, ["front"], "pose_hold", "static", STAND, HIP, "make the dress brighter and she says hello"),
            clip(8, ["wat"], "turn_over_shoulder", "static", HIP, BACK, "walk")])],
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
    three_short = [clip(4, ["front", "full_length"], "pose_hold", "tilt_up", STAND, HIP),
                   clip(4, ["back"], "turn_over_shoulder", "static", HIP, BACK),
                   clip(4, ["side"], "look_up_smile", "slow_push_in", BACK, "looking up, smiling, three-quarter view")]
    ok = good_plan(storyboards=[board("a", clips=three_short), board("b", "blue hour", clips=three_short),
                                board("c", "soft overcast", clips=three_short)])
    assert not [i for i in planner.lint(SPEC, ok) if i.level == "error"]


def test_system_prompt_names_the_allowed_clip_lengths(tmp_path, monkeypatch):
    from ugc import config
    monkeypatch.setattr(config, "MAX_CLIP_S", 4)
    client = FakeClient(good_plan())
    planner.make_plan(SPEC, listing(tmp_path), client=client, max_revisions=0)
    assert "Each clip is 4 seconds" in client.calls[0]["system"]


# ---------- variety and continuity ----------
def errors(plan):
    return " | ".join(i.msg for i in planner.lint(SPEC, plan) if i.level == "error")


def three(actions, cams=("static", "slow_push_in", "tilt_up"), states=None, durations=(4, 4, 4)):
    st = states or [STAND, HIP, BACK, "looking up, smiling"]
    shots = (["front", "full_length"], ["back"], ["side"])
    return [clip(durations[i], shots[i], actions[i], cams[i], st[i], st[i + 1]) for i in range(3)]


def one_board_plan(cl):
    return good_plan(storyboards=[board("a", clips=cl), board("b", "blue hour", clips=cl),
                                  board("c", "soft overcast", clips=cl)])


def test_a_storyboard_must_not_be_just_walking():
    assert "'walk' may be used at most once" in errors(one_board_plan(three(["walk", "walk", "pose_hold"])))
    only_two = one_board_plan(three(["pose_hold", "hand_on_hip", "pose_hold"]))
    assert "at least 3 different actions" in errors(only_two)


def test_consecutive_clips_may_not_repeat_an_action():
    assert "repeat the action 'pose_hold'" in errors(one_board_plan(three(["pose_hold", "pose_hold", "lean"])))


def test_unknown_actions_and_cameras_are_rejected():
    msgs = errors(one_board_plan(three(["moonwalk", "lean", "hair_touch"], cams=("spin", "static", "tilt_up"))))
    assert "unknown action 'moonwalk'" in msgs and "unknown camera 'spin'" in msgs


def test_each_clip_must_start_where_the_previous_one_ended():
    broken = three(["pose_hold", "lean", "hair_touch"], states=[STAND, HIP, "sitting on a bench", "smiling"])
    broken[2] = clip(4, ["side"], "hair_touch", "tilt_up", "a different place entirely", "smiling")
    assert "clip 3 start_state must repeat clip 2 end_state word for word" in errors(one_board_plan(broken))


def test_matching_states_ignore_case_spacing_and_a_trailing_period():
    cl = three(["pose_hold", "lean", "hair_touch"])
    cl[1] = clip(4, ["back"], "lean", "static", "  Standing facing the camera,  one hand on the hip.  ", BACK)
    assert "word for word" not in errors(one_board_plan(cl))


def test_a_good_varied_chain_passes():
    assert not errors(one_board_plan(three(["pose_hold", "turn_over_shoulder", "look_up_smile"])))


def test_every_clip_prompt_carries_the_hand_off_and_the_lock(tmp_path):
    result = planner.make_plan(SPEC, listing(tmp_path), client=FakeClient(good_plan()))
    sb = result["storyboards"][0]
    first, second = sb["clips"]
    assert sb["arc"] and first["chained"] is False and second["chained"] is True
    assert first["prompt"].startswith(result["garment_lock"])
    assert f"The first frame is the exact starting state: {STAND}." in first["prompt"]
    assert f"End in this state: {HIP}." in first["prompt"]
    assert "starting state: " + HIP in second["prompt"] and "One continuous take, no cuts" in second["prompt"]
    assert "Camera: starts low on the hem" in first["prompt"]  # the camera move is spelled out, not just named
    assert first["action"] == "pose_hold" and second["action"] == "turn_over_shoulder"


def test_the_prompt_demands_continuity_and_variety_and_lists_the_libraries(tmp_path):
    client = FakeClient(good_plan())
    planner.make_plan(SPEC, listing(tmp_path), client=client, max_revisions=0)
    system = " ".join(client.calls[0]["system"].split())  # ignore line wrapping
    for needle in ("WORD FOR WORD", "LAST FRAME", "must not be just walking", "at most once", "twirl", "tilt_up",
                   "no fast spins"):
        assert needle in system, needle


def test_longer_clip_lengths_can_be_allowed_for_longer_clip_models(monkeypatch):
    import importlib
    from ugc import config
    monkeypatch.setenv("UGC_CLIP_SECONDS", "5, 8,10,12")
    importlib.reload(config)
    assert config.CLIP_SECONDS == (5, 8, 10, 12) and config.allowed_clip_seconds() == (5, 8)  # MAX_CLIP_S default 8
    monkeypatch.setenv("UGC_MAX_CLIP_S", "30")
    importlib.reload(config)
    assert config.allowed_clip_seconds() == (5, 8, 10, 12)
