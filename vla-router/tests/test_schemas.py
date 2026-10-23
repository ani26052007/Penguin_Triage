import json

import pytest

from router.schemas import (
    ActionPipeline, ActorKind, ActorRef, CodeRef, Destination, EpisodeResult, Evidence, Factor,
    Incident, Mode, Outcome, ReplayResult, SceneSpec, TrainingSpec, Verdict, VerdictLabel,
)


def _json_roundtrip(obj):
    return type(obj).from_dict(json.loads(json.dumps(obj.to_dict())))


SCENE = SceneSpec("pick_mug", (Factor("lighting", "dark_hdr_03"), Factor("material(mug)", "glossy_02")))
TSPEC = TrainingSpec(
    policy_name="vla-under-test",
    image_size=(224, 224), channel_order="RGB",
    norm_stats={"state": {"mean": [0.0, 1.0], "std": [1.0, 2.0]}},
    camera_poses={"front": [0, 0, 1, 0, 0, 0, 1]},
    action_horizon=16, replan_steps=8, denoising_steps=4, control_hz=20.0,
    render_settings={"spp": 1}, state_layout={"arm": [0, 7]}, action_layout={"gripper": [7, 8]},
)
CODE = CodeRef("abc123", {"replan_steps": 8, "render": {"spp": 1}})
POLICY = ActorRef(ActorKind.POLICY, "ckpt-001")
REF_RAW = ActorRef(ActorKind.REFERENCE, "ik-v1", ActionPipeline.RAW)


def test_scene_spec_is_order_independent_and_derives_restores():
    swapped = SceneSpec("pick_mug", (Factor("material(mug)", "glossy_02"), Factor("lighting", "dark_hdr_03")))
    assert swapped == SCENE and swapped.key() == SCENE.key()
    one = SCENE.restore("lighting")
    assert one.factor_names == ("material(mug)",)
    assert SCENE.factor_names == ("lighting", "material(mug)")       # original untouched
    assert SCENE.restore_all().is_nominal
    assert SCENE.with_factor("lighting", "glare_01").factor("lighting") == "glare_01"
    with pytest.raises(KeyError):
        SCENE.restore("camera_offset")


def test_scene_spec_rejects_duplicates_and_empty_nominal():
    with pytest.raises(ValueError):
        SceneSpec("s", (Factor("lighting", "a"), Factor("lighting", "b")))
    with pytest.raises(ValueError):
        SceneSpec("", ())


def test_code_ref_key_includes_eval_config():
    assert CODE.key() != CodeRef("abc123", {"replan_steps": 16, "render": {"spp": 1}}).key()
    # key is independent of dict ordering
    assert CODE.key() == CodeRef("abc123", {"render": {"spp": 1}, "replan_steps": 8}).key()


def test_actor_ref_rules():
    assert POLICY.key() != ActorRef(ActorKind.REFERENCE, "ckpt-001").key()
    assert REF_RAW.key() != ActorRef(ActorKind.REFERENCE, "ik-v1", ActionPipeline.WRAPPER).key()
    with pytest.raises(ValueError):
        ActorRef(ActorKind.POLICY, "ckpt-001", ActionPipeline.RAW)
    with pytest.raises(ValueError):
        ActorRef(ActorKind.REFERENCE, "")


def test_evidence_strength_bounds():
    Evidence("e1", "contract_cfg", "replan_mismatch", 1.0, "eval.replan_steps", "8 vs 16")
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            Evidence("e1", "x", "y", bad)
    with pytest.raises(ValueError):
        Evidence("", "x", "y", 0.5)


def test_verdict_row_bounds():
    with pytest.raises(ValueError):
        Verdict(VerdictLabel.CODE, Destination.HAGENT, row=11)


def test_incident_requires_policy_actor():
    with pytest.raises(ValueError):
        Incident("i1", SCENE, TSPEC, CODE, REF_RAW, (1, 2, 3))


def test_all_schemas_roundtrip_through_json():
    ep = EpisodeResult(3, True, {"subtask": {"reach": True, "grasp": False}})
    replay = ReplayResult("C2", SCENE.restore("lighting"), CODE, REF_RAW, Mode.CLEAN, (ep,),
                          Outcome.PARTIAL, 0.31, 0.62)
    objs = [
        SCENE, SCENE.restore_all(), TSPEC, TrainingSpec("minimal"), CODE, POLICY, REF_RAW, ep, replay,
        Evidence("e7", "staleness", "variable_delay", 0.8, None, "delay grows with load"),
        Incident("i1", SCENE, TSPEC, CODE, POLICY, (4, 5, 6), Mode.INCIDENT, 0.25, 0.8, {"video": "s3://x"}),
        Verdict(VerdictLabel.VISION_FACTOR, Destination.DATA_LADDER, 6, ("e7",), factor="lighting",
                likely_second_cause=True, replays=(replay,)),
    ]
    for obj in objs:
        back = _json_roundtrip(obj)
        assert back.to_dict() == obj.to_dict(), type(obj).__name__


def test_replay_result_helpers_and_duplicate_seeds():
    eps = tuple(EpisodeResult(s, s % 2 == 0) for s in range(10))
    r = ReplayResult("S0", SCENE, CODE, POLICY, Mode.CLEAN, eps)
    assert (r.n, r.successes, r.rate) == (10, 5, 0.5)
    assert r.by_seed()[4] is True
    with pytest.raises(ValueError):
        ReplayResult("S0", SCENE, CODE, POLICY, Mode.CLEAN, (eps[0], eps[0]))


def test_training_spec_rejects_unknown_fields():
    with pytest.raises(ValueError):
        TrainingSpec.from_dict({"policy_name": "p", "made_up": 1})
