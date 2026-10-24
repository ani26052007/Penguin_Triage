"""Stage 1 check 1: config contract (plan.md 3.2.4)."""

import logging

import pytest

from router import vocab as V
from router.schemas import Factor, SceneSpec
from router.stage1.cfg_contract import ConfigContractCheck, ConfigMapping
from router.stage1.synthetic import make_config_mapping, make_eval_config, make_incident, make_training_spec

CHECK = ConfigContractCheck(make_config_mapping())


def run(cfg, scene=None, check=CHECK, **kw):
    return check.run(make_incident(eval_config=cfg, scene=scene, **kw))


def mutate(path, value):
    cfg = make_eval_config()
    node = cfg
    *parents, last = path.split(".")
    for p in parents:
        node = node[p]
    node[last] = value
    return cfg


def test_clean_config_has_no_evidence():
    assert run(make_eval_config()) == []


@pytest.mark.parametrize("path, value, field", [
    ("policy.replan", 16, "replan_steps"),                 # replan steps mismatch
    ("policy.chunk", 32, "action_horizon"),                # chunk length mismatch
    ("policy.denoise", 16, "denoising_steps"),
    ("env.hz", 10.0, "control_hz"),                        # control frequency
    ("env.image_size", [256, 256], "image_size"),
    ("norm_stats.action.mean", [0.2] + [0.0] * 7, "norm_stats"),
    ("env.cameras.front", [0.54, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], "camera_poses"),   # 4 cm shift
])
def test_planted_config_fault_is_flagged_with_its_key(path, value, field):
    found = run(mutate(path, value))
    assert [e.id for e in found] == [f"s1.cfg.{field}"]
    e = found[0]
    assert e.source == "contract_cfg" and e.signal == V.SIG_CONTRACT_VIOLATION
    assert e.location.startswith("eval_config.") and e.location.split(".", 1)[1] in path
    assert 0.5 <= e.strength <= 1.0
    assert "no scene factor declares it" in e.detail


def test_exact_mismatch_has_full_strength():
    assert run(mutate("policy.replan", 16))[0].strength > 0.999


def test_wrong_instruction_is_a_violation_and_matching_one_is_not():
    found = run(mutate("env.instruction", "grab the mug"))
    assert [e.id for e in found] == ["s1.cfg.instruction_templates"]
    assert found[0].location == "eval_config.env.instruction" and "grab the mug" in found[0].detail
    assert run(mutate("env.instruction", "pick up the red cube")) == []              # another object, same template
    assert len(run(mutate("env.instruction", {"mug": "pick up the mug", "cube": "lift the cube"}))) == 1


def test_render_settings_difference_is_not_a_contract_violation():
    found = run(mutate("env.render.samples", 4))
    assert [e.signal for e in found] == [V.SIG_RENDER_SETTINGS_DIFFER]       # row 5 evidence, not row 4
    assert found[0].location == "eval_config.env.render"                     # the mapped field; the leaf is in detail
    assert "env.render.samples: expected 64, got 4" in found[0].detail


def test_declared_factor_covers_its_field_only():
    cfg = mutate("env.cameras.front", [0.54, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])
    declared = SceneSpec("pick_mug", (Factor("camera_offset", "+4cm_x"),))
    covered = ConfigContractCheck(make_config_mapping(factor_covers={"camera_offset": "camera_poses"}))
    assert run(cfg, declared, covered) == []                                  # the declared test condition
    assert len(run(cfg, SceneSpec("pick_mug"), covered)) == 1                 # same diff, nothing declares it
    assert len(run(cfg, declared)) == 1                                       # factor not mapped: still a violation
    both = mutate("policy.replan", 16)
    both["env"]["cameras"] = cfg["env"]["cameras"]
    assert [e.id for e in run(both, declared, covered)] == ["s1.cfg.replan_steps"]   # other fields still checked


def test_numeric_tolerance_and_graded_strength():
    cfg = mutate("env.hz", 20.01)                                             # 5e-4 relative: inside rtol 1e-3
    assert run(cfg) == []
    tight = ConfigContractCheck(make_config_mapping(rtol=1e-4))
    near = run(mutate("env.hz", 20.0035), check=tight)                        # 1.75e-4 relative vs 1e-4: just outside
    assert near and 0.5 < near[0].strength < 0.9
    far = run(mutate("env.hz", 25.0), check=tight)
    assert far[0].strength > near[0].strength


def test_missing_key_or_spec_field_is_skipped_with_a_warning(caplog):
    cfg = make_eval_config()
    del cfg["env"]["hz"]
    with caplog.at_level(logging.WARNING, logger="router.stage1"):
        assert run(cfg) == []
    assert "env.hz" in caplog.text
    from dataclasses import replace
    spec = replace(make_training_spec(), replan_steps=None)
    assert run(mutate("policy.replan", 99), training_spec=spec) == []        # nothing to compare against


def test_unmapped_field_is_not_compared():
    keys = {k: v for k, v in make_config_mapping().keys.items() if k != "replan_steps"}
    assert run(mutate("policy.replan", 16), check=ConfigContractCheck(ConfigMapping(keys))) == []


def test_mapping_rejects_unknown_fields():
    with pytest.raises(ValueError):
        ConfigMapping(keys={"learning_rate": "opt.lr"})
    with pytest.raises(ValueError):
        ConfigMapping(keys={}, factor_covers={"lighting": "exposure"})
