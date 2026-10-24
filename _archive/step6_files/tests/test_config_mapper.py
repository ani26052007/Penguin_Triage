"""Config-key mapper (plan.md 3.2.12): scripted model, cached and hand-checkable output."""

import json

import pytest

from hagent.exceptions import ModelError
from hagent.models import ScriptedModel

from router.config_mapper import (MappingError, StaleMapping, config_paths, map_config_keys, spec_fields,
                                  structure_key, validate_keys)
from router.llm import LLMOutputError
from router.stage1.cfg_contract import ConfigContractCheck, ConfigMapping
from router.stage1.synthetic import CONFIG_KEYS, make_eval_config, make_incident, make_training_spec

SPEC, CFG = make_training_spec(), make_eval_config()


def reply(keys=None):
    return json.dumps({"keys": dict(CONFIG_KEYS if keys is None else keys)})


def test_paths_and_fields():
    paths = config_paths(CFG)
    assert paths["policy.chunk"] == 16 and "env" in paths and "env.cameras.front" in paths and "norm_stats" in paths
    assert not any(p.startswith("env.image_size.") for p in paths)         # lists are leaves
    assert list(spec_fields(SPEC))[0] == "action_horizon"                  # FIELD_ORDER
    assert "channel_order" not in spec_fields(SPEC)                        # not a contract field


def test_model_proposal_is_validated_written_and_returned(tmp_path):
    cache = tmp_path / "maps" / "synthetic_vla.json"
    m = ScriptedModel([reply()], name="nano")
    r = map_config_keys(SPEC, CFG, m, cache_path=cache, factor_covers={"camera_offset": "camera_poses"})
    assert r.source == "model" and r.attempts == 1 and not r.approved and r.unmapped == ()
    assert r.mapping == ConfigMapping(keys=CONFIG_KEYS, factor_covers={"camera_offset": "camera_poses"})
    doc = json.loads(cache.read_text())
    assert doc["keys"] == dict(CONFIG_KEYS) and doc["model"] == "nano" and doc["approved"] is False
    assert doc["structure_key"] == structure_key(SPEC, CFG)
    assert doc["review"]["action_horizon"] == {"path": "policy.chunk", "training_value": "16", "eval_value": "16"}
    assert not list(cache.parent.glob("*.tmp"))


def test_second_call_reads_the_file_and_does_not_call_the_model(tmp_path):
    cache = tmp_path / "m.json"
    map_config_keys(SPEC, CFG, ScriptedModel([reply()]), cache_path=cache)
    m2 = ScriptedModel([])
    r = map_config_keys(SPEC, CFG, m2, cache_path=cache)
    assert r.source == "cache" and m2.calls == [] and r.mapping.keys == CONFIG_KEYS
    assert map_config_keys(SPEC, CFG, None, cache_path=cache).source == "cache"      # no model needed


def test_cache_is_hand_checkable_hand_edits_are_honoured(tmp_path):
    cache = tmp_path / "m.json"
    map_config_keys(SPEC, CFG, ScriptedModel([reply()]), cache_path=cache)
    doc = json.loads(cache.read_text())
    doc["keys"]["denoising_steps"] = None
    doc["keys"]["replan_steps"] = "policy.denoise"                 # a (wrong) human decision stays
    doc["approved"] = True
    cache.write_text(json.dumps(doc))
    r = map_config_keys(SPEC, CFG, None, cache_path=cache)
    assert r.approved and r.mapping.keys["replan_steps"] == "policy.denoise"
    assert "denoising_steps" not in r.mapping.keys and r.unmapped == ("denoising_steps",)


def test_hand_edit_with_a_typo_fails_loudly(tmp_path):
    cache = tmp_path / "m.json"
    map_config_keys(SPEC, CFG, ScriptedModel([reply()]), cache_path=cache)
    doc = json.loads(cache.read_text())
    doc["keys"]["action_horizon"] = "policy.chunc"
    cache.write_text(json.dumps(doc))
    with pytest.raises(MappingError, match="policy.chunc"):
        map_config_keys(SPEC, CFG, None, cache_path=cache)


def test_changed_structure_makes_the_cache_stale_and_is_never_overwritten_silently(tmp_path):
    cache = tmp_path / "m.json"
    map_config_keys(SPEC, CFG, ScriptedModel([reply()]), cache_path=cache)
    doc = json.loads(cache.read_text())
    doc["approved"] = True
    cache.write_text(json.dumps(doc))
    cfg2 = json.loads(json.dumps(CFG))
    cfg2["policy"]["horizon_extra"] = 3
    m = ScriptedModel([reply()])
    with pytest.raises(StaleMapping, match="approved"):
        map_config_keys(SPEC, cfg2, m, cache_path=cache)
    assert m.calls == [] and json.loads(cache.read_text())["approved"] is True
    r = map_config_keys(SPEC, cfg2, ScriptedModel([reply()]), cache_path=cache, refresh=True)
    assert r.source == "model" and json.loads(cache.read_text())["approved"] is False


def test_changed_values_do_not_make_it_stale(tmp_path):
    cache = tmp_path / "m.json"
    map_config_keys(SPEC, CFG, ScriptedModel([reply()]), cache_path=cache)
    cfg2 = json.loads(json.dumps(CFG))
    cfg2["policy"]["chunk"] = 8
    assert map_config_keys(SPEC, cfg2, None, cache_path=cache).source == "cache"


def test_no_cache_and_no_model_is_an_error(tmp_path):
    with pytest.raises(MappingError, match="no model"):
        map_config_keys(SPEC, CFG, None, cache_path=tmp_path / "m.json")


def test_corrupt_cache_files_are_reported(tmp_path):
    cache = tmp_path / "m.json"
    cache.write_text("{not json")
    with pytest.raises(MappingError, match="not valid JSON"):
        map_config_keys(SPEC, CFG, None, cache_path=cache)
    cache.write_text('{"schema": 99}')
    with pytest.raises(MappingError, match="schema"):
        map_config_keys(SPEC, CFG, None, cache_path=cache)


# ---- bad model replies ----------------------------------------------------------

def test_hallucinated_path_is_rejected_then_retried(tmp_path):
    bad = {**CONFIG_KEYS, "action_horizon": "policy.horizon"}
    m = ScriptedModel([reply(bad), reply()])
    r = map_config_keys(SPEC, CFG, m, cache_path=tmp_path / "m.json")
    assert r.attempts == 2 and "policy.horizon" in m.calls[1][-1]["content"]


def test_hallucinated_path_twice_raises_and_writes_nothing(tmp_path):
    bad = {**CONFIG_KEYS, "action_horizon": "policy.horizon"}
    cache = tmp_path / "m.json"
    with pytest.raises(MappingError, match="no valid mapping"):
        map_config_keys(SPEC, CFG, ScriptedModel([reply(bad), reply(bad)]), cache_path=cache)
    assert not cache.exists()


def test_invalid_json_twice_raises(tmp_path):
    with pytest.raises(MappingError):
        map_config_keys(SPEC, CFG, ScriptedModel(["nope", "still nope"]), cache_path=tmp_path / "m.json")


def test_provider_error_raises_mapping_error(tmp_path):
    def boom(_):
        raise ModelError("HTTP 500")
    with pytest.raises(MappingError, match="HTTP 500"):
        map_config_keys(SPEC, CFG, ScriptedModel([boom]), cache_path=tmp_path / "m.json")


@pytest.mark.parametrize("keys, msg", [
    ({"channel_order": "env.hz"}, "not a TrainingSpec field"),
    ({"made_up": "env.hz"}, "not a TrainingSpec field"),
    ({"control_hz": "env.hz", "action_horizon": "env.hz"}, "both map"),
    ({"camera_poses": "env.cameras", "image_size": "env.cameras.front"}, "inside"),
    ({"control_hz": 5}, "does not exist"),
])
def test_validate_keys_rules(keys, msg):
    with pytest.raises(LLMOutputError, match=msg):
        validate_keys(keys, spec_fields(SPEC), config_paths(CFG))


def test_null_means_unmapped(tmp_path):
    keys = {**CONFIG_KEYS, "denoising_steps": None}
    r = map_config_keys(SPEC, CFG, ScriptedModel([reply(keys)]), cache_path=tmp_path / "m.json")
    assert r.unmapped == ("denoising_steps",) and "denoising_steps" not in r.mapping.keys


def test_extra_top_level_key_is_rejected(tmp_path):
    m = ScriptedModel([json.dumps({"keys": CONFIG_KEYS, "route": "code"}), reply()])
    assert map_config_keys(SPEC, CFG, m, cache_path=tmp_path / "m.json").attempts == 2


# ---- the mapping drives the real contract check -----------------------------------

def test_mapped_keys_drive_the_config_contract_check(tmp_path):
    r = map_config_keys(SPEC, CFG, ScriptedModel([reply()]), cache_path=tmp_path / "m.json")
    check = ConfigContractCheck(r.mapping)
    assert check.run(make_incident()) == []
    cfg = json.loads(json.dumps(CFG))
    cfg["policy"]["chunk"] = 8
    [e] = check.run(make_incident(eval_config=cfg))
    assert e.id == "s1.cfg.action_horizon" and e.location == "eval_config.policy.chunk"
