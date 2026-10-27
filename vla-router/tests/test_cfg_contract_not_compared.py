"""Router change 6: the config contract says which mapped fields it could not compare because the
training spec lacks them ("not compared: <field> (missing in spec)"), instead of skipping them silently.
A note, never a skip: the evidence and the verdict are exactly as before."""

from dataclasses import replace

import pytest

from router.router import RouterInputs, route
from router.stage1.cfg_contract import FIELD_ORDER, ConfigContractCheck, ConfigMapping
from router.stage1.registry import run_stage1
from router.stage1.synthetic import CONFIG_KEYS, make_config_mapping, make_eval_config, make_incident, make_training_spec
from tests.router_cases import CONFIG_MAPPING, default_config, router_cases


def note(name: str) -> str:
    return f"contract_cfg: not compared: {name} (missing in spec)"


def mutate(path, value):
    cfg = make_eval_config()
    node = cfg
    *parents, last = path.split(".")
    for p in parents:
        node = node[p]
    node[last] = value
    return cfg


def test_full_spec_has_no_note():
    c = ConfigContractCheck(make_config_mapping())
    inc = make_incident()
    assert c.run(inc) == [] and c.notes(inc) == []


def test_one_note_per_mapped_field_missing_in_the_spec_in_field_order():
    spec = replace(make_training_spec(), instruction_templates=None, replan_steps=None)
    c = ConfigContractCheck(make_config_mapping())
    inc = make_incident(training_spec=spec)
    assert c.run(inc) == []
    assert c.notes(inc) == [note("replan_steps"), note("instruction_templates")]


def test_every_field_can_be_noted():
    spec = replace(make_training_spec(), **{f: None for f in FIELD_ORDER})
    c = ConfigContractCheck(make_config_mapping())
    inc = make_incident(training_spec=spec)
    assert c.run(inc) == [] and c.notes(inc) == [note(f) for f in FIELD_ORDER]


def test_an_unmapped_field_missing_in_the_spec_is_not_noted():
    keys = {k: v for k, v in CONFIG_KEYS.items() if k != "replan_steps"}
    c = ConfigContractCheck(ConfigMapping(keys))
    inc = make_incident(training_spec=replace(make_training_spec(), replan_steps=None))
    c.run(inc)
    assert c.notes(inc) == []                                  # never mapped, so nothing was skipped for the spec


def test_evidence_is_unchanged_when_another_field_is_missing():
    cfg = mutate("policy.denoise", 16)
    c = ConfigContractCheck(make_config_mapping())
    before = c.run(make_incident(eval_config=cfg))
    inc = make_incident(eval_config=cfg, training_spec=replace(make_training_spec(), instruction_templates=None))
    after = c.run(inc)
    assert after == before and [e.id for e in after] == ["s1.cfg.denoising_steps"]
    assert "not compared" not in after[0].detail            # the note is in the notes, not in a field's evidence
    assert c.notes(inc) == [note("instruction_templates")]


def test_the_missing_field_is_still_not_compared():
    inc = make_incident(eval_config=mutate("policy.replan", 99),
                        training_spec=replace(make_training_spec(), replan_steps=None))
    c = ConfigContractCheck(make_config_mapping())
    assert c.run(inc) == [] and c.notes(inc) == [note("replan_steps")]


def test_notes_follow_the_last_run_of_each_incident():
    c = ConfigContractCheck(make_config_mapping())
    missing = make_incident(training_spec=replace(make_training_spec(), control_hz=None))
    c.run(missing)
    other = replace(make_incident(), incident_id="inc-other")
    c.run(other)
    assert c.notes(missing) == [note("control_hz")] and c.notes(other) == []
    c.run(make_incident())                                     # same incident id, full spec: the note goes away
    assert c.notes(missing) == []
    assert c.notes(replace(make_incident(), incident_id="never-run")) == []


def test_a_note_not_a_skip_through_run_stage1():
    inc = make_incident(training_spec=replace(make_training_spec(), camera_poses=None))
    res = run_stage1(inc, [ConfigContractCheck(make_config_mapping())])
    assert res.sources_run == ("contract_cfg",) and res.partial == () and res.evidence == ()
    assert res.notes == (note("camera_poses"),)
    assert "contract_cfg" not in {s for s, _ in res.skipped}


CASES = {c.name: c for c in router_cases()}


@pytest.mark.parametrize("name", sorted(CASES))
def test_router_verdicts_are_unchanged_and_the_gap_is_in_the_result(name):
    """Every router case through ``route()``: the verdict with the notes equals the verdict of the same
    incident with the spec-less fields unmapped (the old silent skip, which nothing could see); only the
    router result's notes differ, by exactly the not-compared lines."""
    case = CASES[name]
    inc = case.incident()
    absent = [f for f in CONFIG_MAPPING.keys if getattr(inc.training_spec, f) is None]
    present = ConfigMapping({k: v for k, v in CONFIG_MAPPING.keys.items() if k not in absent})

    def run(mapping):
        inputs = case.inputs()
        checks = tuple(ConfigContractCheck(mapping) if isinstance(c, ConfigContractCheck) else c
                       for c in inputs.stage1_checks)
        return route(case.incident(), case.quota_backend(), default_config(),
                     RouterInputs(checks, inputs.behavior_source, inputs.behavior_nominal))

    noted, silent = run(CONFIG_MAPPING), run(present)
    assert noted.verdict == silent.verdict
    assert noted.verdict.label is case.expected
    assert [n for n in noted.notes if n not in silent.notes] == [note(f) for f in absent]
