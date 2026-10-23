"""Stage 1 check 3: action contract (plan.md 3.2.4)."""

import pytest

from router import vocab as V
from router.prereq.trace import TraceUnavailable
from router.schemas import Mode
from router.stage1.action_contract import ActionContractCheck
from router.stage1.nominal import MissingNominal, NominalStats
from router.stage1.synthetic import Faults, make_incident, make_nominal, make_source

NOMINAL = make_nominal()
INCIDENT = make_incident()


def run(faults=Faults(), **kw):
    return ActionContractCheck(NOMINAL, make_source(faults), **kw).run(INCIDENT)


def ids(found):
    return sorted(e.id for e in found)


def test_clean_traces_have_no_evidence():
    assert run() == []


def test_gripper_sign_inversion_is_found_and_nothing_else():
    found = run(Faults(gripper_inverted=True))
    assert ids(found) == ["s1.action.gripper_sign"] or "s1.action.gripper_sign" in ids(found)
    e = next(e for e in found if e.id == "s1.action.gripper_sign")
    assert e.source == "contract_action" and e.signal == V.SIG_CONTRACT_VIOLATION
    assert e.location == "action.gripper_sign" and e.strength > 0.9
    assert "s1.action.range" not in ids(found) and "s1.action.dims" not in ids(found)


def test_stuck_gripper_command_is_found():
    found = run(Faults(gripper_stuck=True))
    assert any(i.startswith("s1.action.gripper") for i in ids(found))
    assert all(e.location.startswith("action.") for e in found)


def test_out_of_range_actions_are_found():
    found = run(Faults(action_scale=3.0))
    e = next(e for e in found if e.id == "s1.action.range")
    assert e.location == "action.range" and e.strength > 0.9
    assert "s1.action.gripper_sign" not in ids(found)


def test_wrong_dimension_count_is_exact_and_full_strength():
    found = run(Faults(drop_dim=True))
    e = next(e for e in found if e.id == "s1.action.dims")
    assert e.strength == 1.0 and e.location == "action.dims" and "[7]" in e.detail and "8" in e.detail


def test_degraded_tracking_points_at_the_controller():
    found = run(Faults(gain=0.15))                      # sluggish joint controller
    e = next(e for e in found if e.id == "s1.action.joint_tracking")
    assert e.location == "action.joint_tracking" and e.strength > 0.5
    assert "s1.action.gripper_sign" not in ids(found) and "s1.action.range" not in ids(found)


def test_healthy_gain_variation_does_not_fire():
    assert run(Faults(gain=0.55)) == []


def test_reads_the_requested_mode_and_raises_without_traces():
    src = make_source(clean=Faults(), incident=Faults(gripper_inverted=True))
    assert ActionContractCheck(NOMINAL, src, mode=Mode.CLEAN).run(INCIDENT) == []
    assert ActionContractCheck(NOMINAL, src, mode=Mode.INCIDENT).run(INCIDENT) != []
    with pytest.raises(TraceUnavailable):
        ActionContractCheck(NOMINAL, src).run(make_incident(log_refs={}))


def test_missing_nominal_fails_at_construction():
    with pytest.raises(MissingNominal):
        ActionContractCheck(NominalStats({}), make_source())


def test_unusable_training_spec_warns_and_skips_instead_of_guessing(caplog):
    from dataclasses import replace
    spec = replace(make_incident().training_spec, norm_stats={})
    with caplog.at_level("WARNING", logger="router.stage1"):
        found = ActionContractCheck(NOMINAL, make_source(Faults(action_scale=3.0))).run(make_incident(training_spec=spec))
    assert "s1.action.range" not in ids(found)
    assert "range not checked" in caplog.text
