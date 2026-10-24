"""The incident loop end to end on MockBackend (plan.md 3.2.11, Section 4).

Done when (Step 7): the whole loop runs on MockBackend, including the combined case (code bug +
lighting shift): code verdict, fix, re-run returns vision, hand-off to the data-ladder stub.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import ladder.loop as loop_mod
from hagent.models import ScriptedModel
from ladder.loop import LoopStatus
from ladder.repair import RepairKind
from ladder.rung0 import Rung0Result
from ladder.rung1 import Rung1Outcome
from ladder.rung2 import Rung2Label, triage
from router.schemas import CodeRef, Destination, VerdictLabel
from router.stage2 import Branch
from router.router import route
from tests.ladder_cases import (BAD_PATCH, CASES, PATCHED, PATCHED_INSTRUCTION, ScriptedRepairer, combined_case,
                                condition_specific_case, loop_config, make_loop, rendering_case)
from tests.router_cases import CODE_TRAIN_RENDER

C2 = CodeRef("c2")


# --------------------------------------------------------------------------- #
# (1) the combined case: hagent once, pass_with_residual, re-run says vision, data-ladder hand-off
# --------------------------------------------------------------------------- #

def assert_combined_ending(res, repairer, ladder, code_pass):
    assert repairer.calls == 1                                              # hagent exactly once
    assert res.repairs[0].rung1.outcome is Rung1Outcome.PASS_WITH_RESIDUAL
    assert res.repairs[0].result == "rung1_pass_with_residual"
    assert [(p.label, p.row, p.destination) for p in res.passes] == [
        ("code", 4, "hagent"), ("vision_factor", 6, "data_ladder")]
    assert res.passes[1].factor == "lighting" and res.passes[1].code_key == PATCHED.key()   # re-run on the fixed system
    assert res.status is LoopStatus.DATA_HANDOFF and res.destinations == ("hagent", "data_ladder")
    [h] = ladder.handoffs                                                   # ends at the data-ladder hand-off
    assert h.factor == "lighting" and h.code_ref == PATCHED and not h.via_rung2
    assert h.nominal_scene.is_nominal and h.scene.factor_names == ("lighting",)
    assert h.failing_seeds == tuple(range(10)) and h.verdict.row == 6
    assert res.final.verdict.row == 6 and res.final_code == PATCHED


def test_combined_case_code_fix_then_vision_hand_off():
    case = combined_case()
    rep = ScriptedRepairer([PATCHED])
    loop, ladder = make_loop(case, rep)
    res = loop.run(case.incident())
    assert_combined_ending(res, rep, ladder, PATCHED)
    first = res.passes[0]
    assert first.branch == Branch.G_DEGRADED.value and first.multi_cause_flag and first.likely_second_cause
    assert not first.by_elimination                                         # located: the norm_stats contract violation
    assert res.repairs[0].rung1.target.which == "nominal"
    assert rep.requests[0].kind is RepairKind.CODE and rep.requests[0].verdict.row == 4


def test_code_is_fixed_before_vision():
    """The vision hand-off cannot happen on a pass whose verdict still routes to the repair agent."""
    case = combined_case()
    loop, ladder = make_loop(case, ScriptedRepairer([PATCHED]))
    res = loop.run(case.incident())
    assert res.passes[0].action == "repair:hagent" and res.passes[1].action == "data_handoff"
    assert len(ladder.handoffs) == 1 and ladder.handoffs[0].verdict.row in (6, 7)


# --------------------------------------------------------------------------- #
# (2) the same case with the second-cause flag suppressed ends the same way
# --------------------------------------------------------------------------- #

def test_combined_case_without_a_second_cause_flag_ends_the_same_way():
    """Severe bug: nominal reads clearly failed (Branch G), Branch C never runs, so nothing flags the
    lighting. Rung 1 keys on the branch, so the patch is still judged on the nominal scene."""
    case = combined_case(severe=True)
    rep = ScriptedRepairer([PATCHED])
    loop, ladder = make_loop(case, rep)
    res = loop.run(case.incident())
    first = res.passes[0]
    assert first.branch == Branch.G.value
    assert not first.likely_second_cause and not first.multi_cause_flag     # genuinely undetected
    assert_combined_ending(res, rep, ladder, PATCHED)


def test_combined_case_with_the_flags_forced_off_ends_the_same_way(monkeypatch):
    """Belt and braces: take the degraded combined case and clear both flags on the first verdict."""
    real_route = loop_mod.route
    calls = []

    def route_without_flags(*a, **kw):
        rr = real_route(*a, **kw)
        calls.append(rr)
        if len(calls) == 1:
            rr = replace(rr, verdict=replace(rr.verdict, likely_second_cause=False, multi_cause_flag=False))
        return rr

    monkeypatch.setattr(loop_mod, "route", route_without_flags)
    case = combined_case()
    rep = ScriptedRepairer([PATCHED])
    loop, ladder = make_loop(case, rep)
    res = loop.run(case.incident())
    assert not res.passes[0].likely_second_cause and not res.passes[0].multi_cause_flag
    assert res.passes[0].branch == Branch.G_DEGRADED.value                  # the branch is what Rung 1 reads
    assert_combined_ending(res, rep, ladder, PATCHED)


def test_a_repair_never_loops_on_a_residual_it_cannot_fix():
    """pass_with_residual goes to the router, so hagent is not called again for the lighting."""
    case = combined_case()
    rep = ScriptedRepairer([PATCHED, C2, C2, C2])
    loop, _ = make_loop(case, rep)
    assert loop.run(case.incident()).repairs.__len__() == 1 and rep.calls == 1


# --------------------------------------------------------------------------- #
# Rung 1 -> Rung 2 -> loop
# --------------------------------------------------------------------------- #

def rung2_fn(*replies):
    model = ScriptedModel(list(replies), cost_per_call=0.05)
    return (lambda inp: triage(inp, model)), model


def reply(label, ids=()):
    return json.dumps({"label": label, "reason": f"model says {label}", "evidence_ids": list(ids)})


def test_ambiguous_patch_goes_to_rung2_and_an_accepted_data_gap_reaches_the_data_ladder():
    case = condition_specific_case(0.4)             # patched glossy scene only partly recovered; nominal fine
    fn, model = rung2_fn(reply("data_gap"))
    rep = ScriptedRepairer([PATCHED_INSTRUCTION])
    loop, ladder = make_loop(case, rep, rung2=fn)
    res = loop.run(case.incident())
    assert res.repairs[0].rung1.outcome is Rung1Outcome.AMBIGUOUS and res.repairs[0].rung1.target.which == "incident"
    assert [p.row for p in res.passes] == [4, 6] and res.passes[1].factor == "material(mug)"
    assert res.passes[1].action == "rung2:data_gap"
    assert res.status is LoopStatus.DATA_HANDOFF and ladder.handoffs[0].via_rung2
    assert rep.calls == 1 and len(model.calls) == 1                         # one big-model call per incident
    assert res.big_calls == 1 and res.cost == pytest.approx(0.05)
    assert "RUNG1 AMBIGUOUS" in model.calls[0][1]["content"]                # Rung 2 reads the Rung 1 result


def test_rung2_data_gap_is_rejected_by_the_gate_when_the_rerun_has_no_restore_verdict():
    """Ambiguous patch; the router re-run on it says code by elimination (row 4), not rows 6-7."""
    case = combined_case(patched_nominal=0.4)
    fn, model = rung2_fn(reply("data_gap"))
    rep = ScriptedRepairer([PATCHED])
    loop, ladder = make_loop(case, rep, rung2=fn)
    res = loop.run(case.incident())
    assert res.repairs[0].rung1.outcome is Rung1Outcome.AMBIGUOUS
    assert res.passes[1].row == 4                                           # re-run: still code (elimination)
    assert res.rung2[0].model_label is Rung2Label.DATA_GAP and res.rung2[0].label is Rung2Label.UNCLEAR
    assert not res.rung2[0].accepted and "rows 6-7" in res.rung2[0].rejected_reason
    assert res.status is LoopStatus.UNRESOLVED and ladder.handoffs == []    # no data was ever handed over


def test_rung2_code_sends_the_incident_back_to_the_repair_agent_once_more():
    case = combined_case(patched_nominal=0.4, extra_codes=[C2])
    fn, model = rung2_fn(reply("code"))
    rep = ScriptedRepairer([PATCHED, C2])
    loop, ladder = make_loop(case, rep, rung2=fn)
    res = loop.run(case.incident())
    assert rep.calls == 2 and rep.requests[1].incident.code_ref == PATCHED   # second repair starts from the first patch
    assert any("Rung 2" in f and "model says code" in f for f in rep.requests[1].feedback)
    assert res.repairs[1].rung1.outcome is Rung1Outcome.PASS_WITH_RESIDUAL
    assert [p.row for p in res.passes] == [4, 4, 6] and res.status is LoopStatus.DATA_HANDOFF
    assert len(model.calls) == 1


def test_rung2_unclear_ends_unresolved():
    case = combined_case(patched_nominal=0.4)
    fn, _ = rung2_fn(reply("unclear"))
    loop, ladder = make_loop(case, ScriptedRepairer([PATCHED]), rung2=fn)
    res = loop.run(case.incident())
    assert res.status is LoopStatus.UNRESOLVED and "Rung 2 unclear" in res.reason and not ladder.handoffs


def test_rung2_budget_and_missing_rung2_answer_unclear_without_a_call():
    case = combined_case(patched_nominal=0.4)
    fn, model = rung2_fn(reply("code"))
    res = make_loop(case, ScriptedRepairer([PATCHED]), rung2=fn, cfg=loop_config(max_rung2_calls=0))[0].run(case.incident())
    assert res.status is LoopStatus.UNRESOLVED and model.calls == [] and res.rung2[0].reason == "Rung 2 call budget exhausted"
    res = make_loop(case, ScriptedRepairer([PATCHED]), rung2=None)[0].run(case.incident())
    assert res.status is LoopStatus.UNRESOLVED and res.rung2[0].reason == "Rung 2 not configured"


def test_ambiguous_patch_whose_rerun_is_clean_closes_without_rung2(monkeypatch):
    """The re-run finds nothing wrong at all (row 1): closed, and the big call is not spent."""
    case = condition_specific_case(0.4)
    fn, model = rung2_fn(reply("code"))
    loop, _ = make_loop(case, ScriptedRepairer([PATCHED_INSTRUCTION]), rung2=fn)
    real = loop_mod.route

    def clean(inc, backend, cfg, inputs):
        rr = real(inc, backend, cfg, inputs)
        if inc.code_ref == PATCHED_INSTRUCTION:
            rr = replace(rr, verdict=replace(rr.verdict, label=VerdictLabel.NO_SIGNIFICANT_FAILURE,
                                             destination=Destination.STOP, row=1, factor=None, replays=()))
        return rr
    monkeypatch.setattr(loop_mod, "route", clean)
    res = loop.run(case.incident())
    assert res.status is LoopStatus.CLOSED and model.calls == [] and res.passes[1].action == "close"


# --------------------------------------------------------------------------- #
# the other destinations
# --------------------------------------------------------------------------- #

def test_rendering_mismatch_is_a_config_edit_validated_on_the_incident_scene():
    case = rendering_case()
    rep = ScriptedRepairer([CODE_TRAIN_RENDER])
    loop, ladder = make_loop(case, rep)
    res = loop.run(case.incident())
    assert rep.requests[0].kind is RepairKind.CONFIG and rep.requests[0].verdict.row == 5
    r1 = res.repairs[0].rung1
    assert r1.branch is Branch.C and r1.target.which == "incident" and r1.outcome is Rung1Outcome.PASS
    assert res.status is LoopStatus.CLOSED and res.passes[1].row == 1 and not ladder.handoffs


@pytest.mark.parametrize("name,status", [("sim_asset", LoopStatus.REPORTED), ("infra_stale_server", LoopStatus.REPORTED),
                                         ("no_fault", LoopStatus.CLOSED)])
def test_report_only_and_clean_incidents_never_call_the_repair_agent(name, status):
    case = CASES[name]
    rep = ScriptedRepairer([PATCHED])
    loop, ladder = make_loop(case, rep)
    res = loop.run(case.incident())
    assert res.status is status and rep.calls == 0 and ladder.handoffs == [] and len(res.passes) == 1
    assert res.final.verdict.label is case.expected


def test_a_plain_vision_incident_goes_straight_to_the_data_ladder():
    case = CASES["vision_glossy"]
    loop, ladder = make_loop(case, ScriptedRepairer([PATCHED]))
    res = loop.run(case.incident())
    assert res.status is LoopStatus.DATA_HANDOFF and ladder.handoffs[0].factor == "material(mug)"
    assert len(res.passes) == 1 and not ladder.handoffs[0].partial


# --------------------------------------------------------------------------- #
# caps and failure paths
# --------------------------------------------------------------------------- #

def test_a_patch_that_does_not_help_is_retried_with_feedback_then_capped():
    case = combined_case(severe=True)               # BAD_PATCH leaves the nominal scene at the broken baseline
    rep = ScriptedRepairer([BAD_PATCH])
    loop, ladder = make_loop(case, rep, cfg=loop_config(max_repairs=3))
    res = loop.run(case.incident())
    assert rep.calls == 3 and [r.result for r in res.repairs] == ["rung1_fail"] * 3
    assert res.status is LoopStatus.UNRESOLVED and "repair cap (3)" in res.reason
    assert len(res.passes) == 1 and not ladder.handoffs
    assert rep.requests[0].feedback == () and len(rep.requests[1].feedback) == 1 and len(rep.requests[2].feedback) == 2
    assert "RUNG1 FAIL" in rep.requests[1].feedback[0]


def test_rung0_failures_never_reach_the_simulator():
    case = combined_case()
    rep = ScriptedRepairer([PATCHED])
    loop, ladder = make_loop(case, rep, rung0=lambda r, o: Rung0Result(False, (), "Rung 0 (tests) failed: shape"),
                             cfg=loop_config(max_repairs=2))
    res = loop.run(case.incident())
    assert [r.result for r in res.repairs] == ["rung0_failed"] * 2 and all(r.rung1 is None for r in res.repairs)
    assert not any(k.code == PATCHED.key() for k in loop.backend.cache.keys())    # no episode of the patched code
    assert "shape" in rep.requests[1].feedback[0] and res.status is LoopStatus.UNRESOLVED


def test_a_patch_that_changes_nothing_is_not_validated():
    case = combined_case()
    rep = ScriptedRepairer([case.code_ref])
    loop, _ = make_loop(case, rep, cfg=loop_config(max_repairs=2))
    res = loop.run(case.incident())
    assert [r.result for r in res.repairs] == ["no_change"] * 2 and all(r.rung1 is None for r in res.repairs)
    assert "changed nothing" in rep.requests[1].feedback[0]


def test_pass_cap_ends_the_incident_as_unresolved_with_everything_attached():
    """Evidence that never goes away (a fix that only masks the symptom): each re-run routes to code again."""
    case = combined_case(extra_codes=[C2, CodeRef("c3")], fault_persists=True)
    loop, _ = make_loop(case, ScriptedRepairer([PATCHED, C2, CodeRef("c3")]), cfg=loop_config(max_passes=2))
    res = loop.run(case.incident())
    assert len(res.passes) == 2 and res.status is LoopStatus.UNRESOLVED and "pass cap (2)" in res.reason
    assert res.final is not None and res.final.verdict.location == "obs.normalization" and len(res.repairs) == 2
    assert res.final.external_evidence == ()                 # found again in the patched code's logs
    assert res.rung2 == ()                                   # the cap came first: no big call was spent


def test_code_verdict_on_branch_c_validates_on_the_incident_scene_so_a_lighting_residual_ends_in_rung2():
    """Persistent contract evidence on the re-run makes it Branch C + code: the target is now the lit scene,
    which no code patch can recover. Not a hagent loop: the patch is ambiguous, Rung 2 answers, the
    incident ends unresolved (bounded by the pass and repair caps)."""
    case = combined_case(extra_codes=[C2], fault_persists=True)
    fn, model = rung2_fn(reply("unclear"))
    loop, _ = make_loop(case, ScriptedRepairer([PATCHED, C2]), rung2=fn)
    res = loop.run(case.incident())
    second = res.repairs[1].rung1
    assert res.passes[1].branch == "C" and second.target.which == "incident"
    assert second.outcome is Rung1Outcome.AMBIGUOUS and res.status is LoopStatus.UNRESOLVED
    assert len(model.calls) == 1 and len(res.repairs) == 2


def test_unknown_branch_is_reported_unresolved_rather_than_guessed(monkeypatch):
    real = loop_mod.route
    monkeypatch.setattr(loop_mod, "route", lambda *a, **k: replace(real(*a, **k), branch=Branch.UNKNOWN))
    case = combined_case()
    rep = ScriptedRepairer([PATCHED])
    res = make_loop(case, rep)[0].run(case.incident())
    assert res.status is LoopStatus.UNRESOLVED and "branch unknown" in res.reason and rep.calls == 0


# --------------------------------------------------------------------------- #
# cache and bookkeeping
# --------------------------------------------------------------------------- #

def test_router_and_rung1_share_one_cache():
    case = combined_case()
    loop, _ = make_loop(case, ScriptedRepairer([PATCHED]))
    be = loop.backend.backend
    res = loop.run(case.incident())
    assert loop.backend.hits > 0 and sum(be.calls.values()) == loop.backend.misses
    assert max(be.calls.values()) <= 30                                      # no config ran past the router cap
    assert res.passes[1].cache_hits > 0                                      # the re-run reused earlier episodes


def test_result_serialises_to_json():
    case = combined_case()
    loop, _ = make_loop(case, ScriptedRepairer([PATCHED]))
    d = json.loads(json.dumps(loop.run(case.incident()).to_dict()))
    assert d["status"] == "data_handoff" and d["handoff_factor"] == "lighting" and len(d["passes"]) == 2
    assert d["repairs"][0]["rung1"]["outcome"] == "pass_with_residual"


def test_planted_fault_disappears_from_the_evidence_when_the_patched_code_is_used():
    """Fix F1 (Tier 6, Step 7 fixture item): no evidence fixture. The router's own Stage 1 finds the fault in the
    original code's logs and not in the patched code's logs."""
    case = combined_case()
    inc, cfg = case.incident(), loop_config().router
    before = route(inc, case.quota_backend(), cfg, case.inputs())
    after = route(replace(inc, code_ref=PATCHED), case.quota_backend(), cfg, case.inputs())
    assert before.verdict.location == "obs.normalization" and before.external_evidence == ()
    assert any(e.location == "obs.normalization" for e in before.stage1_evidence)
    assert after.stage1_evidence == () and after.skipped == () and after.verdict.row == 6


def test_the_loop_no_longer_takes_an_evidence_provider():
    import inspect
    params = inspect.signature(loop_mod.IncidentLoop.__init__).parameters
    assert "evidence_provider" not in params and "inputs" in params
