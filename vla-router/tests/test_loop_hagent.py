"""The incident loop end to end with the REAL HagentRepairer (scripted hagent models) on a
patch-aware MockBackend (Fix F2a; ISSUES #6, #7, #42's ScriptedRepairer part).

Success rates and logs are keyed on the code's FILES (``tests.workspace_cases``), so whatever
name hagent's patch gets, a patch that adds the normalisation is what fixes the code bug.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hagent.models import ScriptedModel, bash
from ladder.backend import MockRule
from ladder.loop import LoopStatus
from ladder.repair import HagentRepairer
from ladder.rung1 import Rung1Outcome
from ladder.workspace import FileContains, WorkspaceManager
from router.schemas import CodeRef
from scripts.confusion_matrix import LIGHT, _REF_OK
from tests.ladder_cases import loop_config, make_loop
from tests.router_cases import TRAIN_RENDER, router_cases
from tests.test_ladder_loop import reply, rung2_fn
from tests.workspace_cases import (BLOB, BUG_LINE, EVAL_CONFIG, FIXED_LINE, STEP_A_LINE, SUBMIT, WRAPPER, files_combined_case,
                                   fixed_files, hagent_settings, make_checkout, manager, recording, replace_line,
                                   snapshot)


CASES = {c.name: c for c in router_cases()}


@pytest.fixture
def checkout(tmp_path):
    return make_checkout(tmp_path / "checkout")


@pytest.fixture
def mgr(tmp_path, checkout):
    return manager(tmp_path, checkout)


@pytest.fixture
def c0(mgr):
    """The checkout's base name (``c0@<tree12>``): the incident code of these tests."""
    return mgr.base_ref("c0")


def hagent(mgr, *script) -> tuple[HagentRepairer, ScriptedModel]:
    small = ScriptedModel(list(script))
    return HagentRepairer(mgr, hagent_settings(), mode="small", small=small), small


def test_combined_case_with_the_real_repairer_fixes_the_code_and_hands_off_lighting(mgr, checkout, tmp_path, c0):
    before = snapshot(checkout)
    case = files_combined_case(c0)
    rep, _ = hagent(mgr, replace_line(BUG_LINE, FIXED_LINE), SUBMIT)
    loop, ladder = make_loop(case, rep, workspaces=mgr)
    res = loop.run(case.incident())

    assert [(p.label, p.row, p.destination) for p in res.passes] == [
        ("code", 4, "hagent"), ("vision_factor", 6, "data_ladder")]
    assert res.repairs[0].rung1.outcome is Rung1Outcome.PASS_WITH_RESIDUAL and rep.calls == 1
    assert res.status is LoopStatus.DATA_HANDOFF and ladder.handoffs[0].factor == "lighting"
    assert res.final_code.code_hash.startswith(c0.code_hash + "+") and ladder.handoffs[0].code_ref == res.final_code
    assert res.passes[1].code_key == res.final_code.key()                    # re-run on the patched code
    assert FIXED_LINE in res.repairs[0].diff and res.final_diff == res.repairs[0].code_diff != ""

    # the fix is recoverable from LoopResult alone: a fresh store, the serialised result, the base checkout
    d = json.loads(json.dumps(res.to_dict()))
    assert d["final_diff"] == res.final_diff and d["repairs"][0]["diff"] == res.repairs[0].diff
    fresh = WorkspaceManager(tmp_path / "fresh_store", {"c0": checkout})
    final = CodeRef.from_dict(d["final_code"])
    fresh.register(final, d["final_diff"])
    assert snapshot(fresh.materialize(final)) == fixed_files(checkout)
    assert snapshot(checkout) == before                                       # never written


def test_attempt_two_starts_from_the_accepted_patch_and_a_failed_attempt_does_not_leak(mgr, checkout, c0):
    """Attempt 1 is rejected by Rung 1 (a comment changes nothing). Attempt 2 starts from c0 again and
    writes a half fix (ambiguous Rung 1, accepted as the code under repair). Rung 2 says code; attempt 3
    starts from the half fix and completes it."""
    seen: list = []
    rep, _ = hagent(mgr,
                    replace_line(BUG_LINE, BUG_LINE + "  # attempt 1"), SUBMIT,
                    bash(f"cat {WRAPPER}"), recording(seen, replace_line(BUG_LINE, STEP_A_LINE)), SUBMIT,
                    bash(f"cat {WRAPPER}"), recording(seen, replace_line(STEP_A_LINE, FIXED_LINE)), SUBMIT)
    fn, model = rung2_fn(reply("code"))
    case = files_combined_case(c0, severe=True)
    loop, ladder = make_loop(case, rep, rung2=fn, workspaces=mgr)
    res = loop.run(case.incident())

    assert [r.result for r in res.repairs] == ["rung1_fail", "rung1_ambiguous", "rung1_pass_with_residual"]
    attempt2_sees, attempt3_sees = (m[-1]["content"] for m in seen)          # what `cat` showed the agent
    assert "# attempt 1" not in attempt2_sees and BUG_LINE in attempt2_sees  # the rejected patch did not leak
    assert STEP_A_LINE in attempt3_sees and BUG_LINE not in attempt3_sees    # started from the accepted one
    r1, r2, r3 = res.repairs
    assert r1.code_before == r2.code_before == c0.key()
    assert r3.code_before == r2.code_after and STEP_A_LINE in r3.diff and BUG_LINE not in r3.diff
    assert res.status is LoopStatus.DATA_HANDOFF and len(model.calls) == 1
    assert snapshot(mgr.materialize(res.final_code)) == fixed_files(checkout)


def test_an_unrebuildable_binary_patch_gets_its_own_result_and_feedback(mgr, c0):
    seen: list = []
    rep, _ = hagent(mgr, bash(f"printf 'changed\\000' > {BLOB}"), SUBMIT,
                    recording(seen, replace_line(BUG_LINE, FIXED_LINE)), SUBMIT)
    case = files_combined_case(c0)
    loop, _ = make_loop(case, rep, workspaces=mgr, cfg=loop_config(max_repairs=2))
    res = loop.run(case.incident())
    assert [r.result for r in res.repairs] == ["unrebuildable_patch", "rung1_pass_with_residual"]
    assert res.repairs[0].rung1 is None and "Binary files" in res.repairs[0].diff
    task = next(m["content"] for m in seen[0] if m["role"] == "user")
    assert "could not be rebuilt" in task and "Do not edit or add binary files" in task
    assert res.status is LoopStatus.DATA_HANDOFF


def test_a_config_fix_by_the_real_repairer_closes_the_incident(mgr, c0):
    """ISSUES #2: with the fixture gone, the patched code's eval config comes from the patched file, so
    render_settings_differ disappears and the incident closes on the first repair, not at the repair cap."""
    base = CASES["rendering_mismatch"]
    case = replace(base, name="rendering_files", code_ref=c0,
                   rules=_REF_OK + (MockRule(0.9, code=CodeRef(c0.code_hash, {"render": TRAIN_RENDER})),
                                    MockRule(0.9, files=FileContains(EVAL_CONFIG, "train_like")),
                                    MockRule(0.15, factors_present=(LIGHT,))))
    edit = bash(f"sed -i 's/eval_fast/train_like/' {EVAL_CONFIG}", "Fixing the render preset.")
    rep, _ = hagent(mgr, edit, SUBMIT)
    loop, _ = make_loop(case, rep, workspaces=mgr)
    res = loop.run(case.incident())

    assert [(p.label, p.row, p.destination) for p in res.passes] == [
        ("rendering_mismatch", 5, "config_fix"), ("no_significant_failure", 1, "stop")]
    assert res.status is LoopStatus.CLOSED and len(res.repairs) == 1 and rep.calls == 1
    assert res.repairs[0].rung1.outcome is Rung1Outcome.PASS
    assert res.final_code.eval_config == {"render": TRAIN_RENDER}
