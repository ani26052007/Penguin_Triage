"""Fix F3: Rung 0 adapter, errors end the incident, incident budgets, pass probabilities, a pending
Rung 2 at the pass cap (ISSUES #9, #10, #11, #14). Each test fails on the pre-F3 code."""

from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

import ladder.loop as loop_mod
from hagent.exceptions import ModelError
from hagent.models import Reply, ScriptedModel, bash
from ladder.backend import RolloutBackend
from ladder.loop import LoopStatus
from ladder.repair import HagentRepairer, RepairKind, RepairOutcome, RepairRequest
from ladder.rung0 import CheckResult, Rung0Config, Rung0Result, run_rung0, rung0_hook
from ladder.rung1 import Rung1Outcome
from router.schemas import CodeRef
from tests.ladder_cases import BAD_PATCH, PATCHED, ScriptedRepairer, combined_case, loop_config, make_loop
from tests.test_ladder_loop import reply, rung2_fn
from tests.workspace_cases import (BUG_LINE, FIXED_LINE, SUBMIT, WRAPPER, files_combined_case, git, hagent_settings,
                                   make_checkout, manager, replace_line)


@pytest.fixture
def checkout(tmp_path):
    return make_checkout(tmp_path / "checkout")


@pytest.fixture
def mgr(tmp_path, checkout):
    return manager(tmp_path, checkout)


@pytest.fixture
def c0(mgr):
    return mgr.base_ref("c0")


class Recording(RolloutBackend):
    """Wraps a backend and notes the code of every simulated episode."""

    def __init__(self, inner):
        self.inner, self.codes = inner, set()

    def run(self, scene, code_ref, actor, seeds, mode):
        self.codes.add(code_ref.key())
        return self.inner.run(scene, code_ref, actor, seeds, mode)


class Reviewer:
    """A reviewer model: counts calls; raises ModelError when ``fail``."""
    name = "reviewer"

    def __init__(self, text="APPROVE\nPASS_PROBABILITY: 0.6", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def query(self, messages):
        self.calls += 1
        if self.fail:
            raise ModelError("reviewer API down")
        return Reply(self.text, 10, 5, 0.01)


class FakeEnv:
    def __init__(self, diff="--- a/x\n+++ b/x\n+y\n"):
        self.diff = diff

    def git_diff(self):
        return self.diff

    def execute(self, cmd, timeout_s=None):
        raise AssertionError("no command is configured")


def attempt(mgr, c0, old=BUG_LINE, new=FIXED_LINE, **kw) -> tuple[RepairRequest, RepairOutcome]:
    """A real attempt: the diff of an edit of the base files, and the code it names."""
    work = mgr.materialize(c0)
    p = Path(work) / WRAPPER
    p.write_text(p.read_text().replace(old, new))
    diff = git(work, "diff", "HEAD")
    case = files_combined_case(c0)
    verdict = SimpleNamespace(label=SimpleNamespace(value="code"), row=4)
    req = RepairRequest(RepairKind.CODE, case.incident(), verdict, (), None, 1)
    return req, RepairOutcome(mgr.extend(c0, diff), diff=diff, **kw)


# --- Rung 0 (#9) -------------------------------------------------------------------------------

def test_a_bad_patch_fails_rung0_on_the_rebuilt_code_and_never_reaches_the_simulator(mgr, c0):
    """Attempt 1 breaks the wrapper (a syntax error): `py_compile` fails in the rebuilt patched copy, the agent
    is told, and no simulator episode runs on that code. Attempt 2 is fine."""
    seen: list = []
    small = ScriptedModel([replace_line(BUG_LINE, "return ("), SUBMIT,
                           lambda m: seen.append(m) or bash("cat " + WRAPPER), replace_line(BUG_LINE, FIXED_LINE), SUBMIT])
    rep = HagentRepairer(mgr, hagent_settings(), mode="small", small=small)
    hook = rung0_hook(mgr, hagent_settings(), Rung0Config(test_cmd=f"python3 -m py_compile {WRAPPER}"))
    case = files_combined_case(c0)
    backend = Recording(case.quota_backend(mgr))
    before = set(Path(tempfile.gettempdir()).glob("rung0-*"))
    loop, _ = make_loop(case, rep, rung0=hook, backend=backend, workspaces=mgr)
    res = loop.run(case.incident())

    assert [r.result for r in res.repairs] == ["rung0_failed", "rung1_pass_with_residual"]
    bad = res.repairs[0]
    assert bad.rung1 is None and bad.code_after not in backend.codes            # no episode of the bad code
    assert [(c.name, c.status) for c in bad.rung0.checks] == [("diff", "pass"), ("lint", "skipped"), ("tests", "fail")]
    assert "SyntaxError" in bad.rung0.feedback and "tests" in bad.rung0.feedback
    assert res.repairs[1].rung0.passed and res.status is LoopStatus.DATA_HANDOFF
    assert set(Path(tempfile.gettempdir()).glob("rung0-*")) == before              # no copy of the patched files kept


def test_the_hook_does_not_call_the_reviewer_when_hagent_already_approved(mgr, c0):
    req, out = attempt(mgr, c0, reviews=({"verdict": "APPROVE", "pass_probability": 0.7},), pass_probabilities=(0.7,))
    rev = Reviewer()
    res = rung0_hook(mgr, hagent_settings(), Rung0Config(), reviewer=rev)(req, out)
    assert res.passed and rev.calls == 0 and res.reviewer_calls == 0
    assert res.checks[-1] == CheckResult("reviewer", "skipped", "no reviewer configured")


@pytest.mark.parametrize("reviews", [(), ({"verdict": "REJECT", "pass_probability": None},)])
def test_the_hook_runs_the_reviewer_when_hagent_did_not_approve_the_final_patch(mgr, c0, reviews):
    req, out = attempt(mgr, c0, reviews=reviews)
    rev = Reviewer()
    res = rung0_hook(mgr, hagent_settings(), Rung0Config(), reviewer=rev)(req, out)
    assert res.passed and rev.calls == 1 and res.pass_probability == 0.6 and res.reviewer_calls == 1


def test_a_reviewer_model_error_fails_open_and_is_recorded(mgr, c0):
    res = run_rung0(FakeEnv(), Rung0Config(), reviewer=Reviewer(fail=True))
    assert res.passed and res.reviewer_calls == 0
    assert res.checks[-1].name == "reviewer" and res.checks[-1].status == "error"
    assert "ModelError: reviewer API down" in res.reviewer_error
    assert res.to_dict()["reviewer_error"] == res.reviewer_error
    req, out = attempt(mgr, c0)                                                   # and through the hook
    assert rung0_hook(mgr, hagent_settings(), Rung0Config(), reviewer=Reviewer(fail=True))(req, out).passed


# --- errors end the incident (#10) -------------------------------------------------------------

class Boom(Exception):
    pass


class RaisingRepairer(ScriptedRepairer):
    def repair(self, request):
        raise Boom("repairer exploded")


def boom(*a, **kw):
    raise Boom("exploded")


def run_with_error(stage, monkeypatch):
    rep = RaisingRepairer([PATCHED]) if stage == "repairer" else ScriptedRepairer([PATCHED])
    kw = {}
    case = combined_case(patched_nominal=0.4) if stage == "rung2" else combined_case()
    if stage == "rung0":
        kw["rung0"] = boom
    if stage == "rung1":
        monkeypatch.setattr(loop_mod, "run_rung1", boom)
    if stage == "rung2":
        kw["rung2"] = boom
    loop, _ = make_loop(case, rep, **kw)
    return loop.run(case.incident())


@pytest.mark.parametrize("stage", ["repairer", "rung0", "rung1", "rung2"])
def test_an_exception_in_a_stage_ends_the_incident_unresolved_with_the_error_recorded(stage, monkeypatch):
    res = run_with_error(stage, monkeypatch)
    assert res.status is LoopStatus.UNRESOLVED
    assert (res.error_stage, res.error_type, res.error_message) == (stage, "Boom", res.error_message)
    assert "exploded" in res.error_message and f"error in {stage}: Boom" in res.reason
    assert res.passes and res.final is not None                                    # the evidence so far is attached
    if stage in ("rung0", "rung1"):
        assert res.repairs[-1].result == f"{stage}_error" and res.repairs[-1].diff   # the attempt is kept
    d = json.loads(json.dumps(res.to_dict()))
    assert d["error_stage"] == stage and d["error_type"] == "Boom" and d["status"] == "unresolved"


def test_a_model_error_from_the_repairer_is_not_special():
    class Missing(ScriptedRepairer):
        def repair(self, request):
            raise ModelError("no model configured")
    case = combined_case()
    res = make_loop(case, Missing([PATCHED]))[0].run(case.incident())
    assert res.status is LoopStatus.UNRESOLVED and res.error_type == "ModelError"


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_keyboard_interrupt_and_system_exit_still_propagate(exc):
    class Stop(ScriptedRepairer):
        def repair(self, request):
            raise exc()
    case = combined_case()
    with pytest.raises(exc):
        make_loop(case, Stop([PATCHED]))[0].run(case.incident())


# --- budgets (#11) -----------------------------------------------------------------------------

class CostlyRepairer(ScriptedRepairer):
    """Each attempt costs ``cost`` and ``big`` big calls (and is stopped by a failing Rung 0)."""

    def __init__(self, codes, cost=1.0, big=2):
        super().__init__(codes)
        self.cost, self.big = cost, big

    def repair(self, request):
        out = super().repair(request)
        return RepairOutcome(out.code_ref, out.diff, out.notes, self.cost, self.big, out.code_diff)


def failing_rung0(req, out):
    return Rung0Result(False, (), "Rung 0 (tests) failed")


def test_the_cost_cap_ends_the_incident_before_the_next_attempt():
    case = combined_case()
    rep = CostlyRepairer([PATCHED])
    res = make_loop(case, rep, rung0=failing_rung0, cfg=loop_config(max_repairs=10, max_cost=2.0))[0].run(case.incident())
    assert rep.calls == 2 and res.cost == 2.0
    assert res.status is LoopStatus.UNRESOLVED and "cost cap ($2.0) reached" in res.reason


def test_the_big_call_cap_ends_the_incident_before_the_next_attempt():
    case = combined_case()
    rep = CostlyRepairer([PATCHED])
    res = make_loop(case, rep, rung0=failing_rung0, cfg=loop_config(max_repairs=10, max_big_calls=3))[0].run(case.incident())
    assert rep.calls == 2 and res.big_calls == 4
    assert res.status is LoopStatus.UNRESOLVED and "big-call cap (3) reached" in res.reason


def test_rung2_counts_toward_the_caps_and_is_not_called_once_one_is_hit():
    case = combined_case(patched_nominal=0.4)                    # the patch is ambiguous: Rung 2 is next
    fn, model = rung2_fn(reply("code"))
    res = make_loop(case, CostlyRepairer([PATCHED], cost=1.0, big=1), rung2=fn,
                    cfg=loop_config(max_big_calls=1))[0].run(case.incident())
    assert model.calls == [] and res.rung2 == () and res.passes[1].action == "budget"
    assert res.status is LoopStatus.UNRESOLVED and "big-call cap (1)" in res.reason and "Rung 2 not called" in res.reason
    # and its own spend is counted: with room for one more call it runs, and the cap then stops a further repair
    fn, model = rung2_fn(reply("code"))
    res = make_loop(case, CostlyRepairer([PATCHED], cost=0.0, big=0), rung2=fn,
                    cfg=loop_config(max_big_calls=1, max_repairs=5))[0].run(case.incident())
    assert len(model.calls) == 1 and res.big_calls == 1
    assert res.status is LoopStatus.UNRESOLVED and "big-call cap (1) reached: 1 calls" in res.reason


def test_caps_off_by_default():
    case = combined_case()
    rep = CostlyRepairer([PATCHED], cost=100.0, big=100)
    res = make_loop(case, rep)[0].run(case.incident())
    assert res.status is LoopStatus.DATA_HANDOFF and "cap" not in res.reason


def test_a_cost_cap_with_unpriced_models_warns(mgr, c0):
    rep = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([]))
    case = files_combined_case(c0)
    loop, _ = make_loop(case, rep, workspaces=mgr, cfg=loop_config(max_cost=1.0))
    assert len(loop.warnings) == 1 and "cannot trigger" in loop.warnings[0] and "small" in loop.warnings[0]
    assert make_loop(case, rep, workspaces=mgr)[0].warnings == ()                 # no cap: nothing to warn about
    other = combined_case()                                                      # a repairer with no prices to check
    assert make_loop(other, ScriptedRepairer([PATCHED]), cfg=loop_config(max_cost=1.0))[0].warnings == ()
    rep.settings.small.price_in_per_mtok = 1.0
    assert make_loop(case, rep, workspaces=mgr, cfg=loop_config(max_cost=1.0))[0].warnings == ()


def test_the_warning_reaches_the_result(mgr, c0):
    rep = HagentRepairer(mgr, hagent_settings(), mode="small",
                         small=ScriptedModel([replace_line(BUG_LINE, FIXED_LINE), SUBMIT]))
    case = files_combined_case(c0)
    loop, _ = make_loop(case, rep, workspaces=mgr, cfg=loop_config(max_cost=1.0))
    res = loop.run(case.incident())
    assert res.warnings == loop.warnings and res.to_dict()["warnings"] == list(loop.warnings) != []


# --- pass probabilities (#14) ------------------------------------------------------------------

def test_hagent_review_pass_probabilities_reach_the_result(mgr, c0):
    small = ScriptedModel([replace_line(BUG_LINE, "return (") , SUBMIT,                # reviewed, rejected
                           replace_line("return (", FIXED_LINE), SUBMIT])
    big = ScriptedModel(["REJECT: the edit is not valid python\nPASS_PROBABILITY: 0.1",
                         "APPROVE\nPASS_PROBABILITY: 0.85"])
    rep = HagentRepairer(mgr, hagent_settings(final_review=True), mode="hybrid", small=small, big=big)
    rev = Reviewer()
    case = files_combined_case(c0)
    loop, _ = make_loop(case, rep, workspaces=mgr, rung0=rung0_hook(mgr, hagent_settings(), Rung0Config(), reviewer=rev))
    res = loop.run(case.incident())

    r = res.repairs[0]
    assert r.pass_probabilities == (0.1, 0.85) and [x["verdict"] for x in r.reviews] == ["REJECT", "APPROVE"]
    assert rev.calls == 0 and r.rung0.reviewer_calls == 0                            # hagent's review was not repeated
    assert res.big_calls == 2 and res.repairs[0].result == "rung1_pass_with_residual"
    d = json.loads(json.dumps(res.to_dict()))
    assert d["repairs"][0]["pass_probabilities"] == [0.1, 0.85] and len(d["repairs"][0]["reviews"]) == 2


def test_attempts_without_a_review_have_empty_pass_probabilities():
    case = combined_case()
    res = make_loop(case, ScriptedRepairer([PATCHED]))[0].run(case.incident())
    assert res.repairs[0].pass_probabilities == () and res.to_dict()["repairs"][0]["pass_probabilities"] == []


# --- a pending Rung 2 at the pass cap ----------------------------------------------------------

def test_a_pending_rung2_at_the_pass_cap_is_recorded():
    case = combined_case(patched_nominal=0.4)
    fn, model = rung2_fn(reply("code"))
    res = make_loop(case, ScriptedRepairer([PATCHED]), rung2=fn, cfg=loop_config(max_passes=1))[0].run(case.incident())
    assert res.repairs[0].rung1.outcome is Rung1Outcome.AMBIGUOUS and len(res.passes) == 1
    assert res.status is LoopStatus.UNRESOLVED and res.rung2_pending and model.calls == []
    assert "pass cap (1)" in res.reason and "Rung 2 call was pending" in res.reason
    assert res.to_dict()["rung2_pending"] is True


def test_the_flag_is_off_for_an_ordinary_pass_cap():
    case = combined_case(extra_codes=[CodeRef("c2")], fault_persists=True)
    res = make_loop(case, ScriptedRepairer([PATCHED, CodeRef("c2")]), cfg=loop_config(max_passes=1))[0].run(case.incident())
    assert "pass cap (1)" in res.reason and not res.rung2_pending
