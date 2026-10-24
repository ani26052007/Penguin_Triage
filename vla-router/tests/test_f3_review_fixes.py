"""F2b/F3 review fixes (PLAN_F3_fixes.md). Each test fails on the code before the fixes."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

import ladder.loop as loop_mod
from hagent.models import ScriptedModel, bash
from ladder.loop import IncidentLoop, LoopStatus, rebuild_feedback
from ladder.repair import HagentRepairer, RepairKind
from ladder.rung0 import Rung0Config
from ladder.rung2 import Rung2Label
from ladder.workspace import UnrebuildablePatch, WorkspaceError, WorkspaceManager
from router.decide import DESTINATION_BY_ROW, LABEL_BY_ROW
from router.schemas import CodeRef
from tests.ladder_cases import PATCHED, ScriptedRepairer, combined_case, loop_config, make_loop
from tests.test_f3 import Boom, CostlyRepairer, Recording, attempt, failing_rung0
from tests.test_ladder_loop import reply, rung2_fn
from tests.test_repair import request
from tests.workspace_cases import (BUG_LINE, EVAL_CONFIG, FIXED_LINE, STEP_A_LINE, SUBMIT, WRAPPER, files_combined_case, git,
                                   hagent_settings, make_checkout, manager, replace_line)

FIX = replace_line(BUG_LINE, FIXED_LINE)
BREAK_CONFIG = bash(f"printf 'render: [' > {EVAL_CONFIG}", "Editing the config.")


@pytest.fixture
def checkout(tmp_path):
    return make_checkout(tmp_path / "checkout")


@pytest.fixture
def mgr(tmp_path, checkout):
    return manager(tmp_path, checkout)


@pytest.fixture
def c0(mgr):
    return mgr.base_ref("c0")


# --- 1. budget vs the router's destination ----------------------------------------------------------------

def run_forced(monkeypatch, row: int, limit: str):
    """An ambiguous patch whose router re-run is forced to ``row``; the limit ``budget`` (big-call cap hit by the
    repair) or ``calls`` (max_rung2_calls = 0) is already reached when Rung 2 is due."""
    case = combined_case(patched_nominal=0.4)
    fn, model = rung2_fn(reply("code"))
    if limit == "budget":
        rep, cfg = CostlyRepairer([PATCHED], cost=1.0, big=1), loop_config(max_big_calls=1, max_repairs=2)
    else:
        rep, cfg = ScriptedRepairer([PATCHED]), loop_config(max_rung2_calls=0, max_repairs=2)
    loop, ladder = make_loop(case, rep, rung2=fn, cfg=cfg)
    real = loop_mod.route

    def forced(inc, backend, cfg, inputs):
        rr = real(inc, backend, cfg, inputs)
        if inc.code_ref == PATCHED:
            v = replace(rr.verdict, label=LABEL_BY_ROW[row], destination=DESTINATION_BY_ROW[row], row=row,
                        factor="lighting" if row in (6, 7) else None, replays=())
            rr = replace(rr, verdict=v)
        return rr
    monkeypatch.setattr(loop_mod, "route", forced)
    return loop.run(case.incident()), rep, ladder, model


@pytest.mark.parametrize("row,status", [(6, LoopStatus.DATA_HANDOFF), (7, LoopStatus.DATA_HANDOFF),
                                        (2, LoopStatus.REPORTED), (4, LoopStatus.UNRESOLVED)])
def test_the_two_kinds_of_limit_give_the_same_destination(monkeypatch, row, status):
    """A hit big-call cap used to end a row 6/2 re-run UNRESOLVED; max_rung2_calls already followed the router."""
    by_budget = run_forced(monkeypatch, row, "budget")
    by_calls = run_forced(monkeypatch, row, "calls")
    for res, rep, ladder, model in (by_budget, by_calls):
        assert res.status is status and model.calls == []                      # no Rung 2 call either way
        assert bool(ladder.handoffs) == (status is LoopStatus.DATA_HANDOFF)
    assert by_budget[0].reason.split(":")[0] == by_calls[0].reason.split(":")[0] or row == 4
    if row != 4:                                                              # the label was not obtained, and it says why
        assert "label not obtained: budget" in by_budget[0].rung2[0].reason
        assert by_budget[0].rung2[0].source == "fallback" and by_budget[0].passes[1].row == row


def test_a_cap_hit_still_blocks_a_repair_attempt_and_a_row4_label(monkeypatch):
    res, rep, _, model = run_forced(monkeypatch, 4, "budget")
    assert rep.calls == 1 and model.calls == [] and "big-call cap (1) reached" in res.reason
    assert res.passes[1].action == "budget"


# --- 2. an unreadable eval config of the patched copy is an unrebuildable patch ----------------------------

def test_extend_turns_an_unreadable_patched_config_into_an_unrebuildable_patch(mgr, c0):
    work = mgr.materialize(c0)
    (work / EVAL_CONFIG).write_text("render: [")
    patch = git(work, "diff", "HEAD")
    with pytest.raises(UnrebuildablePatch) as e:
        mgr.extend(c0, patch)
    assert e.value.cause == "config_unreadable" and "eval config" in str(e.value)
    assert "eval config" in rebuild_feedback(1, "config_unreadable", str(e.value))        # advice, not the generic one
    (work / EVAL_CONFIG).unlink()
    with pytest.raises(UnrebuildablePatch) as e:
        mgr.extend(c0, git(work, "diff", "HEAD"))
    assert e.value.cause == "config_unreadable"


@pytest.mark.parametrize("where", ["local", "extend"])
def test_a_patch_that_breaks_the_config_is_recorded_costed_and_the_agent_retries(mgr, c0, monkeypatch, where):
    """``local``: the check of the agent's own copy refuses it. ``extend``: (docker-like, no host copy)
    the rebuild refuses it. Either way the attempt is recorded, its cost kept, the agent told, and attempt 2 runs."""
    if where == "extend":
        monkeypatch.setattr(HagentRepairer, "_local_name", lambda self, work: None)
    seen: list = []
    small = ScriptedModel([BREAK_CONFIG, SUBMIT, lambda m: seen.append(m) or FIX, SUBMIT], cost_per_call=0.05)
    rep = HagentRepairer(mgr, hagent_settings(), mode="small", small=small)
    case = files_combined_case(c0)
    res = make_loop(case, rep, workspaces=mgr)[0].run(case.incident())
    assert [r.result for r in res.repairs] == ["unrebuildable_patch", "rung1_pass_with_residual"]
    assert res.error_stage == "" and res.status is LoopStatus.DATA_HANDOFF
    bad = res.repairs[0]
    assert bad.code_after == bad.code_before and "printf" not in bad.diff and "render: [" in bad.diff
    assert res.cost == pytest.approx(0.05 * len(small.calls)) and res.cost > 0.05 * 2
    prompt = next(m["content"] for m in seen[0] if m["role"] == "user")
    assert "eval config" in prompt and "could not be rebuilt" in prompt


# --- 3. nothing escapes IncidentLoop.run ---------------------------------------------------------------------

def corrupt_store_code(mgr, c0):
    _, out = attempt(mgr, c0)
    f = mgr.store.root / "patches" / f"{out.code_ref.code_hash}.patch"
    f.write_bytes(b"garbage")
    return out.code_ref


@pytest.mark.parametrize("which", ["bare", "unknown_label", "old_recipe", "corrupt_store"])
def test_setup_errors_end_the_incident_unresolved_instead_of_raising(mgr, c0, which):
    code = {"bare": CodeRef("c0"), "unknown_label": CodeRef("nope@0123456789ab"),
            "old_recipe": CodeRef("c0+0123456789ab")}.get(which) or corrupt_store_code(mgr, c0)
    case = files_combined_case(c0)
    rep = HagentRepairer(mgr, hagent_settings(), mode="small")           # its workspaces name the incident code
    loop, _ = make_loop(case, rep, workspaces=mgr)
    res = loop.run(replace(case.incident(), code_ref=code))
    assert res.status is LoopStatus.UNRESOLVED and res.error_stage == "setup" and res.error_type == "WorkspaceError"
    assert "error in setup" in res.reason and res.passes == () and rep.calls == 0
    assert json.loads(json.dumps(res.to_dict()))["error_stage"] == "setup"


# --- 8. handoff errors have their own stage ---------------------------------------------------------------------

def test_a_failing_data_ladder_is_stage_handoff():
    class Broken:
        def handoff(self, h):
            raise Boom("data ladder down")
    case = combined_case()
    res = make_loop(case, ScriptedRepairer([PATCHED]), ladder=Broken())[0].run(case.incident())
    assert res.status is LoopStatus.UNRESOLVED and res.error_stage == "handoff" and "data ladder down" in res.error_message


# --- 5. _local_name uses the manager's loader; an unreadable config refuses the attempt --------------------------------

def test_local_name_reads_the_config_with_the_managers_loader_and_refuses_an_unreadable_one(tmp_path, checkout):
    calls = []

    def loader(workdir):
        calls.append(workdir)
        if (Path(workdir) / "BROKEN").exists():
            raise WorkspaceError("loader says no")
        return {"render": {"preset": "eval_fast"}}
    mgr = WorkspaceManager(tmp_path / "store", {"c0": checkout}, config_loader=loader)
    work = mgr.materialize(mgr.base_ref("c0"))
    rep = HagentRepairer(mgr, hagent_settings(), mode="small")
    name = mgr.base_ref("c0").code_hash
    n = len(calls)
    assert rep._local_name(work) == name and len(calls) == n + 1                              # the manager's loader ran
    (work / "BROKEN").write_text("x")
    with pytest.raises(UnrebuildablePatch) as e:
        rep._local_name(work)
    assert e.value.cause == "config_unreadable" and "loader says no" in str(e.value)


def test_a_lossy_rebuild_is_refused_as_a_mismatch(tmp_path, checkout):
    """The patch applies but rebuilds as other files than the agent's own copy (not an encoding loss):
    the `mismatch` branch of ``HagentRepairer.repair``."""
    class Lossy(WorkspaceManager):
        alt = ""

        def extend(self, code_ref, patch):
            return super().extend(code_ref, self.alt)
    mgr = Lossy(tmp_path / "store", {"c0": checkout})
    c0 = mgr.base_ref("c0")
    work = mgr.materialize(c0)
    p = work / WRAPPER
    p.write_text(p.read_text().replace(BUG_LINE, STEP_A_LINE))
    mgr.alt = git(work, "diff", "HEAD")                                   # what the rebuild gives instead of the FIX
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, SUBMIT])).repair(request(code=c0))
    assert out.rebuild_cause == "mismatch" and out.code_ref == c0 and "rebuilds as" in out.rebuild_error
    assert FIXED_LINE in out.diff and "git settings" not in out.rebuild_error


# --- 6. an explicit eval_config survives a patch that does not touch the config ---------------------------------------

EXPLICIT = {"render": {"preset": "explicit"}}


def test_a_patch_that_does_not_touch_the_config_keeps_the_explicit_eval_config(mgr):
    base = mgr.base_ref("c0", eval_config=EXPLICIT)
    assert base.eval_config == EXPLICIT != mgr.base_ref("c0").eval_config
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, SUBMIT])
                         ).repair(request(code=base))
    assert out.code_ref.code_hash != base.code_hash
    assert out.code_ref.eval_config == EXPLICIT
    assert out.code_ref.key().split("+cfg:")[1] == base.key().split("+cfg:")[1]            # the same config key
    again = mgr.extend(out.code_ref, git(mgr.materialize(out.code_ref), "diff", "HEAD"))    # (empty patch: unchanged)
    assert again.eval_config == EXPLICIT


def test_a_patch_that_changes_the_config_takes_the_files_value_from_then_on(mgr):
    base = mgr.base_ref("c0", eval_config=EXPLICIT)
    edit = bash(f"sed -i 's/eval_fast/train_like/' {EVAL_CONFIG}", "Fixing the render preset.")
    a = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([edit, SUBMIT])
                       ).repair(request(RepairKind.CONFIG, code=base)).code_ref
    assert a.eval_config == {"render": {"preset": "train_like"}}
    b = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, SUBMIT])).repair(request(code=a)).code_ref
    assert b.code_hash != a.code_hash and b.eval_config == {"render": {"preset": "train_like"}}   # not the stale explicit one


# --- 7. the report model's spend counts ----------------------------------------------------------------------------------

def test_the_report_models_cost_is_part_of_the_attempts_cost_and_the_cap(mgr, c0):
    def make(report_cost):
        return HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, SUBMIT]),
                              report_model=ScriptedModel(["not json"], cost_per_call=report_cost, repeat_last=True))
    free, paid = make(0.0).repair(request(code=c0)), make(0.3).repair(request(code=c0))
    assert paid.cost > free.cost == 0.0 and paid.cost == pytest.approx(0.3 * paid.cost / 0.3) and paid.cost >= 0.3
    case = files_combined_case(c0)
    loop, _ = make_loop(case, make(0.3), workspaces=mgr, rung0=failing_rung0, cfg=loop_config(max_cost=0.2, max_repairs=5))
    res = loop.run(case.incident())
    assert len(res.repairs) == 1 and "cost cap" in res.reason and res.cost >= 0.3


# --- 11. Rung 0 wired into the loop ---------------------------------------------------------------------------------------

def test_a_real_bad_patch_is_rejected_by_rung0_from_the_loop_config_before_any_episode(mgr, c0):
    seen: list = []
    small = ScriptedModel([replace_line(BUG_LINE, "return ("), SUBMIT,
                           lambda m: seen.append(m) or bash("cat " + WRAPPER), FIX, SUBMIT])
    rep = HagentRepairer(mgr, hagent_settings(), mode="small", small=small)
    case = files_combined_case(c0)
    backend = Recording(case.quota_backend(mgr))
    before = set(Path(tempfile.gettempdir()).glob("rung0-*"))
    cfg = loop_config(rung0=Rung0Config(test_cmd=f"python3 -m py_compile {WRAPPER}"))
    loop, _ = make_loop(case, rep, rung0=None, backend=backend, workspaces=mgr, cfg=cfg)       # no hook passed
    assert loop.rung0 is not None
    res = loop.run(case.incident())
    assert [r.result for r in res.repairs] == ["rung0_failed", "rung1_pass_with_residual"]
    bad = res.repairs[0]
    assert bad.rung1 is None and bad.code_after not in backend.codes and bad.code_after != bad.code_before
    assert "SyntaxError" in bad.rung0.feedback and bad.diff                 # rejected on the real patch, by the real check
    assert "Rung 0 (tests) failed" in next(m["content"] for m in seen[0] if m["role"] == "user")
    assert set(Path(tempfile.gettempdir()).glob("rung0-*")) == before


def test_rung0_is_off_by_default_and_the_config_needs_what_it_builds_from(mgr):
    assert loop_config().rung0 is None
    case = combined_case()
    loop = IncidentLoop(case.quota_backend(), loop_config(), repairer=ScriptedRepairer([PATCHED]), inputs=case.inputs(),
                        data_ladder=make_loop(case, ScriptedRepairer([PATCHED]))[1])
    assert loop.rung0 is None
    with pytest.raises(ValueError, match="needs workspaces"):
        make_loop(case, ScriptedRepairer([PATCHED]), rung0=None, cfg=loop_config(rung0=Rung0Config()))
    with pytest.raises(ValueError, match="not both"):
        make_loop(case, HagentRepairer(mgr, hagent_settings()), cfg=loop_config(rung0=Rung0Config()))   # default hook passed too


def test_the_readme_rung0_example_builds_a_loop(mgr, c0):
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    block = re.search(r"```python\n(# rung0-example.*?)```", readme, re.S).group(1)
    case = files_combined_case(c0)
    loop0, ladder = make_loop(case, HagentRepairer(mgr, hagent_settings(), mode="small"), workspaces=mgr, rung0=None)
    ns = dict(router_cfg=loop_config().router, rung1_cfg=loop_config().rung1, backend=case.quota_backend(mgr),
              inputs=case.inputs(workspaces=mgr), data_ladder=ladder, reviewer_model=ScriptedModel(["APPROVE"]),
              repairer=HagentRepairer(mgr, hagent_settings(), mode="small"))
    exec(block, ns)
    assert callable(ns["loop"].rung0) and ns["loop"].cfg.rung0 is not None
