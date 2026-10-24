"""HagentRepairer: the real ``Repairer``, driving hagent's ``solve()`` with scripted models on
working copies from a ``WorkspaceManager`` (Fix F2a: ISSUES #6, #7, #8, #14)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from hagent.config import EnvConfig
from hagent.models import ScriptedModel, bash
from ladder.workspace import load_eval_config
from ladder.repair import HagentRepairer, RepairKind, RepairRequest
from router.report import verdict_text
from router.router import route
from router.schemas import CodeRef
from tests.ladder_cases import CASES
from tests.router_cases import default_config
from tests.workspace_cases import (BLOB, BUG_LINE, EVAL_CONFIG, FIXED_LINE, STEP_A_LINE, SUBMIT, WRAPPER, hagent_settings,
                                   make_checkout, manager, recording, replace_line, snapshot)

FIX = replace_line(BUG_LINE, FIXED_LINE)


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


def request(kind=RepairKind.CODE, feedback=(), *, code: CodeRef, case="code_global") -> RepairRequest:
    """``code``: the code under repair, a workspace name (``mgr.base_ref("c0")``)."""
    case = CASES[case]
    rr = route(case.incident(), case.quota_backend(), default_config(), case.inputs())
    assert rr.verdict.row == 4
    return RepairRequest(kind, replace(case.incident(), code_ref=code), rr.verdict, rr.evidence, rr.branch, 1,
                         tuple(feedback))


def user_text(messages) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


def test_repair_returns_a_rebuildable_code_ref_and_leaves_the_original_checkout_alone(mgr, checkout, c0):
    before = snapshot(checkout)
    r = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, SUBMIT]))
    out = r.repair(request(code=c0))
    assert out.code_ref.code_hash.startswith(c0.code_hash + "+") and out.code_ref.key() != c0.key()
    assert out.code_ref.eval_config == load_eval_config(mgr.materialize(out.code_ref))   # read from the patched files
    assert FIXED_LINE in out.diff and out.code_diff == mgr.diff_of(out.code_ref) and "Submitted" in out.notes
    assert (mgr.materialize(out.code_ref) / WRAPPER).read_text().count(FIXED_LINE) == 1
    assert snapshot(checkout) == before and r.calls == 1


def test_each_attempt_starts_from_the_code_it_is_asked_to_repair(mgr, c0):
    """ISSUES #7: attempt 2 sees attempt 1's accepted patch; asked to repair c0 again (attempt 1 was not
    accepted), it sees c0, not the rejected patch."""
    seen: list = []
    small = ScriptedModel([replace_line(BUG_LINE, STEP_A_LINE), SUBMIT,
                           bash(f"cat {WRAPPER}"), recording(seen, SUBMIT),
                           bash(f"cat {WRAPPER}"), recording(seen, SUBMIT)])
    r = HagentRepairer(mgr, hagent_settings(), mode="small", small=small)
    a = r.repair(request(code=c0)).code_ref
    assert r.repair(request(code=a)).code_ref.key() == a.key()          # looked, changed nothing
    assert r.repair(request(code=c0)).code_ref.key() == c0.key()
    after_accept, after_reject = (m[-1]["content"] for m in seen)       # the `cat` the agent saw
    assert STEP_A_LINE in after_accept and BUG_LINE not in after_accept
    assert BUG_LINE in after_reject and STEP_A_LINE not in after_reject


def test_the_agent_cannot_write_the_original_checkout_through_the_printed_path(mgr, checkout, c0):
    """ISSUES #8: the prompt names the working copy, so an agent that writes to the printed path edits
    the copy; the original path is never printed."""
    before = snapshot(checkout)
    seen: list = []

    def write_printed_path(messages):
        seen.append(messages)
        path = re.search(r"Controller/wrapper code: (\S+)", user_text(messages)).group(1)
        return bash(f"echo 'HACKED = True' >> {path}/{WRAPPER}", "Writing where the prompt says.")

    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([write_printed_path, SUBMIT])
                         ).repair(request(code=c0))
    assert snapshot(checkout) == before
    assert str(checkout) not in json.dumps(seen) and str(checkout.resolve()) not in json.dumps(seen)
    assert "HACKED = True" in out.diff and out.code_ref.key() != c0.key()     # it edited the copy


def test_under_docker_the_prompt_names_the_container_workdir(mgr):
    settings = hagent_settings()
    settings.env = EnvConfig(backend="docker", image="unused", workdir="/workspace/repo")
    r = HagentRepairer(mgr, settings, mode="small")
    assert r.prompt_path(Path("/tmp/somewhere")) == "/workspace/repo"
    assert HagentRepairer(mgr, hagent_settings()).prompt_path(Path("/tmp/x")) == "/tmp/x"


def test_prompt_frames_the_verdict_as_a_hypothesis_and_carries_the_feedback(mgr, c0):
    seen = []
    small = ScriptedModel([lambda m: seen.append(m) or SUBMIT])
    HagentRepairer(mgr, hagent_settings(), mode="small", small=small, repro_command="run_rung1 --seeds 50").repair(
        request(code=c0, feedback=("attempt 1: simulator validation failed.", "Rung 2 (big model) says a code problem remains: x")))
    task = user_text(seen[0])
    assert "Router hypothesis (verify this first" in task and "code (decision-table row 4)" in task
    assert "run_rung1 --seeds 50" in task and "Earlier attempts on this incident failed validation" in task
    assert "simulator validation failed" in task and "Rung 2 (big model)" in task
    assert "clean commit" not in task.lower() and "bug ledger" not in task.lower()


def test_the_report_models_task_is_used_and_the_verdict_line_still_comes_from_code(mgr, c0):
    """ISSUES #14 (DECISIONS F2a): the validated model task is the TASK; it cannot replace the verdict line."""
    req = request(code=c0, case="combined_code_plus_light")             # located: the normalisation evidence
    ev = req.evidence[0]
    model_task = f"Normalise the policy inputs in {WRAPPER} before they reach the policy."
    reply = json.dumps({"summary": "inputs are not normalised", "evidence_ids": [ev.id], "task": model_task,
                        "localization": [{"kind": "file", "name": WRAPPER, "evidence_ids": [ev.id],
                                          "reason": "builds the policy input"}]})
    seen = []
    HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([lambda m: seen.append(m) or SUBMIT]),
                   report_model=ScriptedModel([reply])).repair(req)
    task = user_text(seen[0])
    assert f"<task>\n{model_task}" in task
    assert f"The automatic router classified this failure as: {verdict_text(req.verdict)}" in task
    assert "Router hypothesis (verify this first" in task


def test_without_a_report_model_the_task_is_the_default(mgr, c0):
    seen = []
    HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([lambda m: seen.append(m) or SUBMIT])
                   ).repair(request(code=c0))
    assert "<task>\nThe VLA policy fails" in user_text(seen[0])


def test_config_kind_is_stated_in_the_task(mgr, c0):
    seen = []
    HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([lambda m: seen.append(m) or SUBMIT])
                   ).repair(request(RepairKind.CONFIG, code=c0))
    assert "configuration edit" in user_text(seen[0])


def test_unsubmitted_run_keeps_the_code_ref_so_the_loop_sees_no_change(mgr, c0):
    noop = bash("true", "Thinking.")
    out = HagentRepairer(mgr, hagent_settings(max_steps=2), mode="small",
                         small=ScriptedModel([noop], repeat_last=True)).repair(request(code=c0))
    assert out.code_ref == c0 and "not submitted" in out.notes and not out.rebuild_error


def test_a_binary_edit_is_reported_as_unrebuildable(mgr, c0):
    binary = bash(f"printf 'changed\\000' > {BLOB}", "Editing the blob.")
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([binary, SUBMIT])).repair(request(code=c0))
    assert out.code_ref == c0 and out.rebuild_error and "Binary files" in out.diff


def test_a_config_edit_shows_up_in_the_patched_eval_config(mgr, c0):
    """ISSUES #2: the patched CodeRef's eval config is what the patched file says, not the base's."""
    assert c0.eval_config == {"render": {"preset": "eval_fast"}}
    edit = bash(f"sed -i 's/eval_fast/train_like/' {EVAL_CONFIG}", "Fixing the render preset.")
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([edit, SUBMIT])
                         ).repair(request(RepairKind.CONFIG, code=c0))
    assert out.code_ref.eval_config == {"render": {"preset": "train_like"}} != c0.eval_config
    assert out.code_ref.key() != c0.key() and mgr.diff_of(out.code_ref)


def test_an_agent_that_commits_its_fix_still_yields_the_patch(mgr, c0):
    """ISSUES #56: hagent diffs against the commit it started at, so `git commit` does not hide the
    change (it used to give an empty patch, or a `mismatch` refusal)."""
    commit = bash("git -c user.name=agent -c user.email=agent@example.invalid commit -qam 'fix wrapper'",
                  "Committing the fix.")
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, commit, SUBMIT])
                         ).repair(request(code=c0))
    assert not out.rebuild_error, out.rebuild_error
    assert FIXED_LINE in out.diff and out.code_ref.key() != c0.key()
    assert (mgr.materialize(out.code_ref) / WRAPPER).read_text().count(FIXED_LINE) == 1
