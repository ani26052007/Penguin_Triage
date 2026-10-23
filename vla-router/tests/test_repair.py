"""HagentRepairer: the real ``Repairer``, driving hagent's ``solve()`` with scripted models."""

from __future__ import annotations

import subprocess

import pytest

from hagent.config import Limits, Settings
from hagent.env import SUBMIT_MARKER
from hagent.models import ScriptedModel, bash
from ladder.repair import HagentRepairer, RepairKind, RepairRequest, patched_code_ref
from ladder.rung1 import git_diff_hash
from router.router import route
from router.schemas import CodeRef
from tests.ladder_cases import CASES
from tests.router_cases import default_config

SUBMIT = bash(f"echo {SUBMIT_MARKER} && git add -A && git diff --cached", "Done, submitting.")
FIX = bash("sed -i 's/a - b/a + b/' app.py && git diff --stat", "Fixing the operator.")


@pytest.fixture
def repo(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a - b\n")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return tmp_path


def settings(**limits) -> Settings:
    base = dict(max_steps=6, max_cost=0.0, max_wall_s=0, cmd_timeout_s=10, plan_at_step=0, final_review=False,
                ctx_char_limit=0, enable_explorer=False)
    base.update(limits)
    return Settings(limits=Limits(**base))


def request(kind=RepairKind.CODE, feedback=()) -> RepairRequest:
    case = CASES["code_global"]
    rr = route(case.incident(), case.quota_backend(), default_config(), case.evidence)
    assert rr.verdict.row == 4
    return RepairRequest(kind, case.incident(), rr.verdict, rr.evidence, rr.branch, 1, tuple(feedback))


def test_repair_returns_the_patched_code_ref_and_leaves_the_original_checkout_alone(repo):
    r = HagentRepairer(repo, settings(), mode="small", small=ScriptedModel([FIX, SUBMIT]))
    req = request()
    out = r.repair(req)
    assert out.code_ref.code_hash.startswith("c0+") and out.code_ref != req.incident.code_ref
    assert out.code_ref.eval_config == req.incident.code_ref.eval_config
    assert "return a + b" in out.diff and "Submitted" in out.notes
    assert (repo / "app.py").read_text().endswith("return a - b\n")           # the copy was edited, not the repo
    assert r.calls == 1


def test_prompt_frames_the_verdict_as_a_hypothesis_and_carries_the_feedback(repo):
    seen = []
    small = ScriptedModel([lambda m: seen.append(m) or SUBMIT])
    HagentRepairer(repo, settings(), mode="small", small=small, repro_command="run_rung1 --seeds 50").repair(
        request(feedback=("attempt 1: simulator validation failed.", "Rung 2 (big model) says a code problem remains: x")))
    task = next(m["content"] for m in seen[0] if m["role"] == "user")
    assert "Router hypothesis (verify this first" in task and "code (decision-table row 4)" in task
    assert "run_rung1 --seeds 50" in task and "Earlier attempts on this incident failed validation" in task
    assert "simulator validation failed" in task and "Rung 2 (big model)" in task
    assert "clean commit" not in task.lower() and "bug ledger" not in task.lower()


def test_config_kind_is_stated_in_the_task(repo):
    seen = []
    HagentRepairer(repo, settings(), mode="small", small=ScriptedModel([lambda m: seen.append(m) or SUBMIT])).repair(
        request(RepairKind.CONFIG))
    assert "configuration edit" in next(m["content"] for m in seen[0] if m["role"] == "user")


def test_unsubmitted_run_keeps_the_code_ref_so_the_loop_sees_no_change(repo):
    noop = bash("true", "Thinking.")
    out = HagentRepairer(repo, settings(max_steps=2), mode="small",
                         small=ScriptedModel([noop], repeat_last=True)).repair(request())
    assert out.code_ref == request().incident.code_ref and "not submitted" in out.notes


def test_patch_hash_recipe_matches_the_cli_default(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "i"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("2\n")
    from hagent.env import LocalEnv
    patch = LocalEnv(tmp_path).git_diff()
    assert patched_code_ref(CodeRef("c0"), patch).code_hash == f"c0+{git_diff_hash(str(tmp_path))}"
    assert patched_code_ref(CodeRef("c0"), "  \n") == CodeRef("c0")
