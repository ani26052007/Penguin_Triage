"""Rung 0 (plan.md Section 4): diff sanity, lint, tests, smoke run via bash, then the hagent reviewer."""

from __future__ import annotations

import subprocess

import pytest

from hagent.env import LocalEnv
from hagent.models import ScriptedModel
from ladder.rung0 import Rung0Config, diff_sanity, files_in_diff, run_rung0


@pytest.fixture
def repo(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "t")
    git("config", "user.email", "t@t")
    (tmp_path / "wrapper.py").write_text("def scale(x):\n    return x - 1\n")
    (tmp_path / "other.py").write_text("y = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    return tmp_path


@pytest.fixture
def env(repo):
    e = LocalEnv(repo, timeout_s=10)
    yield e
    e.close()


def edit(repo, name="wrapper.py", text="def scale(x):\n    return x + 1\n"):
    (repo / name).write_text(text)


def marker_cmd(repo, name):
    return f"touch {repo}/{name}"


def test_files_in_diff(repo, env):
    edit(repo)
    edit(repo, "new.py", "z = 3\n")
    assert files_in_diff(env.git_diff()) == ("new.py", "wrapper.py")


def test_empty_patch_fails_before_any_command_runs(repo, env):
    cfg = Rung0Config(lint_cmd=marker_cmd(repo, "lint_ran"))
    res = run_rung0(env, cfg)
    assert not res.passed and res.checks[0].name == "diff" and "empty" in res.feedback
    assert not (repo / "lint_ran").exists()


def test_forbidden_path_and_oversized_patch_fail(repo, env):
    edit(repo)
    d = diff_sanity(env.git_diff(), Rung0Config(forbidden_prefixes=("wrapper.py",)))
    assert d.status == "fail" and "forbidden" in d.detail
    big = diff_sanity(env.git_diff(), Rung0Config(max_diff_lines=2))
    assert big.status == "fail" and "smaller" in big.detail


def test_patch_must_touch_a_located_file(repo, env):
    edit(repo, "other.py", "y = 2\n")
    res = run_rung0(env, Rung0Config(expected_files=("wrapper.py",)))
    assert not res.passed and "none of the located files" in res.feedback
    edit(repo)
    assert run_rung0(env, Rung0Config(expected_files=("wrapper.py",))).passed       # one of them is enough


def test_commands_run_in_order_and_stop_at_the_first_failure(repo, env):
    edit(repo)
    cfg = Rung0Config(lint_cmd="echo lint ok", test_cmd="echo 'AssertionError: shape' && exit 1",
                      smoke_cmd=marker_cmd(repo, "smoke_ran"))
    res = run_rung0(env, cfg)
    assert [(c.name, c.status) for c in res.checks] == [("diff", "pass"), ("lint", "pass"), ("tests", "fail")]
    assert not res.passed and "AssertionError: shape" in res.feedback and "exit code 1" in res.feedback
    assert not (repo / "smoke_ran").exists()


def test_timeout_is_a_failure(repo, env):
    edit(repo)
    res = run_rung0(env, Rung0Config(smoke_cmd="sleep 5", timeout_s=1))
    assert not res.passed and "timed out" in res.feedback


def test_all_checks_pass_without_a_reviewer(repo, env):
    edit(repo)
    res = run_rung0(env, Rung0Config(lint_cmd="true", test_cmd="true", smoke_cmd="true"))
    assert res.passed
    assert [c.status for c in res.checks] == ["pass", "pass", "pass", "pass", "skipped"]


def test_unconfigured_commands_are_skipped_not_passed(repo, env):
    edit(repo)
    res = run_rung0(env, Rung0Config())
    assert [(c.name, c.status) for c in res.checks][1:4] == [("lint", "skipped"), ("tests", "skipped"),
                                                              ("smoke", "skipped")]


def test_reviewer_runs_last_and_only_after_the_cheap_checks(repo, env):
    edit(repo)
    reviewer = ScriptedModel(["APPROVE\nPASS_PROBABILITY: 0.9"])
    res = run_rung0(env, Rung0Config(test_cmd="echo tests done"), task="fix the scale bug", reviewer=reviewer)
    assert res.passed and res.reviewer_calls == 1 and res.pass_probability == pytest.approx(0.9)
    user = reviewer.calls[0][1]["content"]
    assert "fix the scale bug" in user and "return x + 1" in user and "tests done" in user

    skipped = ScriptedModel(["APPROVE"])
    run_rung0(env, Rung0Config(test_cmd="exit 1"), reviewer=skipped)
    assert skipped.calls == []                                   # a failed check never spends the big call


def test_reviewer_reject_fails_with_its_feedback(repo, env):
    edit(repo)
    res = run_rung0(env, Rung0Config(), reviewer=ScriptedModel(["REJECT: wrong sign\nPASS_PROBABILITY: 0.2"]))
    assert not res.passed and "wrong sign" in res.feedback and res.pass_probability == pytest.approx(0.2)
    assert res.checks[-1].name == "reviewer" and res.checks[-1].status == "fail"


def test_pass_probability_is_logged_never_gating(repo, env):
    edit(repo)
    res = run_rung0(env, Rung0Config(), reviewer=ScriptedModel(["APPROVE\nPASS_PROBABILITY: 0.01"]))
    assert res.passed and res.pass_probability == pytest.approx(0.01)


def test_unparseable_reviewer_reply_counts_as_approve(repo, env):
    edit(repo)
    assert run_rung0(env, Rung0Config(), reviewer=ScriptedModel(["¯\\_(ツ)_/¯"])).passed


def test_result_serialises(repo, env):
    edit(repo)
    d = run_rung0(env, Rung0Config(test_cmd="true")).to_dict()
    assert d["passed"] is True and d["checks"][0]["name"] == "diff"
