"""scripts/gpu/passk.py, router_cfg_check.py and auto_repair.py (from the 2026-10-29 session) without a GPU.

No model is called and no simulator is started. The config check runs on the FIXTURE checkpoint
(tests/fixtures/checkpoint_gr1) and the policy-wrapper checkout, and skips when that checkout is absent."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.trace_fixtures import PW

GPU = Path(__file__).resolve().parents[1] / "scripts" / "gpu"
CKPT = Path(__file__).resolve().parent / "fixtures" / "checkpoint_gr1"


def load(name):
    sys.path.insert(0, str(GPU))
    spec = importlib.util.spec_from_file_location(name, GPU / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("name", ["router_cfg_check", "auto_repair"])
def test_the_session_scripts_carry_the_session_header(name):
    head = (GPU / f"{name}.py").read_text().split('"""')[1]
    assert "2026-10-29" in head and "ran only there" in head


# --------------------------------------------------------------------------- #
# passk
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n,c,k,want", [(5, 4, 3, 1.0), (5, 2, 3, 0.9), (5, 1, 3, 0.6), (5, 0, 3, 0.0), (5, 4, 1, 0.8),
                                        (5, 5, 5, 1.0), (10, 3, 2, 1 - 21 / 45), (4, 1, 4, 1.0)])
def test_pass_at_k_matches_known_values(n, c, k, want):
    assert load("passk").pass_at_k(n, c, k) == pytest.approx(want)


def test_pass_at_1_is_the_per_attempt_rate():
    m = load("passk")
    for n in range(1, 8):
        for c in range(n + 1):
            assert m.pass_at_k(n, c, 1) == pytest.approx(c / n)


@pytest.mark.parametrize("n,c,k", [(5, 6, 1), (5, -1, 1), (5, 2, 0), (5, 2, 6)])
def test_pass_at_k_refuses_impossible_arguments(n, c, k):
    with pytest.raises(ValueError):
        load("passk").pass_at_k(n, c, k)


def test_wilson_for_four_of_five_is_the_stated_interval(capsys):
    m = load("passk")
    lo, hi = m.wilson(4, 5)
    assert (round(lo, 2), round(hi, 2)) == (0.38, 0.96)
    assert m.main(["5", "2"]) == 0
    out = capsys.readouterr().out
    assert "pass@3 = 0.90" in out and "assumes a selector" in out


# --------------------------------------------------------------------------- #
# auto_repair: arguments and scoring (no model)
# --------------------------------------------------------------------------- #

def test_auto_repair_presets_and_arguments(tmp_path):
    m = load("auto_repair")
    a = m.parse_args(["repo", "--bug", "BUG-02", "--out", str(tmp_path), "--obs-nominal", "n", "--obs-check", "b"])
    assert (a.attempts, a.evidence) == (5, "obs") and a.score.startswith("absent:policy_wrapper/policy/gr00t_core.py:")
    a = m.parse_args(["repo", "--bug", "BUG-07", "--out", str(tmp_path), "-n", "2", "--checkpoint", "c"])
    assert (a.attempts, a.evidence, a.score) == (2, "cfg", "cfg")
    a = m.parse_args(["repo", "--bug", "BUG-03", "--out", str(tmp_path), "--task-file", "t.md"])
    assert a.evidence == "none"


@pytest.mark.parametrize("argv", [
    ["repo", "--bug", "BUG-02", "--out", "o"],                                # obs needs both directories
    ["repo", "--bug", "BUG-07", "--out", "o"],                                # cfg needs a checkpoint
    ["repo", "--bug", "BUG-03", "--out", "o"],                                # none needs a task file
    ["repo", "--bug", "BUG-99", "--out", "o"],                                # no preset
    ["repo", "--bug", "BUG-03", "--out", "o", "--task-file", "t", "-n", "0"],
    ["repo", "--bug", "BUG-03", "--out", "o", "--task-file", "t", "--score", "diff"],
    ["repo", "--bug", "BUG-03", "--out", "o", "--task-file", "t", "--score", "absent:onlyfile"],
])
def test_auto_repair_refuses_incomplete_arguments(argv):
    with pytest.raises(SystemExit):
        load("auto_repair").parse_args(argv)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


def test_the_line_check_needs_the_text_gone_and_only_that_file_changed(tmp_path):
    m = load("auto_repair")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "f.py").write_text("a = 1\nb = -x if flag else x\n")
    (tmp_path / "other.py").write_text("o = 1\n")
    git(tmp_path, "init", "-q"); git(tmp_path, "add", "."); git(tmp_path, "commit", "-q", "-m", "base")
    spec = "absent:pkg/f.py:-x if flag"
    assert m.changed_files(tmp_path) == [] and not m.score_absent(tmp_path, spec, m.changed_files(tmp_path))
    (tmp_path / "pkg" / "f.py").write_text("a = 1\nb = x\n")
    assert m.changed_files(tmp_path) == ["pkg/f.py"] and m.score_absent(tmp_path, spec, ["pkg/f.py"])
    (tmp_path / "other.py").write_text("o = 2\n")
    assert not m.score_absent(tmp_path, spec, m.changed_files(tmp_path))


# --------------------------------------------------------------------------- #
# router_cfg_check
# --------------------------------------------------------------------------- #

def test_cfg_check_arguments():
    m = load("router_cfg_check")
    a = m.parse_args(["--checkpoint", "ck", "clean=/a", "bug=/b"])
    assert a.repos == [("clean", Path("/a")), ("bug", Path("/b"))] and a.embodiment == "gr1"
    for bad in (["--checkpoint", "ck", "noequals"], ["clean=/a"]):
        with pytest.raises(SystemExit):
            m.parse_args(bad)


@pytest.mark.skipif(not (PW / ".git").exists(), reason=f"no policy-wrapper checkout at {PW}")
def test_cfg_check_on_the_fixture_checkpoint_clean_and_with_a_shorter_chunk(tmp_path, capsys):
    m = load("router_cfg_check")
    from vla_isaac.config import load_task_config
    from tests.helpers import PILOT_TASK

    task = load_task_config(PILOT_TASK)
    assert m.check_repo(PW, task, CKPT, m.DEFAULT_CKPT_ID) == []
    bugged = tmp_path / "pw"
    shutil.copytree(PW, bugged)
    cfg = bugged / "config" / "eval.yaml"
    text = cfg.read_text()
    assert "action_chunk_length: 16" in text
    cfg.write_text(text.replace("action_chunk_length: 16", "action_chunk_length: 8"))
    git(bugged, "commit", "-qam", "edit")
    ids = [e.id for e in m.check_repo(bugged, task, CKPT, m.DEFAULT_CKPT_ID)]
    assert ids == ["s1.cfg.replan_steps"]
    assert m.main(["--checkpoint", str(CKPT), f"clean={PW}"]) == 0
    assert "clean: 0 evidence item(s)" in capsys.readouterr().out
