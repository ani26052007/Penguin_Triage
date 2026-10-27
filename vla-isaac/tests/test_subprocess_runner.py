from __future__ import annotations

import sys

import pytest

from vla_isaac.runner.arena_cmd import PolicySpec, RunSpec
from vla_isaac.runner.subprocess_runner import ArenaRunError, RunInvocation, SubprocessRunner

SPEC = RunSpec("vla_b0", {"type": "t"}, 0, 1, 1, None, PolicySpec("p", "c", "cpu", "h", 1))


def inv(tmp_path, code: str, timeout: float = 30.0) -> RunInvocation:
    return RunInvocation((sys.executable, "-c", code), tmp_path, {"VLA_X": "1"}, timeout, tmp_path / "logs" / "run.log",
                         tmp_path / "out", SPEC)


def test_a_clean_exit_passes_and_the_log_is_kept(tmp_path):
    SubprocessRunner().run(inv(tmp_path, "import os; print('hello', os.environ['VLA_X'])"))
    assert "hello 1" in (tmp_path / "logs" / "run.log").read_text()


def test_a_crashed_subprocess_raises_with_its_exit_code_and_log_tail(tmp_path):
    with pytest.raises(ArenaRunError) as ei:
        SubprocessRunner().run(inv(tmp_path, "import sys; print('segfault-ish'); sys.exit(139)"))
    assert ei.value.returncode == 139 and not ei.value.timed_out and "segfault-ish" in str(ei.value)


def test_a_hung_subprocess_times_out(tmp_path):
    with pytest.raises(ArenaRunError) as ei:
        SubprocessRunner().run(inv(tmp_path, "import time; time.sleep(30)", timeout=0.5))
    assert ei.value.timed_out


def test_a_missing_executable_raises(tmp_path):
    bad = RunInvocation(("/nonexistent/python",), tmp_path, {}, 5, tmp_path / "l.log", tmp_path, SPEC)
    with pytest.raises(ArenaRunError, match="cannot start"):
        SubprocessRunner().run(bad)
