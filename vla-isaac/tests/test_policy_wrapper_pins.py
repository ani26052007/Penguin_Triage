"""The pilot pins (``tests/pw_pins/pilot_pins.py``) run against a copy of a policy-wrapper checkout.

They moved out of policy-wrapper so the agent's repo cannot see them (docs/IMPL_PLAN.md, "Decisions during
Phase 8"). Here they run as ``tests/test_pilot_pins.py`` inside a copy of the clean repo (all pass) and of
each injected repo (some fail where the pins reach). ``CAUGHT`` records which bugs the pins see; the others
are out of the pins' reach by design, for the reason given.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.bugs_support import LEDGER, needs_base, store  # noqa: F401  (store is a fixture)

PINS = Path(__file__).resolve().parent / "pw_pins" / "pilot_pins.py"
pytestmark = needs_base

CAUGHT = {"BUG-01", "BUG-02", "BUG-03", "BUG-05", "BUG-06", "BUG-07", "BUG-08", "BUG-09", "BUG-10", "BUG-12"}
NOT_CAUGHT = {
    "BUG-04": "the swap is inside the action YAML, which is also the pins' joint-name map",
    "BUG-11": "the pins' frames are bright; the path only runs below the brightness threshold",
    "BUG-13": "the pins' stand-in server accepts the state shape; the repo's own tests catch it (Rung 0)",
    "BUG-14": "the prefetch is off unless the incident-mode launcher sets its variable",
}


def run_pins(repo: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    work = tmp_path / "pins"
    shutil.copytree(repo, work, symlinks=True)
    shutil.copy(PINS, work / "tests" / "test_pilot_pins.py")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_pilot_pins.py"],
                          cwd=work, env=env, capture_output=True, text=True, timeout=600)


def test_every_bug_is_classified():
    assert CAUGHT | set(NOT_CAUGHT) == set(LEDGER.bugs) and not CAUGHT & set(NOT_CAUGHT)


def test_the_pins_are_not_in_the_repo_the_agent_gets(store):
    repo = store.repo().repo
    assert not (repo / "tests" / "test_pilot_pins.py").exists()
    assert not [p for p in repo.rglob("*.py") if "pilot_pins" in p.read_text()]


def test_the_pins_pass_on_the_clean_repo(store, tmp_path):
    r = run_pins(store.repo().repo, tmp_path)
    assert r.returncode == 0, r.stdout[-3000:]
    assert "6 passed" in r.stdout.splitlines()[-1]


@pytest.mark.parametrize("bug_id", sorted(LEDGER.bugs))
def test_the_pins_fail_on_an_injected_repo_they_reach(store, tmp_path, bug_id):
    r = run_pins(store.repo(bug_id).repo, tmp_path)
    if bug_id in CAUGHT:
        assert r.returncode == 1 and "failed" in r.stdout, r.stdout[-3000:]
    else:
        assert r.returncode == 0, (NOT_CAUGHT[bug_id], r.stdout[-3000:])
