"""Code identity and working copies (Fix F2a, ISSUES #6, #7, #13 partly): one naming recipe,
a content-addressed store, a fresh one-commit base snapshot, and file-aware mock rules."""

from __future__ import annotations

import pytest

from hagent.env import LocalEnv
from ladder.backend import MockBackend, MockRule
from ladder.workspace import (BASE_REF, FileContains, UnrebuildablePatch, WorkspaceError, WorkspaceManager,
                              code_ref_of, cumulative_diff)
from router.schemas import ActorKind, ActorRef, CodeRef, Mode, SceneSpec
from tests.workspace_cases import (BLOB, BUG_LINE, FIXED, FIXED_LINE, STEP_A_LINE, WRAPPER, git, make_checkout,
                                   manager, snapshot)

C0 = CodeRef("c0")


def edit(mgr: WorkspaceManager, ref: CodeRef, old: str, new: str, path: str = WRAPPER) -> str:
    """One attempt's diff, written the way hagent writes it: edit a working copy, ``git diff HEAD``."""
    work = mgr.materialize(ref)
    f = work / path
    f.write_text(f.read_text().replace(old, new))
    return LocalEnv(work).git_diff()


@pytest.fixture
def checkout(tmp_path):
    return make_checkout(tmp_path / "checkout")


@pytest.fixture
def mgr(tmp_path, checkout):
    return manager(tmp_path, checkout)


# --------------------------------------------------------------------------- #
# the base snapshot
# --------------------------------------------------------------------------- #

def test_base_snapshot_is_a_fresh_repo_with_one_commit_and_no_ignored_files(mgr, checkout):
    before = snapshot(checkout)
    head_before = git(checkout, "rev-parse", "HEAD")
    work = mgr.materialize(C0)
    assert git(work, "rev-list", "--all", "--count").strip() == "1"            # one commit, nothing else
    log = git(work, "log", "--all", "--format=%s%n%b")
    assert "BUG_LEDGER" not in log and "initial import" not in log             # the original history is not copied
    assert not (work / "secrets.txt").exists()                                 # .gitignore respected
    files = snapshot(work)
    assert files == {k: v for k, v in before.items() if k != "secrets.txt"}
    assert git(work, "rev-parse", BASE_REF) == git(work, "rev-parse", "HEAD")
    assert snapshot(checkout) == before and git(checkout, "rev-parse", "HEAD") == head_before   # read only
    assert git(checkout, "status", "--porcelain") == ""


def test_materialize_is_deterministic(mgr):
    patch = edit(mgr, C0, BUG_LINE, FIXED_LINE)
    ref = mgr.extend(C0, patch)
    a, b = mgr.materialize(ref), mgr.materialize(ref)
    assert a != b and snapshot(a) == snapshot(b)                                # byte-identical files
    assert git(a, "rev-parse", "HEAD") == git(b, "rev-parse", "HEAD")           # and identical commits
    assert snapshot(mgr.materialize(C0)) == snapshot(mgr.materialize(C0))


def test_a_changed_base_checkout_is_an_error_not_a_new_meaning_for_its_name(mgr, checkout):
    mgr.materialize(C0)
    (checkout / "README.md").write_text("changed\n")
    with pytest.raises(WorkspaceError, match="changed since"):
        mgr.materialize(C0)


def test_materialize_needs_an_empty_destination(mgr, tmp_path):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    with pytest.raises(WorkspaceError, match="not empty"):
        mgr.materialize(C0, tmp_path / "busy")


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #

def test_the_name_is_the_content_not_the_sequence_of_attempts(mgr):
    """Change (a): the new cumulative diff is read off the files, never concatenated."""
    a = mgr.extend(C0, edit(mgr, C0, BUG_LINE, STEP_A_LINE))
    ab = mgr.extend(a, edit(mgr, a, STEP_A_LINE, FIXED_LINE))                  # two steps
    direct = mgr.extend(C0, edit(mgr, C0, BUG_LINE, FIXED_LINE))               # one step
    readme = mgr.extend(C0, edit(mgr, C0, "controller repo", "controller repository", "README.md"))
    both = mgr.extend(readme, edit(mgr, readme, BUG_LINE, FIXED_LINE))
    both_other_order = mgr.extend(direct, edit(mgr, direct, "controller repo", "controller repository", "README.md"))
    assert ab == direct and both == both_other_order and len({r.key() for r in (C0, a, ab, readme, both)}) == 5
    undone = mgr.extend(a, edit(mgr, a, STEP_A_LINE, BUG_LINE))                # back to the base files
    assert undone == C0 and mgr.diff_of(undone) == ""
    assert mgr.extend(C0, "") == C0


def test_cli_recipe_and_repairer_recipe_agree(mgr):
    """``code_ref_of`` (Rung 1 CLI default "after") names a working copy exactly as ``extend`` does."""
    a = mgr.extend(C0, edit(mgr, C0, BUG_LINE, STEP_A_LINE))
    work = mgr.materialize(a)
    assert code_ref_of(work) == a
    f = work / WRAPPER
    f.write_text(f.read_text().replace(STEP_A_LINE, FIXED_LINE))
    (work / "controller" / "extra.py").write_text("X = 1\n")                   # new files count
    patch = LocalEnv(work).git_diff()
    assert "extra.py" in patch and STEP_A_LINE in patch                         # hagent's diff is this attempt only
    assert BUG_LINE not in patch
    assert code_ref_of(work) == mgr.extend(a, patch)
    assert cumulative_diff(work) == mgr.diff_of(mgr.extend(a, patch))


def test_a_new_manager_on_the_same_store_rebuilds_refs_from_another_instance(mgr, tmp_path, checkout):
    """Change (b): the store is on disk, keyed by code name."""
    ref = mgr.extend(C0, edit(mgr, C0, BUG_LINE, FIXED_LINE))
    expected = snapshot(mgr.materialize(ref))
    other = WorkspaceManager(tmp_path / "store", {"c0": checkout})
    assert snapshot(other.materialize(ref)) == expected
    assert (tmp_path / "store" / "patches" / f"{ref.code_hash}.patch").read_text() == mgr.diff_of(ref)


def test_a_diff_can_be_registered_in_a_fresh_store_and_must_match_its_name(mgr, tmp_path, checkout):
    ref = mgr.extend(C0, edit(mgr, C0, BUG_LINE, FIXED_LINE))
    fresh = WorkspaceManager(tmp_path / "elsewhere", {"c0": checkout})
    with pytest.raises(WorkspaceError, match="not in the store"):
        fresh.materialize(ref)
    fresh.register(ref, mgr.diff_of(ref))
    assert snapshot(fresh.materialize(ref)) == snapshot(mgr.materialize(ref))
    with pytest.raises(WorkspaceError, match="does not hash"):
        fresh.register(ref, mgr.diff_of(ref) + "\n")


def test_a_corrupt_store_file_is_detected(mgr, tmp_path):
    ref = mgr.extend(C0, edit(mgr, C0, BUG_LINE, FIXED_LINE))
    path = tmp_path / "store" / "patches" / f"{ref.code_hash}.patch"
    path.write_text(path.read_text().replace("norm_stats", "norm_statz"))
    with pytest.raises(WorkspaceError, match="corrupt"):
        mgr.materialize(ref)


def test_unknown_names_are_errors(mgr):
    for name in ("c9", "c0+0123456789ab", "not a name!"):
        with pytest.raises(WorkspaceError):
            mgr.materialize(CodeRef(name))


def test_a_binary_edit_from_a_plain_git_diff_is_unrebuildable(mgr):
    work = mgr.materialize(C0)
    (work / BLOB).write_bytes(b"\x00\x09\x08 changed")
    patch = LocalEnv(work).git_diff()                                          # hagent's diff: no --binary
    assert "Binary files" in patch
    with pytest.raises(UnrebuildablePatch):
        mgr.extend(C0, patch)
    with pytest.raises(UnrebuildablePatch):
        mgr.extend(C0, "--- a/nowhere.py\n+++ b/nowhere.py\n@@ -1 +1 @@\n-a\n+b\n")


# --------------------------------------------------------------------------- #
# file-aware mock rules
# --------------------------------------------------------------------------- #

def test_mock_rule_on_files_changes_the_rate_for_a_real_patch(mgr):
    fixed = mgr.extend(C0, edit(mgr, C0, BUG_LINE, FIXED_LINE))
    other = mgr.extend(C0, edit(mgr, C0, "controller repo", "controller repository", "README.md"))
    be = MockBackend([MockRule(1.0, files=FIXED)], 0.0, workspaces=mgr)
    actor, scene = ActorRef(ActorKind.POLICY, "p"), SceneSpec("s")
    assert all(e.success for e in be.run(scene, fixed, actor, range(10), Mode.CLEAN))
    assert not any(e.success for e in be.run(scene, other, actor, range(10), Mode.CLEAN))
    assert not any(e.success for e in be.run(scene, C0, actor, range(10), Mode.CLEAN))
    with pytest.raises(ValueError, match="workspaces"):
        MockBackend([MockRule(1.0, files=FIXED)], 0.0).run(scene, fixed, actor, [0], Mode.CLEAN)


def test_file_predicates_round_trip_and_stay_inside_the_code_root():
    p = FileContains(WRAPPER, "x", present=False)
    assert FileContains.from_dict(p.to_dict()) == p
    for bad in ("/etc/passwd", "../outside.py"):
        with pytest.raises(ValueError):
            FileContains(bad, "x")
