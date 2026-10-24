"""F2a review fixes (PLAN_F2a_fixes.md): base identity in code names, the git environment, bytes
end to end, base snapshot contents, failure cleanup, symlinks, ``unchecked`` validation, record
completeness, and the coverage the F1/F2a test changes had weakened."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

import ladder.loop as loop_mod
from hagent.env import LocalEnv
from hagent.models import ScriptedModel, bash
from ladder.backend import MockBackend, MockRule
from ladder.loop import LoopStatus
from ladder.repair import HagentRepairer
from ladder.workspace import (UnrebuildablePatch, WorkspaceError, WorkspaceManager, code_name, code_ref_of,
                              split_code_name)
from router import vocab as V
from router.behavior_synthetic import BehaviorFaults
from router.decide import decide, row1_state
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.router import route
from router.schemas import ActorKind, ActorRef, CodeRef, Mode, SceneSpec, VerdictLabel
from tests.ladder_cases import loop_config, make_loop
from tests.router_cases import default_config, router_cases
from tests.test_decide import BASE_C, RM, R, table
from tests.test_repair import request
from tests.workspace_cases import (BUG_LINE, FIXED_LINE, SUBMIT, WRAPPER, files_combined_case, git, hagent_settings,
                                   make_checkout, manager, replace_line, snapshot)

CASES = {c.name: c for c in router_cases()}
FIX = replace_line(BUG_LINE, FIXED_LINE)
README_EDIT = bash("sed -i 's/controller repo/controller repository/' README.md", "Editing the readme.")


@pytest.fixture
def checkout(tmp_path):
    return make_checkout(tmp_path / "checkout")


@pytest.fixture
def mgr(tmp_path, checkout):
    return manager(tmp_path, checkout)


def edit(mgr: WorkspaceManager, ref: CodeRef, old: str, new: str, path: str = WRAPPER) -> str:
    """One attempt's diff, written the way hagent writes it."""
    work = mgr.materialize(ref)
    f = work / path
    f.write_text(f.read_text().replace(old, new))
    return LocalEnv(work).git_diff()


def full_snapshot(root: Path) -> dict[str, bytes]:
    """Every regular file under ``root``, ``.git`` included, plus symlink targets."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            out[str(p.relative_to(root))] = os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
    return out


def bytes_edit(path: str, old: bytes, new: bytes) -> str:
    """An agent command that replaces bytes in a file (no text decoding)."""
    return bash(f"python3 - <<'PY'\nimport pathlib\np = pathlib.Path({path!r})\n"
                f"p.write_bytes(p.read_bytes().replace({old!r}, {new!r}))\nPY", "Editing bytes.")


def latin_checkout(root: Path) -> Path:
    """The standard checkout plus a committed latin-1 file: line 1 is not valid UTF-8."""
    co = make_checkout(root, binary=False)
    (co / "latin.txt").write_bytes(b"caf\xe9\nline2\nline3\n")
    git(co, "add", "latin.txt")
    git(co, "commit", "-q", "-m", "latin-1 file")
    return co


# --------------------------------------------------------------------------- #
# 1. base identity in the name
# --------------------------------------------------------------------------- #

def test_the_base_name_carries_the_snapshot_tree(mgr):
    c0 = mgr.base_ref("c0")
    label, tree, digest = split_code_name(c0.code_hash)
    assert label == "c0" and digest is None and len(tree) == 12 and c0.code_hash == f"c0@{tree}"
    work = mgr.materialize(c0)
    assert git(work, "rev-parse", "HEAD^{tree}").startswith(tree)
    assert code_ref_of(work) == c0


def test_a_changed_checkout_gets_a_new_name_even_with_a_fresh_store(tmp_path, checkout):
    old = manager(tmp_path, checkout).base_ref("c0")
    (checkout / "README.md").write_text("changed\n")
    fresh = WorkspaceManager(tmp_path / "fresh_store", {"c0": checkout})
    new = fresh.base_ref("c0")
    assert new.code_hash != old.code_hash and new.code_hash.startswith("c0@")
    with pytest.raises(WorkspaceError, match="base checkout 'c0' changed"):
        fresh.materialize(old)
    assert (fresh.materialize(new) / "README.md").read_text() == "changed\n"


def test_an_old_rollout_cache_does_not_serve_episodes_for_the_new_content(tmp_path, checkout):
    path = str(tmp_path / "rollouts.jsonl")
    scene, actor = SceneSpec("s"), ActorRef(ActorKind.POLICY, "p")
    old = manager(tmp_path, checkout).base_ref("c0")
    first = CachedBackend(MockBackend([MockRule(1.0, code=old)], 0.0), RolloutCache(path))
    assert all(e.success for e in first.run(scene, old, actor, range(10), Mode.CLEAN))

    (checkout / "README.md").write_text("changed\n")
    new = WorkspaceManager(tmp_path / "fresh_store", {"c0": checkout}).base_ref("c0")
    second = CachedBackend(MockBackend([MockRule(1.0, code=old)], 0.0), RolloutCache(path))
    episodes = second.run(scene, new, actor, range(10), Mode.CLEAN)
    assert second.hits == 0 and second.misses == 10 and not any(e.success for e in episodes)


@pytest.mark.parametrize("name,match", [
    ("c0+0123456789ab", "old-recipe"),
    ("c0+0123456789ab+abcdef012345", "old-recipe"),
    ("c0", "bare base label"),
    ("not a name!", "not a workspace code name"),
])
def test_old_recipe_and_bare_names_are_rejected_at_parse_time(name, match):
    with pytest.raises(WorkspaceError, match=match):
        split_code_name(name)


# --------------------------------------------------------------------------- #
# 2. git environment
# --------------------------------------------------------------------------- #

def test_inherited_git_variables_cannot_touch_the_original(mgr, checkout, monkeypatch):
    before = full_snapshot(checkout)
    monkeypatch.setenv("GIT_DIR", str(checkout / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(checkout))
    monkeypatch.setenv("GIT_INDEX_FILE", str(checkout / ".git" / "index"))
    c0 = mgr.base_ref("c0")
    ref = mgr.extend(c0, edit(mgr, c0, BUG_LINE, FIXED_LINE))
    assert FIXED_LINE in (mgr.materialize(ref) / WRAPPER).read_text()
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, SUBMIT])).repair(
        request(code=c0))
    assert out.code_ref == ref and not out.rebuild_error
    assert full_snapshot(checkout) == before


# --------------------------------------------------------------------------- #
# 3. bytes, not text
# --------------------------------------------------------------------------- #

def test_non_utf8_edits_and_context_round_trip_byte_exactly(tmp_path):
    co = latin_checkout(tmp_path / "co")
    mgr = WorkspaceManager(tmp_path / "store", {"c0": co})
    c0 = mgr.base_ref("c0")
    contents = (b"caf\xe9\nline2 \xff\xfe\nline3\n",      # the edit itself is not UTF-8
                b"caf\xe9\nline2 changed\nline3\n",       # the context line next to the edit is not UTF-8
                b"caf\xe8\nline2\nline3\n")               # a different invalid byte
    refs = []
    for content in contents:
        work = mgr.materialize(c0)
        (work / "latin.txt").write_bytes(content)
        refs.append(mgr.adopt(work))
    assert len({r.code_hash for r in refs}) == 3
    fresh = WorkspaceManager(tmp_path / "fresh", {"c0": co})
    for ref, content in zip(refs, contents):
        diff = json.loads(json.dumps(mgr.diff_of(ref)))         # the str form survives JSON losslessly
        fresh.register(ref, diff)
        rebuilt = fresh.materialize(ref)
        assert (rebuilt / "latin.txt").read_bytes() == content
        assert code_ref_of(rebuilt) == ref


def test_file_predicates_read_bytes(tmp_path):
    from ladder.workspace import FileContains
    (tmp_path / "f").write_bytes(b"caf\xe9 normalize_inputs(obs)\n")
    assert FileContains("f", "normalize_inputs(obs)")(tmp_path)
    assert not FileContains("f", "caf�")(tmp_path)                       # no replacement characters


def test_rebuild_feedback_names_the_cause():
    texts = {c: loop_mod.rebuild_feedback(2, c, "detail") for c in
             ("binary", "encoding", "symlink", "does_not_apply", "mismatch")}
    assert len(set(texts.values())) == 5 and all(t.startswith("attempt 2:") and "detail" in t for t in texts.values())
    assert [c for c, t in texts.items() if "binary file" in t] == ["binary"]
    assert "UTF-8" in texts["encoding"] and "symbolic link" in texts["symlink"]
    assert "commit" in texts["mismatch"] and "does not apply" in texts["does_not_apply"]


def test_a_lossy_patch_next_to_non_utf8_context_is_an_encoding_error(tmp_path):
    co = latin_checkout(tmp_path / "co")
    mgr = WorkspaceManager(tmp_path / "store", {"c0": co})
    c0 = mgr.base_ref("c0")
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel(
        [bytes_edit("latin.txt", b"line2", b"line2 changed"), SUBMIT])).repair(request(code=c0))
    assert out.code_ref == c0 and out.rebuild_cause == "encoding" and out.rebuild_error


def test_a_lossy_patch_that_still_applies_is_caught_not_rebuilt_wrong(tmp_path):
    """The agent writes a non-UTF-8 byte; hagent's text patch carries U+FFFD and applies cleanly,
    but would rebuild different bytes. The local cross-check refuses it."""
    co = latin_checkout(tmp_path / "co")
    mgr = WorkspaceManager(tmp_path / "store", {"c0": co})
    c0 = mgr.base_ref("c0")
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel(
        [bytes_edit("latin.txt", b"line3", b"line3 \xff"), SUBMIT])).repair(request(code=c0))
    assert out.code_ref == c0 and out.rebuild_cause == "encoding"


def test_committed_changes_are_kept_in_the_patch(mgr):
    """Was `mismatch` (hagent diffed against HEAD, so the commit hid the change); since F2b (ISSUES #56) the
    patch is against the starting commit and the committed change rebuilds."""
    c0 = mgr.base_ref("c0")
    commit = bash("git -c user.name=a -c user.email=a@a commit -qam fix", "Committing.")
    out = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, commit, SUBMIT])).repair(
        request(code=c0))
    assert out.code_ref != c0 and not out.rebuild_cause and FIXED_LINE in out.diff


# --------------------------------------------------------------------------- #
# 4. base snapshot contents
# --------------------------------------------------------------------------- #

def test_tracked_files_matching_gitignore_are_in_the_snapshot(tmp_path):
    co = tmp_path / "co"
    co.mkdir()
    (co / ".gitignore").write_text("*.yaml\n")
    (co / "cfg.yaml").write_text("gain: 2\n")
    (co / "x.py").write_text("X = 1\n")
    git(co, "init", "-q")
    git(co, "add", "-A")
    git(co, "add", "-f", "cfg.yaml")
    git(co, "commit", "-q", "-m", "init")
    (co / "local.yaml").write_text("untracked: 1\n")                    # untracked and ignored
    mgr = WorkspaceManager(tmp_path / "store", {"c0": co}, lambda work: {})
    work = mgr.materialize(mgr.base_ref("c0"))
    assert (work / "cfg.yaml").read_text() == "gain: 2\n" and (work / "x.py").exists()
    assert not (work / "local.yaml").exists()
    assert "cfg.yaml" in git(work, "ls-files").split()


# --------------------------------------------------------------------------- #
# 5. failure cleanup
# --------------------------------------------------------------------------- #

def test_a_failed_materialize_removes_what_it_created(mgr, checkout, tmp_path, monkeypatch):
    c0 = mgr.base_ref("c0")
    (checkout / "README.md").write_text("changed\n")
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    with pytest.raises(WorkspaceError, match="changed"):
        mgr.materialize(c0)
    assert list(tmp.iterdir()) == []
    with pytest.raises(WorkspaceError, match="changed"):
        mgr.materialize(c0, tmp_path / "new_dest")
    assert not (tmp_path / "new_dest").exists()
    given = tmp_path / "given"
    given.mkdir()
    with pytest.raises(WorkspaceError, match="changed"):
        mgr.materialize(c0, given)
    assert given.is_dir() and list(given.iterdir()) == []


def test_a_view_retry_shows_the_real_error(mgr, checkout):
    c0 = mgr.base_ref("c0")
    (checkout / "README.md").write_text("changed\n")
    for _ in range(2):
        with pytest.raises(WorkspaceError, match="base checkout 'c0' changed"):
            mgr.view(c0)


# --------------------------------------------------------------------------- #
# 6. symlinks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("target", ["/etc/hostname", "../outside.txt", "controller/../../outside.txt"])
def test_escaping_symlinks_in_the_base_are_rejected(tmp_path, checkout, target):
    (tmp_path / "outside.txt").write_text("outside\n")
    (checkout / "link").symlink_to(target)
    with pytest.raises(WorkspaceError, match="symlink"):
        WorkspaceManager(tmp_path / "store", {"c0": checkout}).base_ref("c0")


def test_an_internal_relative_symlink_is_kept(tmp_path, checkout):
    (checkout / "link").symlink_to("controller/norm.py")
    mgr = WorkspaceManager(tmp_path / "store", {"c0": checkout})
    work = mgr.materialize(mgr.base_ref("c0"))
    assert os.readlink(work / "link") == "controller/norm.py"


def test_a_patch_that_adds_an_escaping_symlink_is_unrebuildable(mgr, tmp_path):
    c0 = mgr.base_ref("c0")
    work = mgr.materialize(c0)
    (work / "escape").symlink_to(str(tmp_path))
    patch = LocalEnv(work).git_diff()
    with pytest.raises(UnrebuildablePatch) as e:
        mgr.extend(c0, patch)
    assert e.value.cause == "symlink"


# --------------------------------------------------------------------------- #
# 7. decide(unchecked=...)
# --------------------------------------------------------------------------- #

def test_unchecked_rejects_a_bare_string_and_unknown_names():
    t = table(BASE_C, {RM: R})
    with pytest.raises(TypeError, match="collection"):
        decide([], t, unchecked=V.SIG_RENDER_SETTINGS_DIFFER)
    with pytest.raises(TypeError, match="collection"):
        row1_state([], t, unchecked=V.SIG_STALENESS_VARIABLE)
    with pytest.raises(ValueError, match="render_setting_differ"):
        decide([], t, unchecked={"render_setting_differ"})
    assert decide([], t, unchecked={V.SIG_RENDER_SETTINGS_DIFFER}).row == 10
    every = frozenset().union(*V.SIGNALS_BY_SOURCE.values())
    assert every <= V.ALL_SIGNALS
    decide([], t, unchecked=V.ALL_SIGNALS)                                 # all known names are accepted


# --------------------------------------------------------------------------- #
# 8. record completeness
# --------------------------------------------------------------------------- #

def _rebuild_from_record(d: dict, store: Path, checkout: Path) -> Path:
    fresh = WorkspaceManager(store, {"c0": checkout})
    final = CodeRef.from_dict(d["final_code"])
    fresh.register(final, d["final_diff"])
    return fresh.materialize(final)


def test_an_already_patched_incident_code_rebuilds_from_the_serialised_result(mgr, checkout, tmp_path):
    c0 = mgr.base_ref("c0")
    start = mgr.extend(c0, edit(mgr, c0, "controller repo", "controller repository", "README.md"))
    case = files_combined_case(start)
    rep = HagentRepairer(mgr, hagent_settings(), mode="small", small=ScriptedModel([FIX, SUBMIT]))
    loop, _ = make_loop(case, rep, workspaces=mgr)
    res = loop.run(case.incident())
    assert res.status is LoopStatus.DATA_HANDOFF and res.base == c0.code_hash
    d = json.loads(json.dumps(res.to_dict()))
    assert d["base"] == c0.code_hash
    files = snapshot(_rebuild_from_record(d, tmp_path / "fresh", checkout))
    assert FIXED_LINE.encode() in files[WRAPPER] and files["README.md"].startswith(b"controller repository")


def test_without_an_accepted_fix_final_diff_is_the_incident_codes_own_diff(mgr, checkout, tmp_path):
    c0 = mgr.base_ref("c0")
    start = mgr.extend(c0, edit(mgr, c0, "controller repo", "controller repository", "README.md"))
    case = files_combined_case(start)
    rep = HagentRepairer(mgr, hagent_settings(max_steps=2), mode="small",
                         small=ScriptedModel([bash("true", "Thinking.")], repeat_last=True))
    loop, _ = make_loop(case, rep, workspaces=mgr, cfg=loop_config(max_repairs=1))
    res = loop.run(case.incident())
    assert res.status is LoopStatus.UNRESOLVED and res.final_code == start
    assert res.final_diff == mgr.diff_of(start) != ""
    d = json.loads(json.dumps(res.to_dict()))
    assert snapshot(_rebuild_from_record(d, tmp_path / "fresh", checkout)) == snapshot(mgr.materialize(start))


# --------------------------------------------------------------------------- #
# 9. coverage restored
# --------------------------------------------------------------------------- #

def test_action_gap_behavior_evidence_reaches_the_verdict():
    case = CASES["action_gap"]
    res = route(case.incident(), case.quota_backend(), default_config(), case.inputs())
    assert res.verdict.row == 8 and res.behavior_evidence
    assert {e.id for e in res.behavior_evidence} <= set(res.verdict.evidence_ids)


def test_row_9_through_route():
    case = replace(CASES["action_gap"], behavior_default=BehaviorFaults(scatter=0.08, fail_phase="grasp"))
    res = route(case.incident(), case.quota_backend(), default_config(), case.inputs())
    assert res.verdict.row == 9 and res.verdict.label is VerdictLabel.PERCEPTION_UNRESOLVED, res.verdict.notes
    assert any(e.signal == V.SIG_IMPLIED_TARGET_WRONG for e in res.behavior_evidence)


def test_a_whitespace_only_patch_gives_the_input_ref(mgr):
    c0 = mgr.base_ref("c0")
    assert mgr.extend(c0, "  \n\t\n") == c0


def test_a_stored_diff_that_does_not_apply_is_an_error(mgr):
    c0 = mgr.base_ref("c0")
    diff = "--- a/nowhere.py\n+++ b/nowhere.py\n@@ -1 +1 @@\n-a\n+b\n"
    ref = mgr.register(CodeRef(code_name(c0.code_hash, diff)), diff)
    with pytest.raises(WorkspaceError, match="does not apply"):
        mgr.materialize(ref)
