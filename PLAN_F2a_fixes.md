# PLAN F2a review fixes

Source: the F0/F1/F2a review. Out of scope (F2b): config loader, PartialSkip, trace code keys, hagent's `git diff` recipe.
Order: write the new tests first, run them on the current code (they must fail), then change the code.

1. **Base identity in the name** (`ladder/workspace.py`). A code name is `<label>@<tree12>` or `<label>@<tree12>+<diff12>`.
   `<tree12>` is the first 12 hex of the git tree id of the base snapshot. `WorkspaceManager.base_ref(label)` names a checkout,
   and `materialize` re-snapshots the checkout and refuses a name whose tree does not match ("base checkout changed").
   The store no longer keeps `bases/<label>.tree`: the check is in the name, so it does not depend on which store is used.
   A changed checkout gets new names, so rollout-cache keys (`CodeRef.key()`) for the old content stop matching.
   `split_code_name` rejects old-recipe names (`c0+<12hex>`, chains) and bare labels with errors that say what to do.
   Tests: a changed checkout with a FRESH store gets a different name and the old name is refused; a `CachedBackend` on an old JSONL cache
   misses (0 hits) for the new content.
2. **Git environment.** `workspace._env()` drops every inherited `GIT_*` variable before adding its own.
   hagent: `env.clean_environ()` is used by `LocalEnv.execute` (the agent's commands), `LocalEnv._git`, and DockerEnv's CLI subprocesses.
   Test: with `GIT_DIR` / `GIT_WORK_TREE` / `GIT_INDEX_FILE` pointing at the original, `materialize`, `extend` and a real `HagentRepairer` repair
   leave the original byte-identical (`.git` included). hagent test: `LocalEnv` git helpers and commands see their own cwd.
3. **Bytes, not text.** git output, diffs, the hash input, the store files and `git apply` input are bytes. The public `str` surface
   (`cumulative_diff`, `diff_of`, `RepairOutcome.*diff`, `LoopResult.final_diff`) is a lossless view (`surrogateescape`, JSON-safe).
   `FileContains` compares bytes. `UnrebuildablePatch.cause` is one of `binary`, `encoding`, `symlink`, `does_not_apply`, `mismatch`,
   and the loop's feedback is chosen by cause. Under the local backend the repairer also checks that the rebuilt patch gives the same
   name as the agent's working copy (`mismatch` / `encoding` otherwise), so a lossy patch cannot silently rebuild different bytes.
   Tests: a non-UTF-8 edit and a non-UTF-8 context line round-trip name -> store -> materialize byte-exactly; feedback text per cause.
4. **Base snapshot contents.** Every file tracked in the original checkout is included even if a `.gitignore` matches it
   (`git ls-files` in the original, read only, then `git add -f`). Only untracked ignored files are dropped. Test: a force-added `cfg.yaml` under `*.yaml`.
5. **Failure cleanup.** A failed `materialize` removes what it created (temp dir or the `dest` it made; empties a given empty `dest`).
   `view()` takes a fresh slot per attempt and keeps nothing on failure, so a retry shows the real error. Tests for both.
6. **Symlinks: rejected** (not replaced by contents). A base checkout with an absolute symlink, or a relative one resolving outside the
   tree, is a `WorkspaceError`. A patch that creates one is `UnrebuildablePatch(cause="symlink")`. Tests for base and patch.
7. **`decide(unchecked=...)`.** A bare `str` is a `TypeError`; names not in `vocab.ALL_SIGNALS` are a `ValueError`. No routing change.
8. **Record completeness.** `LoopResult.base` (base name of the incident code). `_Run.code_diff` starts as the incident code's cumulative diff.
   `IncidentLoop(workspaces=...)` (default: the repairer's `workspaces`, if any). Test: an incident whose code is already patched;
   `final_diff` from the serialised result rebuilds `final_code` in a fresh store (fix accepted, and no fix accepted).
   DECISIONS.md: `git clean -X` is in `_snapshot_base`, not `_copy_base`.
9. **Coverage restored.** action_gap's behavior evidence ids are in the verdict (through `route()`); a whitespace-only patch to `extend()`
   gives the input ref; a stored diff that does not apply; row 9 through `route()`. These pin existing behaviour, so they may pass on the old code.
10. **ISSUES.md.** #6 FIXED once 1-5 are in; #8 and #30 PARTLY FIXED; new items for RUNG1_CONFIG leakage (F5), agent-rewritable
    `vla.base` / `refs/vla/base` (F5), behavior "not judged" and obs/action sub-check skips (F2b, top item), and the remaining
    untested branches / hard-coded values (F6).

Files: ladder/workspace.py, ladder/repair.py, ladder/loop.py, ladder/rung1.py (docstring only if needed), router/decide.py, router/vocab.py,
hybrid-agent/src/hagent/env.py, hybrid-agent/src/hagent/docker_env.py; tests; DECISIONS.md, ISSUES.md, PLAN_F2.md (F2b top item).

## Status
Done. New tests: `vla-router/tests/test_f2a_review_fixes.py` (29) and `hybrid-agent/tests/test_git_env.py` (2).
On the old code 26 of the 29 failed. The 3 that passed are coverage pins of behaviour that was already right:
action_gap's evidence ids in the verdict, row 9 through `route()`, and the generic "not a code name" error.
The item-9 whitespace and stored-diff tests failed on the old code only because they use `base_ref`.
Both hagent tests failed on the old code. Now: hybrid-agent 140 passed / 19 skipped; vla-router 653 passed / 1 skipped.
Existing tests now take the base from `mgr.base_ref("c0")` (a bare `CodeRef("c0")` is no longer a workspace name), and
`startswith("c0+")` became `startswith(c0.code_hash + "+")`, which is stricter. The changed-base test's `match="changed since"`
became `"base checkout 'c0' changed"` because the message changed. The Rung 1 CLI file-rule test gives its incident a workspace name.
No assertion was removed.
