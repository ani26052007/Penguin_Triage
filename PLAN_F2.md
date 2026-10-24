# PLAN F2 (proposed split: F2a, then F2b)

Unsplit F2 touches ~19 code files + DECISIONS.md + ISSUES.md (> 15), so it is split as asked.
No read-only file is edited (schemas.py, decide.py, vocab.py stay as they are; `CodeRef` keeps its shape).
Baseline: 602 passed, 1 skipped (fresh venv per README).

## F2a: code identity, workspaces, attempts, diff storage (+ #8 prompt path, #14 report task)

Files: ladder/workspace.py (new), ladder/repair.py, ladder/loop.py, ladder/backend.py, ladder/rung1.py,
router/report.py, DECISIONS.md, ISSUES.md. That is 6 code files.

1. `ladder/workspace.py`, the one place that hashes and builds code.
   - Recipe: `code_hash(base, cum_diff) = base` if the diff is empty, else `f"{base}+{sha1(cum_diff)[:12]}"`.
     `cum_diff` = `git add -A -N; git diff --no-color --no-ext-diff --src-prefix=a/ --dst-prefix=b/ refs/vla/base`
     in a working copy. The same content always gives the same name, whatever chain of attempts produced it.
   - `WorkspaceManager(bases={base_hash: checkout}, store_dir=None)`:
     - `materialize(code_ref, dest=None) -> Path`, in a fresh dir:
       1. `_copy_tree(base)`: the only copy step, where F5 will strip .git.
       2. Commit the copied tree as the base snapshot, with a fixed author and date so commit ids are deterministic, and set `refs/vla/base`.
       3. Apply the cumulative diff and commit it as "accepted", so HEAD is the current accepted code.
       4. Record the base hash in `git config vla.base`.

       Because HEAD is the accepted code, hagent's `git diff HEAD` is the attempt's own diff, under both local and docker.
     - `extend(code_ref, patch) -> CodeRef`: materialize, apply the attempt patch, recompute the cumulative diff, register and return the new ref. A patch that does not apply raises `WorkspaceError`.
     - `register(base, cum_diff)` and `diff_of(code_ref)`. The store is in memory and, when `store_dir` is set, also kept as `<code_hash>.patch` files. Nothing deletes a patch, because F3 needs it.
   - `code_ref_of(workdir, ...)`: the Rung 1 CLI default "after". It reads `vla.base` from the working copy and uses the same recipe.
   - The original checkout is only ever read (copytree source).
2. `HagentRepairer(workspaces, settings, ...)` replaces `repo`. Each attempt runs `materialize(req.incident.code_ref)`,
   which is the current accepted code, because the loop only advances `self.code` after Rung 1 returns pass, residual or ambiguous.
   It then runs `solve()` there and returns `extend(...)`. A failed attempt's dir is thrown away, so nothing leaks into the next one.
   `patched_code_ref` and `rung1.git_diff_hash` are removed (one recipe).
3. `RepairRecord.diff` holds the attempt's diff and `RepairRecord.code_diff` the cumulative diff of `code_after`.
   `LoopResult.final_diff` and `LoopResult.base` are added, and all of these go into `to_dict`.
   To recover the fix from a LoopResult: `register(base, final_diff)`, then `materialize(final_code)`.
4. Backend contract (docstring on `RolloutBackend`): a backend that needs files calls `workspaces.materialize(code_ref)`.
   `MockRule.files: FilePredicate` (e.g. `FileContains(path, text)`, serialisable for the CLI rules).
   `MockBackend(..., workspaces=)` materializes once per code key and caches the result.
   A files rule without workspaces raises. IsaacBackend is untouched.
5. #8: the prompt's repo path is the working copy, or `settings.env.workdir` under docker. `RepoIndex` is built from the working copy.
6. #14 decision: use the model's task. Plan 3.2.12 says the report model writes it, and `_Validator` already checks it (paths, ids).
   Today it is validated and then thrown away. Add `Report.task`. The repairer appends the config note and feedback to it.
   The fallback is still `default_task`. Recorded in DECISIONS.md.

Tests (they fail on the old code):
- Real HagentRepairer combined case on a patch-aware MockBackend: fix, lighting hand-off, `LoopResult.final_diff`, and `materialize(final_code)` gives the fixed files.
- Attempt 2 sees attempt 1's file contents, and a failed attempt does not change the next start.
- The original checkout is unchanged when a scripted agent writes to the path in the prompt, and the original path is not printed.
- `materialize()` is byte-identical per CodeRef (all files except .git).

The trace source for the patch-aware case is file-predicate based and lives in tests/.

## F2b: eval config, trace keys, config-contract unchecked (#2/51, #50, #52)

Files: ladder/workspace.py, ladder/repair.py, ladder/rung1.py, router/router.py, router/stage1/{base,registry,
staleness,cfg_contract,synthetic}.py, router/prereq/{trace,behavior_trace}.py, router/behavior_synthetic.py,
plus (per-check variant) router/stage1/{obs_contract,action_contract,sanity}.py and router/behavior.py.
That is 12 code files with the template variant, 16 with the per-check variant.

0. Top item (F2a review, ISSUES #59): C_behavior "not judged" and the obs/action contracts' "not checked" sub-checks
   become `PartialSkip`s with their signals, like #50 below, so they make their rows UNKNOWN instead of clean.
1. `ConfigLoader.load(root) -> Mapping`. `JsonFileConfigLoader(relpath)` stands in for the real format, which is TODO(verify).
   The manager takes the loader. `extend()` sets `eval_config` from the patched copy, and `base_ref(base)` loads it from the checkout.
   `materialize()` raises when a ref's `eval_config` differs from its files, so a ref cannot claim a config it does not have.
   The Rung 1 CLI "after" also gets its config through the loader.
2. #50: `run_partial` reasons become `PartialSkip(reason, signals)`. A missing mapped key gives a partial skip with that field's signal:
   render_settings -> `render_settings_differ`, anything else -> `contract_violation`.
   The registry collects these signals into `Stage1Result.unchecked`, and `router.unchecked_signals` uses that instead of its staleness special case (same result).
   `RouterResult.partial` keeps its `(source, reason)` shape.
3. #52: `EpisodeTrace.code_key` and `BehaviorTrace.code_key`. `None` counts as foreign.
   A trace whose key is not `incident.code_ref.key()` raises `TraceUnavailable` before the check reads anything.
   That makes the check skipped and its signals unchecked; for staleness, a foreign incident half is a partial skip.
   Where the check lives:
   (a) a helper `load_for_code(source, incident, mode)` called at the 6 load sites (recommended, cannot be bypassed); or
   (b) a template `load()` on the two source ABCs (4 fewer files, but a subclass overriding `load` skips it).
   The synthetic sources stamp the key.

Tests: a HagentRepairer config_fix in a file clears `render_settings_differ` on the re-run, the incident closes, and it does not hit the cap.
A trace with a foreign key is skipped and its rows become undecided.
A missing mapped key makes that field unchecked.

## Not done (as instructed)
decide() rows, IsaacBackend, Rung 0 wiring (F3), .git stripping / leakage (F5), Rung 2.

## Status
F2a done (with the approved changes a-e: diff-against-snapshot naming, always-on disk store, fresh one-commit snapshot honouring .gitignore, patch-aware synthetic logs, `unrebuildable_patch`). F2b next, using a template-method `load()` for #52 per the user decision.
