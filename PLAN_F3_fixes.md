# PLAN F3 fixes (F2b/F3 review; 11 items, no decide() change, no wider scope)

Baseline: hybrid-agent 143 passed / 19 skipped; vla-router 718 passed / 1 skipped (fresh venv, README commands).
New tests live in `vla-router/tests/test_f3_review_fixes.py` (plus the restored/strengthened tests in test_router.py and
test_repair.py). Each fails on the old code.

1. **Budget vs destination** (`loop.py`). `budget_stop()` is checked before a repair attempt (unchanged) and, in the
   pending-Rung-2 branch, before the call only on rows 4/10 (UNRESOLVED, as before). Rows 6/7 and the rest: `_Run.rung2()` itself
   returns the "unclear" fallback, reason "label not obtained: budget (...)", exactly as for `max_rung2_calls`, so both limits give
   the same destination (6/7 data ladder, others the router's).
2. **adopt() config error**. `extend` turns a `WorkspaceError` from reading the patched copy's eval config into
   `UnrebuildablePatch("config_unreadable")`; `rebuild_feedback` has advice for it.
3. **Nothing escapes run**. `_Run.__init__` no longer touches the workspaces; `base_name`/`diff_of` run in `_execute` under stage
   `setup` (bare/unknown code names, unknown labels, corrupt store -> UNRESOLVED, `error_stage="setup"`). Router added to the error test.
4. **Silent skips**: `fire()` with no values, obs image size / channel order / open-loop probe, gripper command length mismatch
   -> `PartialSkip(reason, contract_violation)`; one test each (fixtures that lacked those spec fields get them).
5. **`_local_name`** reads the config with `workspaces.config_loader`; an unreadable config raises `UnrebuildablePatch`
   (attempt refused). "mismatch" branch test restored.
6. **Explicit eval_config** kept by `extend` until a patch changes the config (as the loader reads it) versus the base snapshot.
7. **Report.cost** in `RepairOutcome.cost`.
8. **Stage `handoff`**.
9. Docs: Rung 2 headline reworded; ISSUES item for the Rung-2-not-configured row-4 re-run (behaviour unchanged).
10. Restored tests: test_router.py:107 exact partial; test_repair.py:62 base-config comparison; "mismatch".
11. `LoopConfig.rung0: Optional[Rung0Config]` (+ `rung0_reviewer`), built into `rung0_hook` by `IncidentLoop` from the repairer's
    workspaces/settings; README example; end-to-end test with the real `HagentRepairer`. Stale docs fixed.
