# PLAN F3 (Rung 0 adapter, errors, budgets, pass probabilities; ISSUES #9, #10, #11, #14, #4 Rung 2 part)

Baseline: hybrid-agent 143 passed / 19 skipped; vla-router 666 passed / 1 skipped (venv, `pip install -e` both).
All five issues checked against the code and still open.

1. **Rung 0 adapter (#9)** `ladder/rung0.py: rung0_hook(workspaces, settings, cfg, ...)` -> `Rung0Hook`.
   - `workspaces.materialize(out.code_ref)` into a temp dir, env from `hagent.runner.make_env` (honours docker), commands run
     there; `git_diff()` is the attempt's `out.diff` (the copy's HEAD is the patched code, so its own diff is empty). No patched files kept.
   - Reviewer: `reviewer=None` when `RepairOutcome.final_review_approved` (hagent's last review approved the diff).
     Otherwise the given reviewer runs. `run_rung0` catches `ModelError` from the reviewer: check status `error`, patch passes, `Rung0Result.reviewer_error`.
   - Failures: loop already sends `rung0_failed` back with feedback and skips Rung 1. Pinned by a real-bad-patch test.
2. **Errors (#10)** `IncidentLoop.run` catches `Exception` around the whole run (not KeyboardInterrupt/SystemExit). A stage marker
   (router/repairer/rung0/rung1/rung2) gives `LoopResult.error_stage/error_type/error_message`; status UNRESOLVED; all passes/repairs/rung2s kept; the repair record in flight gets `result="<stage>_error"`.
3. **Budgets (#11)** `LoopConfig.max_cost`, `max_big_calls` (0 = off). Checked before every repair attempt and before Rung 2 (spent >= cap stops). `IncidentLoop.warnings` /
   `LoopResult.warnings` from `HagentRepairer.pricing_warnings(max_cost)` (hagent's `pricing_warnings`). Simulator-episode cap: ISSUES #61.
4. **Rung 2 override (#4)**: implemented after the first pass, as decided by the user: Rung 2's label decides only on re-run rows 4 and 10; rows 6/7 go to the data ladder; every other row follows the router (`tests/test_f3_rung2_override_current.py`).
5. **Pass probabilities (#14)** `RepairOutcome.reviews/pass_probabilities`, copied to `RepairRecord`, in `to_dict`.
6. **Pending Rung 2 at the pass cap**: `LoopResult.rung2_pending`, reason text says so.
