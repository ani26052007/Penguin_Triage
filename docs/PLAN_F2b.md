# PLAN F2b (trimmed)

Order and approach. Baseline: 654 tests pass (`PYTHONPATH=../hybrid-agent/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).

1. **Partial skips (#59).** `vocab.PartialSkip(reason, signals)`. `Check.run_partial` and
   `BehaviorAnalysis` return `PartialSkip`s; `Stage1Result.partial` / `RouterResult.partial` hold
   `(source, PartialSkip)`. `unchecked_signals` unions each skip's own signals; the `"staleness"`
   special case goes. C_behavior "not judged" -> skip with the matching signals (implied target,
   failure phase), added to `partial` when Branch C ran. Action / obs contract sub-check skips ->
   `contract_violation`. Test: vision_glossy, gripper never closes -> row 10, not 6.
2. **Config loader (#2).** After a repair, `extend()` reads the patched ref's `eval_config` from the
   patched working copy (the loader), never copying the base's. Fix `test_repair.py`'s assertion.
   Test: real `HagentRepairer` (scripted models) config_fix -> `render_settings_differ` gone, incident
   closes before the repair cap.
3. **Trace code key (#52).** `TraceSource.load()` is a template method checking the code key;
   subclasses implement `_load()`; structural test that none overrides `load()`.
4. **hagent #56.** Diff against the recorded starting commit, not HEAD. Test: scripted agent that
   `git commit`s still yields its patch.

Design choices go to DECISIONS.md.
