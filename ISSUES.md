# Issues

Numbers are stable. New findings get the next free number.

## Tier 1: The system would route real incidents wrongly

### 1. FIXED in F1: Stage 1 and C_behavior are never run, so the router works blind in production (cross-file).

route() doesn’t call run_stage1 or run_behavior, and nothing outside tests/ does either. Evidence only gets in through extra_evidence. In the loop, that comes from the evidence_by_code fixture, which returns hand-written evidence for c0 and nothing for any other code.

If you strip the fixture away, your own runs show what happens:

- Infra (stale server) closes as “no significant failure”, row 1. Nothing produces staleness_variable, so a clean-mode Stage 0 pass stops the router (05_stage1 Part 11, run 3). This is the worst case because it fails silently.
- Rendering mismatch becomes a vision gap, row 6. C_render never runs without render_settings_differ, so you would spend money on data generation for what is really a config edit (04 §6.6).
- Condition-specific code bugs become vision gaps. This is the strict xfail.
- Rows 8 and 9 can’t be reached. Action gaps fall to row 10.
- Located code verdicts degrade to “by elimination” in Branch G, and become impossible in Branch C.

The plan says Stage 1 runs “on every incident”, and the prompts never ask for that wiring. Step 5 forbade touching the replay stages, and step 7 used a fixture. Steps 8–12 don’t mention it either.

### 2. Fixing the code doesn’t make the evidence go away; the fixture does (cross-file).

(Still open after F2a: a patched CodeRef still copies the base eval_config. F2b.)

Today, evidence vanishes after a patch only because the fixture returns () for any new CodeRef. With real checks, a config fix is still a problem. patched_code_ref keeps the base eval_config (repair.py:70-71), while the config contract reads incident.code_ref.eval_config. So a render-settings fix made by hagent in a file would keep firing render_settings_differ.

The deeper issue is that the eval config exists in two places that are never linked: the repo files hagent edits, and CodeRef.eval_config, which the router hashes and the backend reads.

### 3. Thresholds are never calibrated, and the prompts have no step for it (cross-file).

Plan 3.2.9 says to set thresholds by running the router many times on no-fault incidents, keeping the false-verdict rate under a target, and reporting that rate. Plan Section 8, step 6 includes these calibration runs. No prompt asks for them. Step 8’s runner reports the false-verdict rate, but nothing sets thresholds from it.

These values are all placeholders today:

- fire_z=4.0
- every floor (synthetic.FLOORS, which dominates the synthetic spread anyway)
- min_strength=0
- the behavior thresholds
- the 0.15 / (0.70, 0.95) anchors

You can already see the cost. In the confusion matrix, action_gap became vision_factor in 2/10 repeats, because a no-effect restore drew 9 or 10 out of 30 and read as partial. The router replays also look twice, at 10 seeds and then at the cap, at 95% confidence with no correction. Many replays run per incident.

### 4. Model outputs can change routes, despite “models never decide”.

There are three paths:

- Rung 2 overrides the re-run. After an ambiguous Rung 1, the code and unclear labels are not gated. A row-6 vision re-run can be sent back to hagent, or dropped as UNRESOLVED (01_WHY #1(b), 07 §5.6). Only data_gap is gated.
- The config mapper. An unapproved mapping that maps a field to null turned row 4 into row 10 (06 §6.4, C3). approved is never read anywhere.
- extra_evidence has no provenance check. Anything passed in is trusted.

The structural test can’t catch any of this. It only sees direct absolute imports, and relative, dynamic and transitive imports all pass it (06 §8.4, M2–M6).

### 5. The reference controller is never qualified.

Rows 2 and 4 and the whole sim-vs-policy split depend on it. reference_unqualified has no producer, and no qualification code exists. The reference controller is also judged against the policy’s nominal band, which may not suit a scripted controller.

## Tier 2: The repair loop can’t produce a usable fix yet

### 6. FIXED in F2a and the F2a review fixes: The patched code can’t be rebuilt, and the patch itself is thrown away (cross-file).

(F2a: `ladder/workspace.py`. A code name is `<base>+sha1(cumulative diff)[:12]`, every cumulative diff is kept in an on-disk content-addressed store, and `materialize(code_ref)` rebuilds the files. RepairRecord stores each attempt's diff and the cumulative one; LoopResult has `final_diff`. MockBackend rules can test the files (`MockRule.files`).)

(F2a review: F2a only partly fixed it. A fresh store accepted a changed base checkout under the old name, and the rollout cache never checked; text decoding lost non-UTF-8 bytes; tracked files that a `.gitignore` matches were dropped; a failed build left a temp dir and blocked `view()` retries; escaping symlinks were copied. The F2a review fixes close all five: names are `<label>@<tree12>[+<diff12>]`, everything is bytes, tracked files are kept, failures clean up, escaping symlinks are rejected. hagent's own lossy patch remains: #55, #56, F2b.)

RepairRecord doesn’t store the diff. The loop keeps diff text only when it is pasted into feedback. LoopResult.final_code is a name such as c0+4a5db660a6a9, with no recoverable content. A successful fix is lost.

No backend can build <base>+<sha1(patch)>. On MockBackend, any real patch hash falls to default_rate, so the loop with the real HagentRepairer fails Rung 1 every time and hits the repair cap (08 §4).

### 7. FIXED in F2a: Every repair attempt starts from the original checkout.

(F2a: each attempt materializes `request.incident.code_ref`, the last accepted code; a failed attempt's copy is discarded.)

HagentRepairer copies self.repo on every attempt. Attempt 2, for example after Rung 2 says code, runs on unpatched code, but its result is named c0+h1+h2. The name claims two stacked patches, while the content is one patch written against the original code.

### 8. PARTLY FIXED in F2a: The prompt tells the agent where the original repo is.

(F2a: the prompt names the working copy, or `settings.env.workdir` under docker, and the RepoIndex is built from the working copy. DockerEnv itself has still never been run.)

(F2a review: the prompt no longer names the original, and the F2a review fixes stop two other routes to it: inherited `GIT_*` variables (workspace git, hagent's git helpers and the agent's commands) and escaping symlinks. Still open: `$RUNG1_CONFIG` names the original path (#57), and the agent can rewrite its copy's repo refs (#58).)

The Controller/wrapper code: line prints the original path, not the working copy. In your run, a scripted agent edited the original checkout and the repairer reported no_change (08 §4). LocalEnv gives no isolation. DockerEnv would block this, but it has never been run, because the test image isn’t built and all 19 tests skip.

### 9. Rung 0 isn’t connected.

- run_rung0(env, cfg) doesn’t match the loop’s Rung0Hook(request, outcome) signature, and no adapter exists.
- The patched working copy is deleted when repair() returns, so there is nothing for Rung 0 to lint or test. (F2a: the patch is stored and `materialize()` rebuilds any code; the adapter is F3.)
- Loop tests use passing_rung0, which always passes.
- A reviewer ModelError in Rung 0 crashes instead of failing open. Hagent’s own review path does catch it.

### 10. Errors crash the loop instead of ending the incident as UNRESOLVED.

These all propagate: EnvError (bad command path), ModelError (missing model), Docker start-up failures, and the Rung 0 reviewer. Neither HagentRepairer.repair nor IncidentLoop has a try.

### 11. Budgets leak.

- Hagent’s caps reset on every attempt, so one incident can spend 3× the per-run caps.
- The loop adds up cost and big_calls but never checks them against a limit.
- max_cost is inert when prices are 0, which is the default, and pricing_warnings is never called.
- The plan says the Rung 2 call counts toward max_big_calls. It doesn’t.

### 12. run_rung1 as an agent tool won’t work as intended.

- It shares cmd_timeout_s (60 s). A real Isaac validation run can’t finish in that time.
- The agent’s own run_rung1 runs don’t count toward any simulator budget.
- The reviewer never sees its output, because TEST_CMD_RE doesn’t match run_rung1.
- Calling it as ./run_rung1 loses its own truncation limit.
- No production code registers it in Settings, but the prompt always says “run_rung1 --seeds 50”.

### 13. PARTLY FIXED in F2a: Leakage is open before step 8.

- .git is copied into the working copy. (F2a: the working copy is a fresh `git init` with one snapshot commit; the original history and the files `.gitignore` ignores are not copied. File contents themselves, e.g. a README naming the bug, are F5's.)
- RepoIndex includes .md/.json files, so a BUG_LEDGER.md in the repo becomes the top localization candidate (06 H7).
- incident_id is printed verbatim.
- Rung 2’s free-text reason goes into the hagent prompt without validation.

### 14. PARTLY FIXED in F2a: Logged data the plan needs is discarded.

(F2a: the report model's validated task is now the hagent TASK, with `default_task` as the fallback (DECISIONS.md). Reviews and pass probabilities are still dropped.)

HagentRepairer drops RunResult.reviews and the pass probabilities, so the reviewer-calibration analysis in plan 4.3 has no data to work from. It also discards the report model’s task, even though plan 3.2.12 says that model writes the task. That may be acceptable, but it isn’t documented.

## Tier 3: Statistics

### 15. Replay agreement is never measured.

agreement is a config input with no tool behind it. agreement_rate is only called by tests, and how a seed maps to the policy’s sampling seed is still TODO. The cache also hides non-determinism. The paired test is used only in Rung 1’s regression check, never in the router.

### 16. Fixed global anchors replace per-incident ones.

Incident.reference_rate is never read for routing, even though its docstring says Stage 0 compares against it. The failure anchor is treated as an exact point, ignoring its own uncertainty.

### 17. Rung 1 checks absolute levels, not improvement.

A patch on a target that was already restored passes. A patch that changes nothing on a partial target comes out “ambiguous” and spends the Rung 2 call.

### 18. The second-cause flag is unreliable.

When the nominal run reads no_effect at 10 seeds, the router goes to Branch G only, Branch C never runs, and the flag can’t be set. This happened in 32/50 salts in D-4. Keying Rung 1 on the branch rescues the loop, but you can’t trust the flag.

### 19. Cache problems.

- The cache key doesn’t include which backend produced the episode, so two backends sharing one cache return each other’s results.
- SceneSpec.key() can collide if a factor value contains ; or =.
- 1 and 1.0 in a config give different keys.
- hash(CodeRef) raises an error.
- cache_hits counts re-reads during extension, so the test identity simulated == used − hits only holds when nothing was extended.

## Tier 4: Decision-table behaviors to decide on deliberately

### 20. Row 1 only looks at Stage 0 and variable staleness.

A restored Stage 0 still stops the router even with render_physics_broken and contract_violation evidence present (F1).

### 21. An inconclusive G3 is handled inconsistently.

In Branch G it gives code by elimination; in Branch C it gives row 10. The Branch G case probably should ask for more seeds.

### 22. Two partial restores under a code verdict set no second-cause flag (X3).

### 23. Row 2 reached through evidence attaches no proving replay, although the plan says to “report with the replay that proved it”.

### 24. Row-10 verdicts list evidence that min_strength filtered out.

### 25. “Interacting factors” contradicts itself.

Plan 6.5 says they “only fail together”; the row-7 reading says each factor alone gives a partial restore. Under the 6.5 reading, a real interacting case ends at row 10.

### 26. PARTLY FIXED in F1: G1 runs but decide never reads it.

Row 3 depends entirely on Stage 1 evidence, which combines badly with issue 1. (F1: Stage 1 now runs in route(), and a missing staleness check makes rows 1 and 3 undecided instead of silently false. decide still does not read G1: open.)

### 27. Some failures end at row 10 with “no row matched”: an undeclared scene change, or a failure at the reach phase.

## Tier 5: Stage 1, behavior and LLM layer details

### 28. FIXED in F1: Stage1Result.sources_run lists all five sources even when you passed in fewer checks.

### 29. FIXED in F1: If incident-mode traces are missing, staleness quietly checks only the clean half and records no skip.

### 30. PARTLY FIXED in F1: Every trace check reads CLEAN-mode traces by default, but nothing ensures clean traces are logged for each incident.

BehaviorConfig.mode also defaults to CLEAN, with no reason given. (Defaults kept and justified in DECISIONS.md; a missing log is now a recorded skip that makes its rows undecided. The production source serving Stage 0's clean logs is TODO(verify).) (F2a review: only partly fixed until #52 (F2b): nothing checks that the logs belong to the code under test, so after a patch a source can still serve the original code's logs.)

### 31. Nominal statistics must be per scene condition, which means no-fault runs for every shifted condition.

That costs a lot and isn’t planned for anywhere.

### 32. The config contract treats True and 1 as equal.

### 33. The channel-order and orientation checks can name the wrong fault, such as “GBR” for a one-channel shift or “vertical flip” for a shuffled profile.

That wrong location goes to hagent.

### 34. s1.action.gripper_tracking is never produced by any test.

### 35. The report validator misses absolute paths, extension-less paths, unbracketed ids and config-key fragments, and it accepts real-but-wrong reasons (H1–H6).

These holes matter more since F2a: the model's task is now the hagent TASK, not only report text. F5's to fix.

### 36. rung2.py hard-codes "C_restore:" instead of taking it from vocab.

## Tier 6: Tests that pass but don’t prove what they claim

### 37. test_every_factor_restore_runs... doesn’t detect an early stop: the mutation passed, because the restoring factor sorts last.

### 38. The confusion matrix measures sampling noise on hand-written evidence.

It is not router accuracy.

### 39. Some assertions are weaker than their test names suggest: sanity uses in rather than “only”, and the action test uses or.

One report test asserts the opposite of its name.

### 40. “Router and Rung 1 use the same anchors” is only a convention in the fixture, not enforced.

### 41. Untested: CLI exit code 3, the unpaired-test path, Rung 2 overriding a row-6 re-run, and row 10 → UNRESOLVED.

### 42. PARTLY FIXED in F1 and F2a: Step 7’s “done when” (the combined case running end to end) is met only through the fixture, the always-pass Rung 0 and ScriptedRepairer.

(F1: the evidence fixture is gone; the loop gets evidence from route()'s own checks on code-keyed synthetic logs. F2a: `tests/test_loop_hagent.py` runs the combined case with the real HagentRepairer on file-keyed mock rules and logs. The always-pass Rung 0 remains.)

## Tier 7: Infrastructure and process

### 43. FIXED in F0: hagent isn’t a declared dependency, so the documented pytest command fails at collection.

That breaks your own Master rule 6.

### 44. No live model has ever been run.

Step 1B is skipped and everything uses ScriptedModel. Whether a real model follows the one-bash-block format, uses run_rung1, emits PASS_PROBABILITY or respects the hypothesis framing is unknown.

### 45. Nothing runs the loop in production.

No script builds IncidentLoop with real parts, and HagentRepairer is only built in tests.

### 46. The build order doesn’t match the plan.

The plan’s Section 8 puts policy selection, the failure map and the pilot gate (steps 1–3) first, because they may change everything. You’ve built 7 prompt-steps of router on mocked rates while the pilot question (“does data even fix this failure?”) is still unanswered.

### 47. The numbering is confusing.

Prompt steps and plan Section 8 steps are numbered differently, which is why D-7 says “remove fixture in step 5” about something that is already built.

### 48. Small rule violations.

A few hard-coded literals break Master rule 5: the 0.5 margins, the 0.25 staleness text threshold and n_a=10. Module layout also differs from plan 3.2.14. Both are cosmetic.

## Found during F1

### 49. When the incident was observed in incident mode but its incident-mode logs are missing, route() ends at row 10 (row 1 undecided: staleness_variable not checked).

It could instead replay the failing seeds in incident mode (as G1 does) to obtain the staleness probe's incident half. Not implemented.

### 50. Config contract: a mapped key missing from the eval config is only a log warning ("not compared"), the same silent-skip pattern the staleness probe had.

Its signals are not reported as unchecked.

### 51. Issue 2 is now live: with the fixture gone, a config fix that hagent makes in a file keeps firing render_settings_differ / the instruction violation, because patched_code_ref keeps the base eval_config.

(duplicate of #2; still open after F2a)

The loop tests avoid this only because ScriptedRepairer returns a CodeRef with the fixed eval_config.

### 52. Nothing checks that the logs a TraceSource returns belong to incident.code_ref (EpisodeTrace has no code key).

After a patch, a source that returns the original incident's logs would keep reporting the fixed fault.

### 53. When route() stops at Stage 0 because row 1 is undecided, the verdict notes also list rows 2-9 as undecided for missing replays.

The notes are correct but noisy; the row-1 reason comes first.

### 54. scripts/confusion_matrix.py still feeds hand-written evidence straight to decide (see Tier 6); it never calls route(), so F1 does not change it.

(duplicate of #38)

## Found during F2a

### 55. hagent's patch is `git diff HEAD` without `--binary`, so a binary edit cannot be rebuilt.

The loop records such an attempt as `unrebuildable_patch` and tells the agent not to edit binary files. Fix: use `git diff --binary` in hagent (`GitHelpers.git_diff`).

### 56. hagent's patch depends on the host and on what the agent did with git.

It runs with the user's git config (e.g. `diff.noprefix` makes the patch fail to apply), and it is `git diff HEAD`, so a change the agent commits in the working copy is not in the patch. The workspace's own recipe (`ladder.workspace.cumulative_diff`) is immune to both, but `extend()` starts from hagent's patch.

## Found during the F0/F1/F2a review

### 57. run_rung1's config (`$RUNG1_CONFIG`) is readable by a local agent, and it leaks.

It contains the original checkout's path (`workspace.bases`) and the mock `FileContains` rules, which spell out the fix (e.g. `normalize_inputs(obs, self.norm_stats)`). Under LocalEnv the agent can `cat` it, which leaks the answer in mock evaluations and tells it where the original is. F5.

### 58. The agent can rewrite `vla.base` or `refs/vla/base` in its working copy.

That changes what its own `run_rung1` evaluates (`code_ref_of` reads both from the copy), not what the loop accepts (the loop names hagent's patch with `extend` on a fresh copy). The agent can only mislead itself. F5.

### 59. Checks that only partly ran still count as clean (top item for F2b).

- C_behavior "not judged" (too few grasp records, an inconsistent offset, a phase tie, no failing episodes) returns no evidence, and `route()` counts it as run (`router/router.py`, `_run_c_behavior`). Rows 4 (Branch C), 8 and 9 then evaluate FALSE instead of UNKNOWN, so a code bug located only by an implied-target offset can read as row 6, a vision gap, and money goes to data.
- The action contract's "dimensions / range / gripper sign not checked" (`router/stage1/action_contract.py`, no layout or limits in the training spec) and the observation contract's "no logged values for <x>; not checked" (`router/stage1/obs_contract.py`) only go to the log. Every router case runs with the action contract checking none of those three.

Same pattern as #50 (config contract). Fix with F2b's `PartialSkip`.

### 60. Remaining untested branches and hard-coded values from the review (F6).

- Untested: `RouterInputs` with only one of behavior source / nominal (`ValueError`); `PatchStore.put` collision; a base path that is not a directory; `FileContains.from_dict` with an unknown kind; `file_faults` without workspaces; `code_ref_of` outside a working copy (Rung 1 CLI exit 2); an invalid base label; a non-staleness partial skip in `router.unchecked_signals`.
- Hard-coded: the 12-hex digests (diff and tree); the truncation lengths 200 / 300 / 500 / 1000 in workspace, repair and loop messages; the fixed commit date; the `"staleness"` source special-cased in `router.unchecked_signals`; the rebuild-feedback texts in `ladder/loop.py`.

