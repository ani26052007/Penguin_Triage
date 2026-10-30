# Design decisions

This is my log of the choices I made while building the router and the verification ladder (2026-10-23 onwards), what I rejected and why, and what the mock runs taught me. It is written per build step. The GPU results that came later are in [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md); what is still open is in [FUTURE_PLANS_AND_IMPROVEMENTS.md](FUTURE_PLANS_AND_IMPROVEMENTS.md).

## 1. Foundations: schemas, statistics, backend, cache

**What I chose**

- `run(scene, code_ref, actor, seeds, mode)` takes an `actor` as well as the code. `ActorRef` is `{kind: policy|reference, checkpoint, pipeline: raw|wrapper}`, so reference-controller runs (G2 and C1 raw, G3 through the wrapper) and policy runs never share cache entries. The policy can never run RAW.
- Cache key = code, actor, scene, seed and mode. The code key hashes the code *and* the eval config, so a change to render settings or chunk length does not collide with the old entry.
- **When is a fix "restored"?** I use a middle rule: the interval must exclude the failure rate *and* the point rate must reach the low edge of the nominal band. If the point is below the edge but the interval reaches it, the answer is `inconclusive` (run more seeds). With Stage A at 10 seeds and Stage B at 50, failure rate 0.2 and band 0.7-0.9, the exact pass probabilities are 0.00 for a true rate of 0.2, 0.10 for 0.45 and 0.99 for 0.8.
- `INCONCLUSIVE` means "need more seeds". It is not a fourth verdict; the caller decides what it means at its seed cap.
- Statistics are pure stdlib: Wilson score intervals, exact McNemar for paired seeds, Fisher exact for unpaired ones. A paired comparison is only used when replay agreement is above a calibrated minimum (no default) and the seed sets match.
- Two looks (Stage A, then cumulative Stage B) share the error budget by a Bonferroni split. A partial result after Stage A goes to Stage B, and after Stage B to Rung 2. A partial fix is never a pass.
- The cache is in memory with an append-only JSONL file; the first write wins. Because a cache hides non-determinism, the replay-agreement tool must bypass it (tested).

**What I rejected**

- The plan's literal rule ("interval overlaps the band and excludes failure"): at 10 seeds it passed about 51% of patches whose true rate was 0.45.
- The strict rule (whole interval above the band edge): truly fixed patches (0.8) passed only 19% of the time.
- Putting the checkpoint inside the code reference: it mixes code identity with actor identity.
- scipy (stdlib is enough; scipy is only a cross-check in one test) and Clopper-Pearson intervals (too conservative at 10 seeds).

## 2. The decision table (`router/decide.py`)

**What I chose**

- `decide(evidence, replays, config) -> Verdict` is pure and order-independent (tested by shuffling 4000 random tables). It does no statistics itself: callers classify each replay first, and `decide` rejects unclassified replays and duplicate ids. Signal names and replay ids live in `router/vocab.py`.
- Every row evaluates to TRUE, FALSE or UNKNOWN (replay missing or inconclusive). The first TRUE row wins, **but if a higher row is UNKNOWN the verdict is row 10 (`undetermined`)** and names the replays to extend. A missing replay can never be read as "no effect". This is exactly what happened on the real GPU evidence: with no reference-controller replay (G2) the router refused to say `code`.
- A clean-mode pass stops at row 1 only when there is no `staleness_variable` evidence; with it, the verdict is row 3 (infrastructure).
- A located code cause (contract violation, constant delay, implied-target offset, or G3 failing while G2 passes) is reported as located; otherwise it is "by elimination".
- Render-setting differences get their own signal, `render_settings_differ`. If every undeclared difference were a contract violation, row 4 would always win and row 5 could never be reached.
- Partial restores on rows 3, 5, 6 and 7 set `likely_second_cause`. A full restore next to a code verdict is not flagged, because that is what a condition-specific code bug looks like.
- Row 6 needs exactly one factor to restore. If two or more restore, the verdict is row 10: picking one would make the answer depend on test order.
- A reference controller that only partly succeeds is not trusted.
- The only tuning knob is `min_strength` (default 0); its value has to come from calibration.

**What I rejected**

- Flagging a second cause whenever any lower row also matches (it flags every condition-specific code bug).
- Treating a render difference as a code violation (row 5 dead).
- Picking the highest-rate or first factor when several restore.
- Treating a missing replay as FALSE (hides incomplete tables).

**What the mock showed**

- Mock confusion matrix (30 seeds, 10 repeats): 97% correct, 0.9% undetermined. The remaining errors are sampling noise: a no-effect restore at a true rate of 0.15 reads as partial about 3% of the time, which turned 2 of 10 `action_gap` repeats into `vision_factor`. Thresholds need calibrating on no-fault incidents.
- The plan says interacting factors "only fail together", but row 7 assumes each factor alone restores partly. Under the first reading a real interacting case ends at row 10. I followed the row 7 reading in the mock.

## 3. Replays and the orchestrator (`router/replay.py`, `stage0.py`, `stage2.py`, `branches/`)

**What I chose**

- Non-replay evidence enters through `route(..., extra_evidence=...)`.
- One replay runner per incident. The seed set is the failing seeds, padded with fresh ones, and every replay uses a prefix of it through the cache.
- Seed caps: start at 10; Stage 0 extends to 20, other replays to 30. Over 50 random salts this split gave 485/500 correct routes against 476/500 with every cap at 20.
- Stage 0 stops the router if it is row 1, or if it is still inconclusive at 20 seeds (then nothing later can change the verdict).
- Branches follow Stage 2: clearly failed means Branch C, clearly working means Branch G, degraded or inconclusive means both. Running C when degraded is what lets a partial vision restore flag a second cause.
- G1 (policy, incident scene, incident mode) is recorded for the report but not read by `decide`.
- `C_all` runs only with two or more factors and no single full restore; `C_render` runs only with `render_settings_differ` evidence.

**What I rejected**

- Every cap at 20 (`vision_interacting` 38/50, `code_partial_offset` 43/50).
- Branch C only when degraded (no second-cause flag on combined cases).
- A second copy of the row 1 rule inside Stage 0 (two definitions of "no failure" could drift apart).

**What the mock showed**

- Episodes per incident: 10-20 with no fault, 30-90 otherwise.
- Combined code-plus-lighting case: the label was right 50/50, but the second-cause flag was missing in 32/50, because at 10 seeds a 0.35 nominal often reads as "clearly failed" and Branch C is skipped. The re-run after the code fix is what catches the second cause.

## 4. Stage 1 checks (`router/stage1/`)

**What I chose**

- Checks read traces through a `TraceSource` interface. A missing trace raises `TraceUnavailable` and the check is recorded as skipped, so "not run" is never read as "checked, found nothing".
- I defined the minimum trace schema the checks need (per-frame stats, observation/applied step pairs, raw actions, commanded vs actual joints and gripper, sim health), since the plan left it open.
- Strength is a z-score against nominal statistics mapped to 0-1 (exactly 0.5 at the firing threshold; 1.0 only for exact violations such as a config mismatch).
- The unit is the episode: each metric is a median over frames, and the check scores the median over episodes, so one bad episode in eight does not fire.
- Render statistics and policy-input statistics are kept apart. The sanity check reads only render stats and the observation contract only policy-input stats, so a preprocessing bug routes as code, not as a broken simulator.
- Channel order uses the opponent vector (channel means minus their mean), which does not depend on brightness. Orientation uses row and column profile correlations. On the GPU box these two flagged BUG-01 (RGB/BGR swap) and BUG-02 (image rotated 180°).
- Staleness subtracts the action-chunk offset so the chunking sawtooth is not counted as delay. A constant extra delay in clean mode is `staleness_constant` (controller latency); delay that appears only in incident mode is `staleness_variable` (policy server). On the GPU box the constant half flagged BUG-09 at about 2 steps.
- The config contract needs no traces: it compares the eval config to the training spec through a key mapping, with declared scene factors allowed to change their fields. On the GPU box it flagged BUG-05, BUG-06 and BUG-07.

**What I rejected**

- Pooling frames across episodes (one bad episode would dominate).
- Dividing channel means by brightness (unstable in dark scenes).
- Using jitter to tell server delay from code delay (a server can be uniformly slow; presence in clean mode is what separates them).

**What I learned**

- Nominal statistics must come from the same scene condition as the incident. On the real dark-lighting cell the observation check fired `value_range` and `channel_order` with no code bug present, so contract evidence alone cannot separate a vision gap from a code bug. Only the one-factor-restore replay can.

## 5. Behaviour analysis and the model layer (`router/behavior.py`, `report.py`, `llm.py`)

**What I chose**

- Behaviour analysis (implied target and failure phase) is its own step, reading its own `BehaviorTrace`. It emits exactly one of offset, ok or wrong-object, or nothing when the data cannot say, so it never creates a self-conflicting row.
- The failure phase is the first unfinished subtask of each failing episode, reported only if one phase covers at least 60% of failures.
- Models are reached through hagent's client. `query_json` asks for strict JSON, retries once with the rejection reason, then gives up.
- **The report writer can never change the verdict.** Its reply must be exactly `{summary, evidence_ids, task, localization}`; any other key (such as `route` or `label`) rejects it. Evidence ids must exist, and named files and symbols must be in the repo index or the evidence. A second bad reply falls back to a report rendered by code from the evidence, with no model text.
- The hagent prompt is rendered by code; the model supplies only the task text.
- A structural test (`test_no_model_in_routing.py`) checks that the decision code imports no model layer.

**What I rejected**

- Letting the model write the whole hagent prompt (it could contradict the verdict).
- Repairing a bad reply by dropping the bad parts (a model that cites something that does not exist cannot be trusted on the rest).
- A default or empty config mapping when the model fails (the contract check would silently miss violations).

**Known limit:** the validator catches names that do not exist, not explanations that are wrong but well-formed.

## 6. Verification ladder and incident loop (`ladder/`)

**What I chose**

- **Rung 1 validates the scene the router's branch names.** Branch G validates on the nominal scene, Branch C on the incident scene; the other scene is a regression check. Outcomes: `pass`; `pass_with_residual` (target restored, other scene unchanged), which re-runs the router and never goes back to hagent; `fail`, which goes back to hagent; and `ambiguous`, which goes to Rung 2.
- Rung 2 is one model call on the router's re-run. A `data_gap` answer stands only when the router itself found a one-factor or all-factors restore (a pure gate function); otherwise the result is `unclear`. It never raises.
- The loop: route, act on the destination, and after a fix re-route the patched code. Caps are 3 router passes, 3 repair attempts and 1 Rung 2 call per incident; hitting any cap ends the incident `UNRESOLVED` with everything attached.
- Code before vision comes from the table's precedence plus acting on one verdict per pass.
- The repairer returns a new code reference and works on a throwaway copy, so the checkout is never edited and every attempt starts from the code under repair.
- Rung 0 (diff sanity, lint, tests, smoke run, then the reviewer) stops at the first failure, so neither the simulator nor the big model is spent on a broken patch.

**What I rejected**

- Choosing the Rung 1 scene from the second-cause flag (an undetected second cause would make hagent loop forever).
- Always validating on the incident scene (the combined case could never pass) or always on nominal (a condition-specific fix would pass without touching the broken scene).
- A model or loose string match in the data-gap gate, and retrying Rung 2 after a gate rejection.
- Running the reviewer both in Rung 0 and in hagent's final review (two big calls for one patch).

**What the mock showed**

- Combined case on the mock (50 salts): hagent was called exactly once in 50/50 runs; handed off to the data ladder in 43/50, undetermined in 5 and wrongly closed in 2. All 7 misses come from Stage 0 noise, not from the loop.
- A code verdict that persists with a lighting residual ends the incident `UNRESOLVED` within the caps. It does not loop.
