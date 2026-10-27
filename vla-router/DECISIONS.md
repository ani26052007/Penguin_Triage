# Decisions

## Step 2: Foundations (schemas, ladder/stats, backend, cache) — 2026-10-23

**Chose**
- `run(scene, code_ref, actor, seeds, mode)`: added an `actor` argument (user-confirmed). `ActorRef = {kind: policy|reference, checkpoint, pipeline: raw|wrapper}`, so the reference controller (G2/C1 raw, G3 through the wrapper) and policy runs never share cache entries. The policy cannot be RAW.
- Cache key = (`CodeRef.key()`, `ActorRef.key()`, `SceneSpec.key()`, seed, mode). `CodeRef.key()` hashes the code hash *together with the eval config*, so render-settings or chunk-length changes don't collide. This extends plan 3.2.2's "code hash".
- "Restored" uses the **middle rule** (user-confirmed): excludes the failure rate AND point rate >= band low edge. If the point rate is below the edge but the interval reaches it, the result is `inconclusive` (extend seeds). Partial and no_effect are as in plan 3.2.9. Exact P(pass) for Stage A 10 + Stage B 50, failure 0.2, band 0.7–0.9: true 0.2 -> 0.00, true 0.45 -> 0.10, true 0.8 -> 0.99.
- `Outcome.INCONCLUSIVE` is a "need more seeds" signal, not a fourth verdict. Callers decide what it means at their seed cap.
- Wilson score interval; exact McNemar (paired, on discordant seeds) and Fisher exact (unpaired). Pure stdlib. `compare()` pairs only if agreement >= `min_agreement` (required argument, no default: it is calibrated) and the seed sets match.
- Early look: Bonferroni split of `alpha_total` over Stage A / Stage B intervals (`stage_a_fraction` tunes Stage A strictness). Stage B is cumulative (reuses Stage A seeds). `staged_operating_characteristics` computes exact pass/fail/ambiguous rates and expected seeds, for the Section 7 report and for calibration.
- After Stage A, a partial result goes to Stage B, and after Stage B to Rung 2 (ambiguous). A partial fix is never a pass.
- The failure anchor is treated as a point value (its own Stage 0 uncertainty is ignored).
- Cache: in-memory with append-only JSONL; the first put wins. The replay-agreement tool must call `CachedBackend.backend` directly, since the cache hides non-determinism (tested).
- File placement: cache in `router/prereq/` (per 3.2.14); backend in `ladder/` (shared by router, Rung 1, data pipeline); `Outcome` enum in `router/schemas`.

**Rejected**
- The literal plan rule ("interval overlaps band and excludes failure"): at 10 seeds it passed ~51% of true-45% patches.
- The strict rule (whole interval >= band low edge): truly fixed patches (0.8) passed only 19%.
- Keeping checkpoint inside `code_ref`: mixes code identity with actor identity.
- scipy as a dependency: stdlib is enough; scipy is only used in one test as a cross-check.
- Clopper-Pearson intervals: too conservative at 10 seeds; Wilson is the standard small-n choice.
- An Isaac backend implementation: left as `IsaacBackend` raising NotImplementedError, with TODO(verify) notes.

## Step 3: Decision logic (router/decide.py, tests, scripts/confusion_matrix.py) — 2026-10-23

**Chose**
- `decide(evidence, replays, config) -> Verdict` is pure and order-independent (tested by shuffling 4000 random tables). Callers classify each replay's `outcome` with `ladder.stats` first; `decide` rejects unclassified replays, duplicate ids, and does no statistics itself. Signal names and replay ids are the contract with later steps and live in `router/vocab.py`.
- Each row is evaluated TRUE / FALSE / UNKNOWN (UNKNOWN = replay missing or INCONCLUSIVE). First TRUE row wins, but if a higher-precedence row is UNKNOWN the verdict is row 10 and names the replays to extend. Evidence conflict at the winning row also gives row 10. A missing or inconclusive replay can therefore never be silently read as "no effect".
- Row 1 vs row 3 (user-confirmed): a clean-mode pass stops at row 1 only if there is no `staleness_variable` evidence; with it, row 3 (infra). Stage 0 stays one clean-mode replay.
- Row 4 confidence: `by_elimination=False` for a located cause (contract violation, constant delay, implied-target offset, or G3 failing while G2 passes, location `action_side`). `by_elimination=True` for Branch G with none of these; location `observation_side_or_policy_loading` if G2 and G3 both pass. No new field on `Verdict`.
- Render-settings differences are signal `render_settings_differ`, separate from `contract_violation`. plan 3.2.4 calls any undeclared difference a contract violation, which would make row 4 win every time and row 5 unreachable.
- Partial restores: rows 3, 5, 6, 7 accept PARTIAL and set `likely_second_cause`. For a code/infra winner, `likely_second_cause` is set when a lower render/vision row restores only partly. A FULL restore alongside a code verdict is not flagged, since that is what a condition-specific code bug looks like (3.2.1 #2). If the check was inconclusive the notes say so.
- Row 6: exactly one single factor must restore. Two or more restoring (fully or partly) give row 10 (picking one would be test-order dependence). A single partial plus a full `C_all` is row 7 (the factors add up). Row 7 sets `Verdict.factor` to the factor names joined by `;`.
- Reference controller "fails" = RESTORED not reached (partial or no effect); a degraded reference is not trusted.
- Required replays follow the branch: G2 in Branch G, C1 in Branch C, both if the branch is unknown; one `C_restore:<f>` per factor of stage0's scene; `C_all` when two or more factors; `C_render` when render settings differ.
- `DecideConfig.min_strength` (default 0) is the only knob; the cut-off is a step-6 calibration output.
- The confusion-matrix script is a fixture: a fixed superset of replays (no early stop, no extend-seeds loop) and hand-written evidence per fault. Real runners and Stage 1 checks are step 6. Mock partial rates are 0.35, away from the 30-seed boundary; at 0.45 about half the repeats came back inconclusive.

**Rejected**
- Flagging a second cause whenever any lower row also matches: flags every condition-specific code bug.
- Treating a render difference as a code contract violation (row 5 dead).
- Picking the highest-rate factor, or the first, when several restore.
- Running Stage 0 in both modes (second option offered for row 1 vs 3).
- Treating a missing replay as FALSE: hides incomplete tables.

**Findings to carry into step 6**
- Mock matrix (30 seeds, 10 repeats): 97% correct, 0.9% undetermined. The remaining errors are noise: a no-effect restore at true rate 0.15 reads as PARTIAL about 3% of the time (k>=9 of 30), and a partial single restore wins row 6. In `action_gap` this turned 2 of 10 repeats into `vision_factor` (flagged). This is the multiple-replay false-verdict issue (3.2.9); thresholds or confidence need calibrating on no-fault incidents.
- plan 6.5 says interacting factors "only fail together", but row 7 says "only the all-factors restore works". Those differ: if failure needs both factors, restoring either one fixes it and `decide` returns row 10 (two restoring factors). The mock follows the row 7 reading (each factor alone is a partial restore).

## Step 4: Replay stages and orchestrator (router/replay.py, stage0.py, stage2.py, branches/, router.py) — 2026-10-23

**Chose**
- Non-replay evidence comes in as `route(..., extra_evidence=...)` (user-confirmed, option b). Stage 1 and C_behavior are not built; tests pass the Step 3 hand-written records. `code_condition_specific` gets no evidence and is a strict `xfail` (user-confirmed): replays alone route it to `vision_factor`, which is plan 3.2.1 #2's channel-swap trap.
- One `ReplayRunner` per incident. Seed set = failing seeds (deduplicated, in order), padded with fresh seeds. Every replay uses a prefix of it and goes through `CachedBackend`. A plain backend is wrapped in a fresh cache.
- Seed caps: start at 10. Stage 0 extends to 20 (plan 3.2.4); other replays extend to 30 (top of 3.2.9's "20-30"). Over 50 salts, routable classes: 476/500 correct with every cap at 20, 485/500 with this split.
- Stage 0 stop: `decide` on Stage 0 alone. Stop if row 1, or if Stage 0 is still inconclusive at 20 (row 1 undecided forces row 10, so more replays cannot change the verdict). One definition of "no significant failure", including the staleness exception.
- Branches by Stage 2 outcome: C → C only; G → G only; degraded → G and C; inconclusive → G and C (`decide` needs both G2 and C1). Running C when degraded is what lets a partial vision restore set `likely_second_cause` (plan 6.5 "Combined"); G's code verdict still wins by precedence.
- G1 = policy, incident scene, INCIDENT mode, new id `G1` in `vocab`. `decide` does not read it (row 3 is stage0 + staleness); it is recorded for the report. Run in Branch G, and in Branch C when clean-mode Stage 0 passes but the incident was seen in incident mode. Skipped when the incident was observed in clean mode.
- C_all runs only with 2+ factors and no single RESTORED restore (plan 3.2.7). C_render runs only with `render_settings_differ` evidence. Its code ref is the eval config with `render_config_key` (default `"render"`) set to `training_spec.render_settings`. If the training spec has none, it is skipped with a note and `decide` returns row 10.
- `Incident.nominal_scene` (optional, default `SceneSpec(scene.nominal)`): the failure-map nominal Stage 2 runs. Needed for `action_gap`, where the shifted scene's base differs from the task nominal (Step 3's `stage2_scene`).
- `RouterResult` logs per incident: `episodes_simulated` (cache misses), `episodes_used`, `cache_hits`, extended replays, branch, stages. One INFO line on logger `router`.
- Test views: `QuotaBackend` (exact successes per 10 seeds) checks logic exactly. Salted `MockBackend` (10 salts) checks noise: exact for 5 noise-free classes, >=8/10 for the rest.

**Rejected**
- Every cap at 20: `vision_interacting` 38/50 and `code_partial_offset` 43/50 (0.35 rates read inconclusive at 20).
- Branch C only when degraded: no C restores means no second-cause flag on combined cases.
- Re-implementing the row 1 rule in Stage 0: two definitions of "no failure" could drift apart.
- Gating C_render by comparing configs in the router: that comparison is the Stage 1 config contract.
- Using G1 as row 3 evidence: would change `decide`, which is outside this step.

**Findings**
- Episodes per incident (50 salts, mock): 10-20 for no fault, 30-90 otherwise (means 30-60). That is under plan 3.2.9's 60-150 because the mock's reference controller and restores are rarely ambiguous.
- `combined_code_plus_light`: the label is right 50/50, but the second-cause flag is missing in 32/50. At 10 seeds a 0.35 nominal reads NO_EFFECT (k<=3) about half the time, so Stage 2 says "clearly failed", Branch C is skipped and nothing flags the vision part. The re-run after the code fix (step 7) is what catches it.
- `code_partial_offset`: 7/50 undetermined, all from Stage 0 inconclusive at its 20-seed cap (true 0.35, k>=10 of 20).
- `action_gap`: 4/50 `vision_factor`, the same 0.15-reads-as-partial noise as in Step 3.

## Step 5: Stage 1 checks (router/stage1/, router/prereq/trace.py) — 2026-10-23

**Chose**
- Checks read traces through a `TraceSource` interface (`load(incident, mode)`), not `EpisodeResult.logs`: the logs are not tied to an incident, and this keeps the plan's `run(incident) -> list[Evidence]` signature. Missing traces raise `TraceUnavailable`; the registry records the check as skipped, so "not run" is never read as "checked, nothing found". `InMemoryTraceSource` is the mock.
- The plan leaves the trace schema undefined, so `router/prereq/trace.py` defines the minimum the five checks need (per-frame stats, observation/applied step pairs, raw actions, commanded vs actual joints and gripper, sim health). Real log format and units are TODO(verify).
- Strength = `NominalStats.deviation`: z against nominal mean/std (with a `min_std` floor, since a constant metric has zero spread), mapped by r²/(1+r²) with r = z/`fire_z`. Exactly 0.5 at the threshold, never 1.0 except for exact violations (config mismatch, resolution, action dims). Nominal stats are plain data (`to_dict/from_dict`). `fire_z=4.0` and the floors in `synthetic.FLOORS` are placeholders; step 6 calibrates them.
- Unit = episode; each metric is the median over frames, and a check scores the median over episodes. One bad episode in eight does not fire.
- Render and policy-input stats are separate (`FrameSample.render` / `.policy_input`). Sanity reads only render stats, the observation contract only policy-input stats, so a normalization or channel bug in preprocessing routes as code (row 4), not as a broken sim (row 2).
- Channel order uses the opponent vector `means - mean(means)`, which does not depend on brightness. Orientation uses row/column profile correlation against a reference. Open-loop replay (training frames through current vs reference preprocessing) is an `OpenLoopProbe` interface; the real one is a stub.
- Action chunking: `extra_delay = applied - obs - chunk_offset`, so the sawtooth is not counted as delay.
- Staleness: constant extra delay in clean mode = `SIG_STALENESS_CONSTANT` (location `controller.latency`). Delay that appears only in incident mode = `SIG_STALENESS_VARIABLE` (the vocab name; location `policy_server`). The discriminator is presence in clean mode, not jitter. Incident traces are read only if the failure was observed in incident mode.
- Config contract needs no traces: eval config vs training spec through a `ConfigMapping` (spec field -> eval-config key, plus `factor_covers`: declared scene factor -> field it legitimately changes). Tolerances are data. Render-settings differences emit `SIG_RENDER_SETTINGS_DIFFER` (row 5), everything else `SIG_CONTRACT_VIOLATION`. Instruction templates match with `{...}` as a wildcard. Location is the mapped field; the leaf path is in `detail`.
- Contract and sanity checks default to clean-mode traces. Registry: `run_stage1(incident, checks)` validates sources against `EVIDENCE_SOURCES`, one check per source, source and id uniqueness. `route()` is unchanged; Stage 1 output goes in through `extra_evidence`.
- Mock side: `router/stage1/synthetic.py` (trace generator with one `Faults` field per planted fault, nominal built with the checks' own metric functions).

**Rejected**
- Reading `EpisodeResult.logs`: no tie to the incident.
- Pooling frames across episodes: one bad episode would dominate; also hides how many episodes show the fault.
- Dividing channel means by brightness for the channel-order test: unstable on dark scenes.
- Treating staleness jitter as the server/code discriminator: a server can be uniformly slow; presence in clean mode is what separates them.
- One shared frame-stats record: would let a normalization bug fire the sanity check.

**Findings**
- Stage 1 evidence changes the Step 4 xfail case: with the instruction-template bug found by the config contract, `route()` returns CODE located at `eval_config.env.instruction`; without it the same incident returns `vision_factor` (tested; the strict xfail itself was not touched).
- Nominal stats must be calibrated on the same scene condition as the incident (declared lighting/materials move brightness and range metrics). A near-grey scene cannot reveal a channel swap.
- Constant-delay strength at the placeholder settings is 0.9 for a 3-step delay (floor 0.25); real values depend on step 6 calibration.

## Step 6: Behavior analysis + report layer (router/behavior.py, report.py, llm.py, config_mapper.py, factor_search.py) — 2026-10-23

**Chose**
- C_behavior is a separate `run_behavior(incident, source, nominal)`; its evidence goes in through `route(..., extra_evidence=...)`. It is not a sixth Stage 1 source (user-confirmed), so `registry.py` and its test are untouched. It reads a new `BehaviorTrace` / `BehaviorSource` (mirrors `TraceSource`, raises `TraceUnavailable`), not `EpisodeTrace`, so the settled Stage 1 trace format does not change. Position only, no orientation.
- Implied target over failing episodes: bias = |mean(ee - target)|, scatter = RMS about that mean, wrong-object share. Strength = `NominalStats.deviation`, like Stage 1. Exactly one of offset / ok / wrong is emitted, or none when the data cannot say (too few grasps, bias fires but is not consistent), so this module never creates an ok + wrong row 10 conflict. Offset needs bias to fire and bias/RMS >= 0.9; large scatter or wrong object is `implied_target_wrong`. `implied_target_ok` has no deviation, so its strength is 1 - the largest metric strength (> 0.5).
- Failure phase = first incomplete subtask of each failing episode, modal phase if unique and >= 60% of failures. Strength is that share, which is categorical and not calibrated against nominal (stated in the evidence text). A "reach" failure emits `failure_phase:reach`, which `decide` does not count as action-gap evidence.
- LLM client is hagent's (`Model`, `OpenAICompatModel`, `ScriptedModel`, `ModelConfig`). Router-side extras in `router/llm.py`: `query_json` (strict JSON, one retry with the rejection reason, then `LLMRetriesExhausted`) and its own TOML (`router_models.example.toml`: `[report]`, `[mapper]`, `[factor_search]`), because hagent's `settings_from_dict` rejects unknown sections.
- Report writer returns a separate `Report` and never touches the frozen `Verdict`. Reply must be exactly `{summary, evidence_ids, task, localization}`; any other key (`route`, `label`, ...) rejects the reply. Evidence ids must exist; localization files/symbols must be in the repo index (`RepoIndex.from_dir`, `ast` for symbols) or in the evidence; config keys must be in the evidence; path-like tokens and `[id]` tokens in the prose get the same check. A rejected reply gets one retry; a second bad reply or a provider error gives the fallback: evidence ranked by strength, rendered by code with no model text.
- The hagent prompt is the Step 1A template rendered by code: verdict line, repro, rollout summary and repo path come from the Verdict and `ReportContext`; the model supplies only `task`; the location block is built from validated localization. Built only for hagent / config_fix destinations.
- Config-key mapper: model proposes `ConfigMapping.keys`; every path must exist in the eval config, no path twice or nested. Written to a JSON file with a `structure_key` (hash of names, not values), the keys, `approved`, and a side-by-side `review` block. Later calls read the file and do not call the model; hand edits are re-validated. A changed structure raises `StaleMapping`, even for an approved file. No fallback mapping on model failure.
- Factor search is an interface plus a stub (`VisionFactorSearch.propose`): row 9 only, OpenAI-style `image_url` parts, bad replies give an empty result, never invented factors. Running the replays for candidates is not done here.
- Structural test (`test_no_model_in_routing.py`): decide / router / stage code / behavior do not import the LLM layer or hagent, and the LLM modules do not import decide or the router.

**Rejected**
- Adding behavior as a sixth Stage 1 source: it reads different logs and is not log-only in the Stage 1 sense; would have changed the registry.
- Per-episode implied-target metrics: a consistent offset only shows in the group mean.
- Letting the model write the whole hagent prompt: it could restate or contradict the verdict, or leak text outside the evidence.
- Repairing a bad reply (dropping the bad id or file and keeping the rest): a model that cites something that does not exist cannot be trusted on the rest.
- Checking identifiers in prose: `np.clip()` and similar library calls would be false rejections. Only structured names and path-like tokens are checked.
- Falling back to a default or empty config mapping: the contract check would silently miss violations.
- Overwriting an approved mapping when the config structure changes.

**Findings**
- `NominalStats` for C_behavior must be built from groups of no-fault episodes about the size of the incident's failing set; a bias from few episodes is noisier. Placeholders to calibrate (step 6 of the build order): `fire_z`, `min_consistency` 0.9, `min_phase_share` 0.6, `min_episodes` 3.
- `llm.py` uses `tomllib`, and hagent needs Python >= 3.11; the router `pyproject.toml` still says >= 3.10.
- Known limit: a model can still write a wrong but well-formed explanation (a real file, a real id, wrong reason). The checks catch things that do not exist, not things that are wrong.

## Step 7: Verification ladder + incident loop (ladder/rung0.py, rung1.py, rung2.py, repair.py, loop.py, ladder/bin/run_rung1) — 2026-10-23

**Chose**
- **Rung 1 validates the scene the router's BRANCH names, not the flags** (user decision). Branch G (incl. degraded): target = nominal scene. Branch C with a code/config verdict: target = incident scene. Both scenes always run, before (code under repair) and after (patch), on the router's seeds in CLEAN mode, through the router's cache; the non-target scene is a regression check. Outcomes: `pass` (target restored, other restored or not broken before), `pass_with_residual` (target restored, other not restored but not worse) -> router re-run, never back to hagent; `fail` (target at the broken baseline, or any regression) -> hagent; `ambiguous` after Stage B -> Rung 2. Target decision is `ladder.stats.staged_decision` (Stage A then cumulative Stage B, Bonferroni alpha split).
- Regression = `compare(...).clearly_worse` on the other scene (paired/unpaired by the calibrated agreement, alpha of the current stage), or restored before and clearly below the band after. A non-target scene that was never broken and is still not clearly worse counts as restored.
- **Gap in the user's table, filled:** target restored and the other scene *unchanged* and not restored is `pass_with_residual` (improved is recorded as a flag). It cannot be a regression, and sending it to hagent is the infinite loop the branch rule exists to prevent.
- Rung 1 `agreement` and `min_agreement` are required config values (calibration outputs, no defaults, like the router's anchors). Branch UNKNOWN has no target: `rung1_target` raises and the loop ends the incident UNRESOLVED instead of guessing.
- `run_rung1 --seeds N` is `python -m ladder.rung1`, wrapped by `ladder/bin/run_rung1` for hagent's `[[commands]]` (tested through `LocalEnv.install_commands`). Config is one JSON (`--config` or `$RUNG1_CONFIG`: incident, branch, thresholds, backend; `after` defaults to incident code + hash of the working-tree diff, the same recipe as the repairer). Prints at most ~6 lines: pass rate, interval, decision, regression state, top error mode. Exit 0 pass/residual, 1 fail, 3 ambiguous, 2 error. `--seeds N` is the cumulative Stage B count; Stage A is min(10, N//2).
- **Rung 2** (`ladder/rung2.py`, LLM side): one `query_json` call on a `Rung2Input` the loop builds from the router RE-RUN (verdict, replay table, evidence, Rung 1 text). Strict keys `{label, reason, evidence_ids}`, ids must exist, one retry for a malformed reply only. The gate `data_gap_allowed(verdict)` is a pure function: a `data_gap` claim stands only for router row 6/7 with a one-factor or all-factors restore attached; otherwise the result is `unclear`, `accepted=False`, the model's label is kept in `model_label` for the log. A gate rejection is not a retry (one call). No model, provider error or two bad replies -> `unclear`, never raises. Every decision is logged and returned.
- **Loop** (`ladder/loop.py`): route -> act on the destination -> after a fix re-run the router on the patched code. `STOP` closes, `REPORT` is terminal (row 10 is UNRESOLVED, rows 2/3/8/9 REPORTED), `DATA_LADDER` hands off to a `DataLadder` stub, `HAGENT`/`CONFIG_FIX` go to a `Repairer` (config edit is a hagent task, then Rung 1, per 3.2.10). Caps: 3 router passes, 3 repair attempts, 1 Rung 2 call per incident; any cap hit = UNRESOLVED with the passes, repairs and last router result attached. Rung 1 fail, Rung 0 fail and a no-change patch all go back to the repairer with the failure text as feedback. After an ambiguous Rung 1 the router is re-run first, then Rung 2 reads that re-run (skipped if the re-run is row 1: closed). Rung 2 `code` -> repair again; `data_gap` (gate-accepted) -> hand-off; `unclear` -> UNRESOLVED.
- **Code before vision** is the decision table's precedence (row 4 before 6-7) plus the loop acting on one verdict per pass; a vision verdict that carries `likely_second_cause` is handed off with `partial=True` for the data side to see.
- **Repairer interface returns a new `CodeRef`** (user-approved). `HagentRepairer` renders the Step 6 prompt (`write_report` localization, `build_hagent_prompt`, failure feedback appended to the task) and calls `solve()` on a throwaway COPY of the repo (local backend), so the checkout is never edited and each attempt starts from the code under repair. Patched ref = `<base hash>+<sha1(patch)[:12]>`.
- **Rung 0** (`ladder/rung0.py`): diff sanity (non-empty, size cap, forbidden prefixes, touches a located file), then lint, tests, smoke run through `env.execute`, then hagent's `helpers.review` last (fails open; `pass_probability` logged only). Stops at the first failure so neither the simulator nor the big model is spent on a broken patch. The loop takes Rung 0 as an optional hook.
- **FIXTURE TO REMOVE IN STEP 5:** `tests/ladder_cases.py::evidence_by_code` is the per-CodeRef `EvidenceProvider` mock. Stage 1 is not wired into `route()`, so the loop asks an `EvidenceProvider` (`Incident -> Evidence`) per pass; the mock returns the hand-written Stage 1 record for the original code and nothing for the patched code (the contract violation is gone). When Stage 1 is wired into `route()`, delete the fixture and the `evidence_provider` argument.
- Structure test `tests/test_ladder_structure.py` extends the no-model-in-routing rule: `rung1.py` imports no LLM layer / hagent / Rung 2; `rung2.py` imports none of the decision code; `loop.py` imports neither hagent nor the LLM layer (models arrive as injected callables).

**Rejected**
- Keying the Rung 1 scene on `likely_second_cause` / `multi_cause_flag`: an undetected second cause leaves them False, the patch is judged on a scene it cannot fully fix, and hagent loops forever.
- Always validating on the incident scene: the combined case could never pass (lighting residual); always on nominal: a Branch C condition-specific fix would pass without touching the broken scene.
- Treating an unchanged, unrestored other scene as `fail`.
- A model (or a looser string match) in the data-gap gate; retrying the Rung 2 call on a gate rejection (a second big call for the same question).
- Running the reviewer in Rung 0 and again in hagent's `final_review` (two big calls for one patch): pass `reviewer=None` when `solve()` ran with `final_review`.
- Editing the repo in place for repeated attempts (attempt 2 would start from attempt 1's patch).
- Writing a model report for REPORT destinations inside the loop: the last router result (verdict, replay table, evidence) is attached to `LoopResult`; `write_report` stays a separate call.

**Findings**
- Combined case on the salted `MockBackend` (50 salts, 10-30 seeds): hagent called exactly once in 50/50; Rung 1 `pass_with_residual` 48, `pass` 2; hand-off to the data ladder 43/50, `undetermined` 5 (the re-run's Stage 0 on the lit scene, true rate 0.35, is still inconclusive at its 20-seed cap), and 2 wrongly `closed` (that Stage 0 read as restored: row 1). Both are router Stage 0 noise from Step 4, not the loop. The severe variant (nominal reads clearly failed, flag never raised) gives the same 43/5/2.
- A code verdict on Branch C that persists after a fix with a lighting residual makes Rung 1 target the lit scene, which no code patch can recover: the patch is ambiguous, Rung 2 is called, the incident ends UNRESOLVED (bounded by the caps; tested). It does not loop.
- A patch that changes nothing on a target that is only PARTIAL (not at the failure anchor) is `ambiguous`, not `fail`, because the anchors are absolute (3.2.9). It spends the Rung 2 call. A patch whose `CodeRef` is unchanged never reaches Rung 1.
- `ladder.rung1.top_error_mode` reads `failure_phase` / `error_mode` from the episode logs; the mock logs have neither, so it prints "unknown".

**TODO(verify)**
- Real `IsaacBackend` (Rung 1 CLI exits 2 for `kind: isaac`); the patched `CodeRef` must be buildable by it (apply `RepairOutcome.diff` to the base checkout).
- Rung 0 lint / test / smoke commands for the real controller repo; the short smoke run (a few headless steps).
- Episode-log field names for the failure mode; Rung 2's optional inputs (per-subtask progress, action-trace statistics, key frames) are not wired.
- Calibrated `agreement` / `min_agreement`; that Rung 1's parallel runs do not share a batching policy server in CLEAN mode.
- hagent `[[commands]]` path to `run_rung1` inside the container (must be executable there, with `python3` and this package on `PYTHONPATH`).

## Fix F0 (packaging)
- Declared `hybrid-agent` in `dependencies` of vla-router/pyproject.toml; added build-system, `dev` (pytest) and `scipy` extras, and `packages.find` limited to `ladder*`/`router*` (flat layout otherwise fails auto-discovery). README gives one command, `pip install -e "../hybrid-agent[dev]" -e ".[dev]"`, so pip resolves the name from the local path.
- Pinned by `tests/test_packaging.py` (fails on old pyproject/README). The scipy test already used `importorskip`; unchanged.
- Rejected: absolute `file://` direct reference (not portable); PyPI name only without the combined command (pip would look on PyPI); vendoring/PYTHONPATH; a README-only fix.
- NEEDS-REVIEW: the dependency name is bare (`hybrid-agent`), so `pip install -e .` alone fails unless hagent is installed first or in the same command.
- TODO(verify): scipy-present path of `test_fisher_matches_scipy_when_available` (no network here to install scipy).

## Fix F1 (log-only evidence wired into route(); ISSUES #1, #26 (partly), Tier 5, Tier 6 fixture item)

**What**
- `route(incident, backend, config, inputs=None, *, extra_evidence=())` runs Stage 1 (`run_stage1`) on every incident, right after the Stage 0 replay and before the Stage 0 stop decision, and C_behavior (`run_behavior_analysis`) whenever Branch C runs (Branch C, degraded, unknown; plan 3.2.7). Stage 1 must precede the stop: `staleness_variable` is what keeps a stale-server incident (clean Stage 0 passes) from closing as row 1.
- New `RouterInputs` (Stage 1 checks; behavior source + its nominal stats). `RouterConfig` only gains `behavior: BehaviorConfig` (thresholds). Why not inside `RouterConfig`: the config is frozen plain values shared through `LoopConfig`; the inputs are handles to log stores plus per-scene-condition calibration data. `inputs=None` means every source is recorded as skipped, never as clean. A non-`RouterInputs` 4th argument raises `TypeError` (the old 4th positional was evidence).
- `extra_evidence` stays; `RouterResult` records it separately (`external_evidence`) next to `stage1_evidence` and `behavior_evidence`. `evidence` stays the union `decide` saw, so Rung 2 and repair are unchanged.
- `RouterResult.skipped` / `partial` ((source, reason)), `sources_run`, `unchecked`; the verdict notes name every skipped or partly skipped check (only the notes change).
- `Stage1Result.sources_run` is now a field: only checks that ran. Registered sources with no check are listed in `skipped` ("no check configured").
- `Check.run_partial` (default: `run` + no reasons); `StalenessProbe` overrides it: an incident observed in incident mode with missing (or pair-less) incident-mode traces is a partial skip, `staleness_variable` unchecked.
- `IncidentLoop(..., inputs=...)` replaces `evidence_provider`; the `evidence_by_code` fixture is gone. Loop and router cases use `CodeKeyedTraceSource` / `CodeKeyedBehaviorSource` (synthetic logs keyed on `CodeRef.key()`), so a fix removes its planted fault from the logs.

**Unchecked signals in `decide` (user decision).** `decide(..., unchecked=())`. Rule: a row may only evaluate FALSE if every check that could make it TRUE ran; if one was skipped, FALSE becomes UNKNOWN, TRUE stays TRUE, and the existing "UNKNOWN higher row -> row 10" rule does the rest. Only an evidence-based result flips; a result fixed by the replays does not.
- Source -> signals (`vocab.SIGNALS_BY_SOURCE`, used by `route` to build the set): contract_cfg -> contract_violation, render_settings_differ; contract_obs, contract_action -> contract_violation; staleness -> staleness_constant, staleness_variable (partial skip: staleness_variable only); sanity -> render_physics_broken; behavior (only when Branch C ran) -> implied_target_offset/ok/wrong, failure_phase:*. `staleness_variable` is dropped when the incident was observed in clean mode (not applicable, not a skip).
- Row -> signals (`vocab.ROW_SIGNALS`, the default of `DecideConfig.row_signals`): 1 staleness_variable; 2 render_physics_broken (final FALSE only); 3 staleness_variable (not when Stage 0 is NO_EFFECT); 4 contract_violation, staleness_constant, implied_target_offset (the Branch C FALSE; Branch G elimination is already TRUE); 5 render_settings_differ; 8 implied_target_ok, failure_phase:grasp/lift/place and 9 implied_target_wrong (only when nothing restores). Rows 6, 7: replays only. `reference_unqualified` is not mapped (no producer, ISSUES #5).
- **Row-1 inverse rule:** `staleness_variable` can only make row 1 FALSE, so when it is unchecked row 1's TRUE becomes UNKNOWN.
- `decide.row1_state` lets `route` stop after Stage 0 unless row 1 is FALSE (TRUE wins; UNKNOWN blocks everything below). It replaces route's own "Stage 0 inconclusive" check, so route has no row-1 rule of its own (D-4).
- Relaxing a signal later (e.g. accepting a vision verdict without behavior logs) is a config change: `DecideConfig(row_signals=...)`, not a code change.
- Intended consequences: with logs missing, vision, rendering, action-gap and Branch C code cases end at row 10; Branch G code cases too (row 2 is undecided without sanity); `no_fault` observed in incident mode ends at row 10, not row 1. A clean-mode incident without logs can still close at row 1. Row 2 proved by a failing reference replay stays row 2.

**Default trace modes.** Staleness reads both modes (by design). Observation, action, sanity and `BehaviorConfig.mode` stay CLEAN: Stage 0 always produces clean-mode episodes of the failing seeds on the code under test, also on a re-run after a fix, while incident-mode logs exist only for the original code; and clean mode keeps server staleness out of tracking, gripper and grasp-position metrics (a stale server would read as a gains bug or as perception scatter). Render/physics health and preprocessing do not depend on the mode. A missing clean log is now a recorded skip with an UNKNOWN row, not a silent clean.

**Cases.** All 11 router cases route from synthetic logs with no hand-written evidence. code_condition_specific stays the wrong-instruction-string bug (the glossy mug's instruction does not match the training template; config contract, location `eval_config.instruction`); combined plants unnormalised policy inputs in `c0` (observation contract, `obs.normalization`); infra plants a load-dependent delay in incident-mode traces; action_gap fails at grasp with the right implied target. The strict xfail is now a passing test. `tests/fixtures/decide_pre_f1.py` is a frozen copy of the pre-F1 `decide`, the oracle for "empty unchecked set = old verdict".

**Rejected**
- An override in `route()` turning row 1 into row 10 (a second row-1 rule; user decision).
- Treating "row 6 + skipped note" as the lower-confidence verdict for code_condition_specific without contract checks (user decision: row 10).
- A blanket "FALSE -> UNKNOWN for any mapped signal" without looking at why the row is FALSE: it would undecide row 3 for every incident whose clean Stage 0 fails, and rows 8-9 whenever a factor restores.
- Checks inside `RouterConfig`; a default "healthy for any code" log source in the loop cases (the combined case lists every code it logs; others have no logs).
- Keeping `evidence_provider` alongside `inputs` (two evidence paths).
- Replacing code_condition_specific with an obs-contract `swap_rb` bug (user decision); the observation contract is covered by the combined case.

**TODO(verify)**
- The production `TraceSource` / `BehaviorSource` must serve the clean-mode episodes of Stage 0 for `incident.code_ref` (`EpisodeResult.logs`), so a re-run after a fix reads the patched code's logs, not the incident's original ones.
- Python 3.11 (only 3.12 is installed here).
- `vla-router/router/decide.py` and `router/vocab.py` are read-only (mode 444) in the checkout; both were edited for this fix with the user's approval and left at 444.

## Fix F2a (code identity, workspaces, attempts, diff storage; ISSUES #6, #7, #8, #13 partly, #14 task, #42 repairer part)

**What**
- `ladder/workspace.py` is the one place that names and builds code. A name is `<base>` or `<base>+sha1(cumulative diff)[:12]` (superseded by the F2a review fixes: `<label>@<tree12>[+<diff12>]`, the hash over the diff's bytes). The cumulative diff is read off the files against the base snapshot (`cumulative_diff`: `git diff refs/vla/base`, fixed flags, `--binary`, user/system git config off), never concatenated from attempt diffs, so the same files always get the same name and an undone change gets the base name back. `CodeRef` is unchanged.
- `WorkspaceManager(store, bases)`. The store is always on disk and content-addressed: `patches/<name>.patch` (checked against its own name on read) and `bases/<base>.tree` (the snapshot's git tree; a base checkout that changes later is an error, not a new meaning for its name). (Superseded by the F2a review fixes: the tree is in the name, the store keeps no `bases/` record.) Any manager on the same store and bases can rebuild any name; `register(ref, diff)` re-adds a `LoopResult.final_diff` to a fresh store.
- `materialize(ref)` builds a fresh directory: the base's files (`_copy_base` copies them without `.git` at any depth; `git clean -X`, which drops what the tree's `.gitignore` files ignore, runs in `_snapshot_base`, renamed `_snapshot` by the F2a review fixes, which also keep tracked files that a `.gitignore` matches), `git init` with ONE snapshot commit (fixed author/date: deterministic commit ids), `refs/vla/base`, then the cumulative diff committed on top, so HEAD is the code under repair and hagent's `git diff HEAD` is one attempt's change. `git config vla.base` lets a working copy name itself (`code_ref_of`, the Rung 1 CLI default "after"). `view(ref)` is a cached copy for readers (mock rules, synthetic logs). The original checkout is only read.
- `extend(ref, patch)`: apply one attempt's patch to a fresh copy of `ref` and name the result. A patch that does not apply raises `UnrebuildablePatch`.
- `HagentRepairer(workspaces, settings, ...)`: each attempt materializes `request.incident.code_ref` (the last accepted code; the loop only advances on Rung 1 pass / residual / ambiguous), runs `solve()` there, and returns `extend(...)`. `RepairOutcome` gains `code_diff` and `rebuild_error`. The loop records `unrebuildable_patch` with feedback "do not edit or add binary files" (not `no_change`). `RepairRecord.diff` / `.code_diff` and `LoopResult.final_diff` are in `to_dict`.
- #8: the prompt's repo path is the working copy, or `settings.env.workdir` under docker; `RepoIndex` is built from the working copy.
- Backend contract (docstring of `ladder.backend`): a backend gets the files of a `CodeRef` through `WorkspaceManager` (`view` to read, `materialize` to build or run). `MockRule.files` takes a file predicate (`FileContains`, serialisable for the CLI); `MockBackend(workspaces=...)` evaluates each (predicate, name) once. The CLI accepts `"workspace": {"store", "bases"}` for file rules. `IsaacBackend` untouched.
- Synthetic logs are patch-aware: `CodeKeyedTraceSource` / `CodeKeyedBehaviorSource` take `file_faults` (same predicates, looked up after the code key), so the real repairer's patch gets logs and F1's rule does not make every row undecided after a real repair.

**#14 report task (user decision: use it).** The repair TASK is `Report.task`: the report model's validated `task`, or `default_task` when there is no model, no task or a fallback. The model's text fills only the TASK slot; the verdict line (`verdict_text`), the hypothesis framing (template), repro command, rollout summary and repo path are still filled by code. Why: plan 3.2.12 says the report model writes the task, and `_Validator` already checks it (ids, paths); before F2a it was validated and thrown away. Cost: the validator holes (ISSUES #35) now reach the agent's prompt (F5).

**Rejected**
- Naming by concatenated attempt diffs or by the chain of attempt hashes (`c0+h1+h2`): two routes to the same files got two names, and the name claimed patches the content did not have.
- An optional / in-memory store (user decision): the only copy of a fix would die with the process, and F3 needs to rebuild patched code for Rung 0.
- Copying the original `.git` into working copies (user decision): leaks history (ISSUES #13), and its state (dirty tree, hooks, config) would change names.
- Naming by git tree id: robust to diff formatting, but the decision was "a hash of the cumulative diff"; the diff recipe pins every flag instead.
- Using the agent's working directory as the result under the local backend: docker has no host copy, so both paths go through `extend(patch)`.
- Folding an unrebuildable patch into `no_change` (user decision): the agent would be told it changed nothing when it did.

**Known limits / TODO**
- ISSUES #2 stays open: a patched `CodeRef` copies the eval config it started from (F2b brings the config loader).
- hagent's patch is not `--binary` (ISSUES #55) and follows host git config and agent commits (ISSUES #56).
- Determinism is over file bytes and commit ids, not `.git` internals (index timestamps).
- TODO(verify): the docker path of the repairer has not been run (no image).
- `chmod u+w` was applied to vla-router and hybrid-agent by the user; read-only modes are no longer a constraint and were not restored.

## F2a review fixes (ISSUES #6, #8, #13, #30 corrected; review of F0/F1/F2a; plan: PLAN_F2a_fixes.md)

**What**
- **Base identity in the name.** A code name is `<label>@<tree12>` (a base) or `<label>@<tree12>+<sha1(diff bytes)[:12]>`; `<tree12>` is the git tree id of the base snapshot. `WorkspaceManager.base_ref(label)` names a checkout as it is now; `materialize` re-snapshots and refuses a name whose tree no longer matches ("base checkout 'c0' changed ..."). A changed checkout therefore gets new names whichever store is used, and rollout-cache entries (`CodeRef.key()`) for the old content stop matching. The store's `bases/<label>.tree` record is gone. `split_code_name` rejects old-recipe names (`c0+<12 hex>`, chains) and bare labels at parse time, each with a message that says what to do. Incidents and Rung 1 CLI configs that use workspaces must carry these names (`bases` maps labels).
- **Git environment.** Every git call in `ladder/workspace.py` runs without any inherited `GIT_*` variable (plus `GIT_OPTIONAL_LOCKS=0`, `core.fsmonitor=false`, global/system config off). hagent: `env.clean_environ()` for `LocalEnv.execute` (the agent's commands), `LocalEnv._git`, and DockerEnv's host-side runtime calls (the container only ever got `-e` variables).
- **Bytes.** git output, diffs, hash input, store files (`patches/<name>.patch`, bytes) and `git apply` input are bytes. The `str` surface (`cumulative_diff`, `diff_of`, `RepairOutcome.diff`/`code_diff`, `LoopResult.final_diff`) is the lossless `surrogateescape` view (`as_text` / `as_bytes`); `json.dumps` escapes the lone surrogates, so a serialised `LoopResult` still rebuilds exactly. `FileContains` compares bytes.
- **Rebuild causes.** `UnrebuildablePatch.cause`: `binary`, `encoding` (the patch contains U+FFFD: a text decoder lost bytes), `symlink`, `does_not_apply`, `mismatch`. `RepairOutcome.rebuild_cause`; `loop.rebuild_feedback` picks the advice by cause (only `binary` says "do not edit binary files").
- **Local cross-check.** Under the local backend, `HagentRepairer` also names the files the agent left (`code_ref_of(work)`); if `extend(patch)` gives a different name, the attempt is refused (`encoding` if the patch has U+FFFD, else `mismatch`, e.g. committed changes). Why: hagent's patch is decoded with `errors="replace"` (its recipe is F2b's), so a non-UTF-8 byte in an added line applies cleanly and would otherwise rebuild different bytes under a valid name. Under docker the host copy is not the agent's, so only `extend` judges.
- **Snapshot contents.** Every file the original checkout tracks is kept, even if a `.gitignore` matches it (`git ls-files -z --stage` in the original, read only; submodule entries skipped; then `git add -f` with literal pathspecs). Only untracked ignored files are dropped (`git clean -X`). A checkout that is not in a git repository uses the `.gitignore` files alone; any other `ls-files` failure (e.g. dubious ownership) is an error, not a silent fallback.
- **Failure cleanup.** A failed `materialize` removes the directory it created (temp dir or a `dest` it made) and empties a given empty `dest`. `view()` takes a fresh slot per attempt and keeps nothing on failure, so a retry shows the real error.
- **Symlinks: rejected, not replaced by their contents.** An absolute link, or a relative one that resolves outside the tree, makes the base a `WorkspaceError` (checked after `git clean -X`, so an ignored `.venv` with absolute links does not matter) and a patch an `UnrebuildablePatch(cause="symlink")`. Why reject: replacing a link by its target changes what the code is (a linked directory becomes a copy, possibly a large non-code tree), and the target's content would still not be part of the name.
- **`decide(unchecked=...)`.** A bare `str`/`bytes` is a `TypeError`; names outside `vocab.ALL_SIGNALS` (every `SIG_*` plus `failure_phase:<phase>`) are a `ValueError`. Routing unchanged.
- **Record completeness.** `LoopResult.base` (base name of the incident code) is in `to_dict`. `_Run.code_diff` starts as the incident code's cumulative diff, so `final_diff` rebuilds `final_code` also when the incident started on patched code and when no fix was accepted. `IncidentLoop(workspaces=...)`, default the repairer's `workspaces`; without one, `base` and `final_diff` stay "" (scripted repairers).

**Rejected**
- Keeping the per-store `bases/<label>.tree` record: a fresh store accepted changed content under the old name, and the rollout cache never consulted the store.
- Full 40-hex tree ids in names: 12 hex as for the diff part; the tree is re-checked on every `materialize`.
- Taking the patch from the agent's working directory instead of hagent's: docker has no host copy (as in F2a). The local cross-check detects the difference instead.
- Replacing escaping symlinks by file contents (see above).
- A blanket "only non-binary changes" rule: the feedback now names the real cause.

**Known limits / TODO**
- hagent's own patch is still text with `errors="replace"`, without `--binary`, and follows host git config (ISSUES #55, #56; F2b). Under docker a lossy patch whose U+FFFD lands only in added lines is not detectable.
- No production code builds incidents with `base_ref` yet (ISSUES #45).
- The F2a review's leakage items (`$RUNG1_CONFIG` contents, agent-rewritable `vla.base` / `refs/vla/base`) are F5 (ISSUES #57, #58).


## F2b (partial skips, eval config from the files, trace code key, hagent start commit; ISSUES #59, #2/#51, #52, #56; plan: PLAN_F2b.md)

**What**
- **`vocab.PartialSkip(reason, signals)`** (#59). A check that ran but could not look at part of its inputs says which signals that part would have emitted. `Check.run_partial` returns `list[PartialSkip]`; `Stage1Result.partial` and `RouterResult.partial` hold `(source, PartialSkip)`; `router.unchecked_signals` unions the skipped sources' signals with each partial skip's own. The `"staleness"` special case is gone (the staleness probe's skip names `staleness_variable`); the only remaining rule is that `staleness_variable` does not apply to an incident observed in clean mode. A `PartialSkip` must name known signals (`ValueError` otherwise).
- **C_behavior** (`BehaviorAnalysis.partial`): every "not judged" note is also a `PartialSkip`: the implied-target ones mark `implied_target_{offset,ok,wrong}`, the phase ones mark all `failure_phase:*`, "no failing episodes" marks every behavior signal. `route()` adds them to `partial` (source `behavior`) when Branch C ran and recomputes the unchecked set, so rows 4, 8 and 9 go UNKNOWN instead of FALSE. A logged "not judged" is not a failure of the check: `behavior` stays in `sources_run`.
- **Action / observation contract**: a sub-check that cannot run (no action layout, no action min/max, no single-index gripper layout; a metric with no logged values) is a `PartialSkip` on `contract_violation`. The log warning stays.
- **Eval config from the files** (#2, #51). `WorkspaceManager(config_loader=...)`, default `file_config_loader("config/eval.yaml")` (YAML, or JSON for `.json`). `extend`, `adopt`, `code_ref_of` (`config_loader` argument) and `base_ref` (when no explicit config is given) read the eval config of the working copy; the patched `CodeRef.eval_config` is never copied from the code it was built from. A missing or unreadable config file is a `WorkspaceError`, not `{}`: an empty config would read as "nothing to compare". `run_rung1` uses the same loader (`workspace.eval_config_path` in its JSON overrides the path). `pyyaml` is an optional extra (`[yaml]`, included in `[dev]`), imported lazily by the default loader only; without it JSON configs and custom `config_loader`s work and a YAML file is a clear `WorkspaceError`. hybrid-agent never uses it. An explicit `base_ref(label, eval_config)` stays the caller's statement; the loader should agree with it.
- **Trace code key** (#52). `EpisodeTrace.code_key` and `BehaviorTrace.code_key` (`CodeRef.key()`, i.e. code and eval config). `TraceSource.load` / `BehaviorSource.load` are template methods: they call `_load` and then `verify_code_key`; traces of another code, or with no key at all, raise `TraceUnavailable`, so the checks are recorded as skipped. Subclasses implement `_load`; `tests/test_trace_code_key.py` fails if any source in `router/` or `tests/` overrides `load`. The synthetic sources stamp the incident's key; `make_traces`/`make_source`/`make_behavior_source` stamp `make_incident()`'s key by default.
- **hagent starts from a recorded commit** (#56, hybrid-agent submodule). `GitHelpers.record_start()` (called by `LocalEnv` and `DockerEnv` constructors) stores `HEAD`; `git_diff()` and `git_tracked_diff_hash()` diff against it, so an agent's own `git commit` stays in the patch. A repository without commits keeps the old fallback. The submodule change is committed (`hybrid-agent` 75e8dc6) and the superproject points at it.
- **Test fixtures.** Router cases now give the training spec the synthetic action layout and limits (`tests.router_cases.ACTION_SPEC`); without them every case would run with `contract_violation` unchecked and row 4 could never be ruled out. The test checkout has `config/eval.yaml`.

**Rejected**
- Treating unkeyed traces as acceptable ("unknown provenance, assume it matches"): that is exactly what lets the old code's logs stand for a patched one. Real log writers must stamp `code_key` (TODO(verify)).
- Marking the whole `behavior` source unchecked on any "not judged": only the signals of the part that was not judged are unchecked (a phase that is judged still counts).
- Defaulting a missing config file to `{}`.

**Known limits / TODO**
- `config/eval.yaml` is a guess at where the controller repo keeps its eval config (`EVAL_CONFIG_PATH`, TODO(verify)); a repo with several config files needs its own loader.
- #55 (`--binary`) and the host-git-config half of #56 are not done.
- Row 8/9 become UNKNOWN whenever a log has no subtask records, so a deployment that never logs them reads row 10 for action-gap cases; that is the intended cost of #59.

## Fix F3 (Rung 0 adapter, errors, budgets, pass probabilities; ISSUES #9, #10, #11, #14; plan: PLAN_F3.md)

**What**
- **Rung 0 adapter** (#9). `ladder.rung0.rung0_hook(workspaces, settings, cfg, reviewer=, task=)`. It builds `outcome.code_ref` with `WorkspaceManager.materialize` into a temp dir, runs `run_rung0` on an env from `hagent.runner.make_env` (so a docker backend is honoured) and deletes the dir. `git_diff()` is `outcome.diff` (this attempt's patch): the rebuilt copy's HEAD already is the patched code. No second copy of the patched files is kept.
- **Reviewer not run twice.** `RepairOutcome.final_review_approved` is true only when hagent's last logged review is APPROVE. If hagent never reviewed, its review raised (not logged), or it ended on a REJECT (the rejection cap lets the submission through unreviewed), the Rung 0 reviewer runs. `reviewer=None` is passed when it is true.
- **Reviewer `ModelError` fails open**: check status `error`, the rung passes, `Rung0Result.reviewer_error` holds `Type: message`. Only `ModelError` (what hagent's own review catches); other exceptions end the incident (#10).
- **Errors** (#10). `_Run.execute` catches `Exception` around the whole run, router included, because "nothing propagates out of `IncidentLoop.run`". `LoopResult.error_stage` (router | repairer | rung0 | rung1 | rung2), `error_type`, `error_message`; status UNRESOLVED, reason `error in <stage>: ...`; passes, repairs and Rung 2 results so far are kept, and the in-flight repair record gets `result = "<stage>_error"`.
- **Budgets** (#11). `LoopConfig.max_cost`, `max_big_calls`; 0 is off; spent `>=` cap stops. Checked before every repair attempt and before Rung 2 (action `budget`, the Rung 2 call is not made). The spend counted is what the loop already summed: repairer cost and big calls (hagent calls + reviews), Rung 0 reviewer, Rung 2. `IncidentLoop.warnings` / `LoopResult.warnings` come from `HagentRepairer.pricing_warnings(max_cost)`, which runs hagent's `pricing_warnings` on a copy of the settings with the incident cap as `max_cost`; also `log.warning`.
- **Pass probabilities** (#14): `RepairOutcome.reviews/pass_probabilities` (all of hagent's reviews, in order, including rejected ones) -> `RepairRecord` -> `to_dict`.
- **Pending Rung 2 at the pass cap**: `LoopResult.rung2_pending` and a reason that says Rung 2 never ran.

**Rejected**
- Catching only the four named stages: a router bug would still crash a batch run; the error is recorded with its stage either way.
- Catching `BaseException`: would hide Ctrl-C.
- Checking caps only between passes: one pass can make three repair attempts.
- Stopping at `>` the cap: the next attempt would start already over budget.
- A second copy of the patched files kept for Rung 0 (or a repairer-owned work dir surviving `repair()`): the name is the content.
- Running the reviewer when hagent's review merely exists: a final REJECT submission still deserves a review.
- Making the reviewer's `pass_probability` gate anything: still logged only.
- A simulator-episode cap: no counter exists yet (ISSUES #61).

**Rung 2 override rule** (#4, decided by the user after F3's first pass). Rung 2's label decides only on re-run rows 4 and 10; the router's re-run verdict decides every other row. In the pending-Rung-2 branch of `_Run._execute`, after the Rung 2 call: rows 4 and 10 use the label (`code` -> repair; `unclear`, or a `data_gap` the gate turned into unclear -> UNRESOLVED); rows 6 and 7 -> data ladder whatever the label (`via_rung2` only if the label was `data_gap`); every other row falls through to the router's destination. `Rung2Result.overridden` holds the reason ("overridden by router: ...") when the label did not decide; `model_label` is always kept. The call is still made on every non-STOP row (row 1 closes without it), so the label is logged for rows 2, 3, 5, 8, 9 and the call budget logic is unchanged. A missing or exhausted Rung 2 now hands a row 6/7 re-run to the data ladder instead of UNRESOLVED. `decide()` untouched. Tests: `tests/test_f3_rung2_override_current.py` (rows 4, 10, 6, 7, others x each label; its first two tests used to pin the old behaviour).
Rejected: skipping the call on rows the router decides alone (loses the logged label); dropping `via_rung2` (it still tells the data ladder the model agreed).

**Known limits**
- hagent's own caps restart on each attempt, so one attempt can overshoot an incident cap by up to one run.
- Rung 2's and the Rung 0 reviewer's prices are not checked by the warning (they are not hagent settings).

## F2b/F3 review fixes (11 items; plan: PLAN_F3_fixes.md)

**What**
- **Budget vs destination** (1). `budget_stop()` gates only a repair attempt and, in the pending-Rung-2 branch, the call on re-run rows 4 and 10 (action `budget`, UNRESOLVED, no Rung 2 record, as before). On every other row `_Run.rung2()` itself returns the "unclear" fallback when a cap is hit ("label not obtained: budget (...)"), exactly like an exhausted `max_rung2_calls`: rows 6/7 go to the data ladder, the rest follow the router. Both kinds of limit now give one destination per row.
- **Unreadable eval config** (2, 5). `WorkspaceManager.extend` raises `UnrebuildablePatch(cause="config_unreadable")` when the patched files' config cannot be read (underscore like the other causes); `HagentRepairer._local_name` reads the agent's copy with `workspaces.config_loader` and raises the same, so the attempt is recorded, its cost kept, the agent gets feedback (`rebuild_feedback` has advice for it) and the remaining attempts run. A copy that no longer names itself (no `vla.base`) still skips the local cross-check.
- **Setup errors** (3, 8). `_Run.__init__` no longer touches the workspaces; `base_name`/`diff_of` run in `_execute` under stage `setup`, so a bare/old-recipe/unknown-label name or a corrupt store ends UNRESOLVED with `error_stage="setup"`. The data-ladder hand-off has its own stage `handoff`.
- **No silent skips left** (4). `fire()` with no values, the observation contract's image size and channel order, and a gripper command whose length differs from the raw actions are `PartialSkip(reason, {contract_violation})`. The open-loop probe is not in that list (ISSUES #65): it is an optional extra check, so `probe=None` is not a skip. `Check.notes(incident)` (default `[]`) feeds `Stage1Result.notes` and `RouterResult.notes`; the obs contract returns "open-loop probe not configured". `decide()` is unchanged. Fixtures: the router cases' training spec has `image_size` / `channel_order`; they still pass `synthetic.NominalProbe`, and tests also run without it.
- **Explicit eval config** (6). `extend` keeps the config of an explicit `base_ref(label, eval_config)` until a patch changes the config as the loader reads it (compared with the loader on the base snapshot, rebuilt with `git read-tree` / `checkout-index` of `BASE_REF`); then the patched files' value is taken, and stays so for later patches. A patch that only reformats the file (same loaded mapping) keeps the explicit config. The comparison is by loaded value, not file bytes, because the loader is pluggable.
- **Report cost** (7). `HagentRepairer.render()` returns the `Report` too; `RepairOutcome.cost = result.total_cost + report.cost`, so `max_cost` sees it (also on the not-submitted and unrebuildable returns).
- **Rung 0 in the loop** (11). `LoopConfig.rung0: Optional[Rung0Config]` (default None = off). `IncidentLoop` builds `rung0_hook(workspaces, settings, cfg.rung0, reviewer=rung0_reviewer)` from `workspaces`/`settings` (default: the repairer's); a missing source or passing a hook as well is a `ValueError` at construction. README has the example, pinned by a test.
- **Tests restored** (10). `test_router.py` asserts the exact partial skips again (empty, or the failure-phase signals of `behavior` for the cases whose log has no subtask records); `test_repair.py` compares the patched config against the base's (the agent also edits the config, so a copied-over config fails); the `mismatch` branch has a test (a manager whose rebuild gives other files). 10d left as is.
- **Docs** (9, 11). The Rung 2 headline says the label decides only on re-run rows 4 and 10; ISSUES #64 records the unconfigured-Rung-2 row-4 behaviour (unchanged); the stale F2b submodule line, PLAN_F3.md item 4 and the `repair.py` docstring are fixed. ISSUES #9 is FIXED now that the wiring test exists.

**Rejected**
- Failing the incident when a cap is hit on rows 6/7 or other rows: the data ladder / router destination does not need the label, and the other limit already followed it.
- Making a missing probe a constructor choice (e.g. `probe=False`) to keep old deployments routing: widens scope; recorded in ISSUES #65 instead.
- Comparing config files by bytes for the explicit-config rule: the loader may read anything.


## Router change 1: backend id in the rollout-cache key (ISSUES #19; sim side Phase 1, docs/IMPL_PLAN.md §5)

**What**
- `CacheKey` gains `backend: str = ""`; `CacheKey.make(..., backend="")`. `CachedBackend` reads `getattr(backend, "backend_id", "")` and puts it in every key, so two backends sharing one cache file never return each other's episodes.
- `RolloutBackend.backend_id: str = ""` (class attribute, not abstract). `MockBackend` keeps `""`. vla-isaac's `IsaacBackend` sets `isaac:<hash of Arena / GR00T commits, runner, task config, factor table>`.
- Backward compatible: a key with an empty id is serialised without the `backend` field and `from_dict` defaults it to `""`, so a cache written before this change loads and still matches the mock. Tests: `tests/test_backend_cache.py` (`test_cache_key_includes_the_backend_...`, `test_cache_lines_from_before_the_backend_id_...`).

**Rejected**
- Making `backend_id` abstract: every test double and the mock would have to change, and old caches would stop matching.
- Folding the id into `ActorRef.key()` or `CodeRef.key()`: it is neither the actor nor the code, and those keys are also used outside the cache.

**Known limits**
- Two backends that both leave the id empty still share entries (as before); a real backend must set one.
