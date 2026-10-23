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
- hagent is not a declared dependency of the router `pyproject.toml` (`router/llm.py` and `ladder/rung0.py`, `ladder/repair.py` import it); `pip install -e hybrid-agent` is assumed.
