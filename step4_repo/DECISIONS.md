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
