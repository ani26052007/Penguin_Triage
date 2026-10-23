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
