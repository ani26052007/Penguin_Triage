# Build prompts: VLA repair ladder + hagent

## How to use

1. Start a new session for each step. Attach `plan.md` and the hagent `README.md` (and the repo, if the tool can read it).
2. Paste the **Master prompt** first, then **one** step prompt below. Never two steps in one session.
3. When the step is done: run the tests, read the summary, then run the **Review prompt** in a *fresh* session before moving on.
4. Commit after each reviewed step. Keep a `DECISIONS.md` in the repo; the Master prompt tells the model to append to it.

Steps marked **[needs human]** require something only you have (API key, Isaac machine, policy code).

---

## Master prompt (paste first, every session)

```
You are helping build a research system: when a VLA policy fails in Isaac Sim, route the cause
(code bug / vision gap / other) and fix it with the cheapest tool. The attached plan.md is the
source of truth; section numbers below refer to it. hagent (attached README) is the existing
repair agent and is already built.

Rules:
1. Do ONLY the step I give you. Do not start later steps or add features not in plan.md.
2. If plan.md and my step disagree, or a design choice is unclear, ask me ONE short question
   before writing code. Do not silently deviate from plan.md.
3. No simulator, GPU or real policy is available to you. Put everything simulator-related behind
   an interface and build a mock backend. Never guess Isaac, GR00T or Cosmos APIs: mark them
   TODO(verify) and list them in your final summary.
4. Python 3.11. hagent core stays stdlib-only. Other packages only if essential; say why.
5. Every module: small interface, type hints, docstring stating its contract. Thresholds and
   sizes are config values, never hard-coded constants.
6. Write tests with the code. All tests (old and new) must pass before you finish. Run them.
7. Do not weaken any hagent loop-protection guarantee (caps, detector, review cannot loop,
   failures degrade without crashing).
8. Append a short entry to DECISIONS.md: what you chose, why, what you rejected.
9. Finish with a summary of at most 10 lines: files changed, tests run and result,
   TODO(verify) items, open questions.
```

---

## Step 1A: Harden hagent

```
STEP 1A. Improve hagent (plan.md 3.3, README "Known limitations").
Build:
1. Real token accounting: read `usage` from the API response; fall back to the character estimate
   only if absent. Cost cap uses real tokens.
2. Docker/Podman environment implementing the existing env interface (execute, cwd, git_diff,
   git_diff_hash). Options: no network, CPU/memory limits, repo copied in (not bind-mounted
   read-write to the original), per-command timeout kills the whole process tree.
3. Reviewer returns APPROVE/REJECT plus a pass probability (0-1). Log it in the trajectory and
   `usage`. It must never gate anything. Unparseable reply still counts as APPROVE.
4. Task prompt template loader: a file with placeholders for task, router verdict (framed as
   "hypothesis to verify first"), location evidence, repro command, rollout summary, repo path.
5. Custom commands: a config list of extra shell commands (name, path, description) injected into
   the system prompt, each with its own output-truncation limit. (Used later for `run_rung1`.)
Done when: all old tests pass, new tests cover items 1-5, `demo_offline.py` still works, and the
container env passes the same env tests as the local env (skip cleanly if no Docker).
Do not: change the loop, detector, or compaction logic.
```

## Step 1B: Benchmark hagent on a live model [needs human]

```
STEP 1B. Live benchmark harness for hagent (plan.md Section 8 step 4).
Build a script `bench/run_bench.py` that runs `--mode small|big|hybrid` over a folder of tasks
(each task = small git repo with a failing test + issue text), N repeats each, and writes a CSV:
task, mode, repeat, resolved (hidden test passes), steps, cost, small/big calls, exit_status.
Also build 6 tiny sample tasks (bugs: off-by-one, wrong sign, swapped args, wrong default,
stale cache, wrong index) so I can run it immediately.
Also print a summary table: resolve rate and mean cost per mode.
I will run it with my API key; do not hard-code keys or model ids (read config.toml).
Done when: the script runs end to end with the scripted/offline model, and I can run it live
with one command.
```

---

## Step 2: Shared foundations

```
STEP 2. Foundations (plan.md 3.2.2, 3.2.3, 3.2.9, 3.2.14).
Create package `ladder/` and `router/schemas`:
1. Schemas: Incident, SceneSpec (nominal + named factor changes), TrainingSpec, Evidence
   (id, source, signal, strength, location, detail), ReplayResult, Verdict. JSON-serializable.
2. `ladder/stats.py` (shared by router and Rung 1): proportion with confidence interval;
   three-valued outcome (restored / partial / no_effect) from a rate vs a failure anchor and a
   nominal band; paired test on discordant seeds and unpaired fallback; staged-seed decision
   (stage A, stage B, ambiguous) that accounts for the early look.
3. `RolloutBackend` interface: run(scene_spec, code_ref, seeds, mode={clean,incident}) ->
   per-episode results + logs. Plus a MockBackend whose success rates are set by a table.
4. Rollout cache keyed by (code hash, checkpoint, scene spec, seed, mode).
Done when: stats are unit-tested on known cases (including borderline intervals and the
paired vs unpaired switch), and MockBackend + cache are tested.
Do not: write any router logic yet.
```

## Step 3: Decision table and mocked router

```
STEP 3. Decision logic (plan.md 3.2.8, 3.2.14).
1. `router/decide.py`: a PURE function from (evidence list, replay table) to Verdict, implementing
   the 10-row decision table with the stated precedence, confidence rules (located evidence =
   higher, elimination only = lower), the "likely second cause" flag, and "undetermined".
2. Tests for every row, for precedence conflicts (e.g. contract violation AND a vision restore),
   partial restores, conflicting evidence, and noisy borderline tables.
3. A confusion-matrix script: given a list of (fault, expected verdict) cases and a MockBackend
   configured per fault, run decide() and print the matrix.
Done when: all 10 rows have tests, and the matrix script runs on 11 mocked fault classes
(plan.md 6.5).
Do not: implement replay runners or Stage 1 checks.
```

## Step 4: Router stages with replays

```
STEP 4. Replay stages (plan.md 3.2.4 Stage 0, 3.2.5, 3.2.6, 3.2.7).
Build, on top of RolloutBackend:
1. Stage 0 reproduce (extend seeds 10 -> 20 when ambiguous; "no significant failure" exit).
2. Stage 2 branch split with the three outcomes.
3. Branch G replays: infra (clean vs incident mode), simulator (reference controller on nominal),
   pipeline isolation (reference controller actions through the current wrapper).
4. Branch C replays: reference controller on the shifted scene, one-factor restores for every
   declared factor, all-factors restore, conditional render-settings replay.
5. `router/router.py` orchestrator: runs stages, collects Evidence + replay table, calls decide().
   No early stop across hypotheses. All replays use the same seed set and the cache.
Done when: with MockBackend, the full router returns the right verdict for each of the 11 mocked
fault classes, and episodes-per-incident is logged.
Do not: implement Stage 1 checks or any LLM call.
```

## Step 5: Stage 1 checks from logs

```
STEP 5. Stage 1 checks (plan.md 3.2.4 Stage 1).
Each check is a module: run(incident) -> list[Evidence], with strength calibrated against a
nominal distribution passed in as data.
1. Config contract (eval config vs training spec vs scene spec; undeclared differences flagged).
2. Observation contract (channel means, range, orientation, resolution; plus the open-loop
   check that replays training frames through current preprocessing, behind an interface).
3. Action contract (ranges, gripper sign, commanded vs actual gripper width, tracking error).
4. Staleness probe (constant delay = code latency, variable delay = server staleness).
5. Render/physics sanity (black/saturated frames, NaN, interpenetration, resting pose).
Done when: each check has synthetic-trace unit tests, including one planted fault per check
(e.g. an action trace with a known delay that the probe recovers) and a clean trace with no flag.
Do not: touch replay stages or the decision table, except to register these as evidence sources.
```

## Step 6: Behavior analysis and LLM layer

```
STEP 6. Behavior analysis + report layer (plan.md 3.2.7 C_behavior, 3.2.11, 3.2.12).
1. Behavior analysis: implied target (end-effector pose at gripper close vs true object pose;
   scatter and consistent offset) and failure phase from subtask progress. Emits Evidence.
2. LLM client reuse from hagent models.py. Report writer: input = Verdict + evidence; output =
   strict JSON (summary, hagent task prompt using the Step 1A template, ranked localization).
   It may cite only evidence ids that exist, and name only files/symbols present in the repo or
   evidence. Invalid JSON: retry once, then fall back to evidence ranked by strength.
3. Config-key mapper (small model) for the config contract, output cached and hand-checkable.
4. Optional factor search stub for verdict row 9 (plan.md 3.2.12): interface only, calls a vision
   model with nominal vs failing frames, returns candidate factors. Mark endpoint TODO(verify).
Done when: tested with scripted models, including invalid JSON, hallucinated evidence ids and
hallucinated file names (all rejected).
Do not: let any LLM output change a route.
```

---

## Step 7: Ladder and incident loop

```
STEP 7. Verification ladder + loop (plan.md Section 4, 3.2.11).
1. Rung 0: lint/tests/diff sanity and a short smoke run via bash, then the hagent reviewer.
2. Rung 1: `run_rung1 --seeds N` CLI (registered as a hagent custom command from Step 1A) using
   ladder/stats staged seeds; prints a compact result: pass rate, interval, decision, top error mode.
3. Rung 2: one big-model call reading the router re-run output; returns code / data gap /
   unclear. A "data gap" must match router rows 6-7, otherwise it is rejected.
4. Incident loop: incident -> router -> destination (hagent / config edit / data ladder / report)
   -> re-run router on the fixed system -> close or route again. Capped passes per incident.
   Code is fixed before vision when "likely second cause" is flagged.
Done when: the whole loop runs end to end on MockBackend, including a combined case
(code bug + lighting shift): code verdict, fix, re-run returns vision, hand-off to the data ladder stub.
```

## Step 8: Bug injection and evaluation set

```
STEP 8. Bug suite + router evaluation (plan.md 6.4, 6.5, 3.3 "No leakage").
1. Injection framework: each bug = a parameterized patch on a toy wrapper repo (channel swap,
   image flip, gripper sign, index slice, chunk length, latency, frame offset, gain decay,
   normalization offset, condition-specific instruction bug, one-resolution preprocessing bug).
   Ledger (bug id, diff, symptom, expected verdict) is stored OUTSIDE the agent's container.
2. No-leakage container builder: squash history to one commit, strip tell-tale comments and
   flag names from a configurable list, normalize file timestamps. Include an audit script that
   tries `git log`, `git diff`, grep for ledger strings, and fails if anything leaks.
3. Evaluation set generator: all 11 classes in plan.md 6.5, calibration/test split, held-out bug types.
4. Evaluation runner: confusion matrix, undetermined rate, accuracy when committing, episodes per
   incident, false-verdict rate on no-fault incidents, localization hit rate. Ablation switches:
   elimination only / + Stage 1 / full.
Done when: the runner produces all metrics on MockBackend and the leakage audit passes.
```

---

## Step 9: Isaac backend [needs human]

```
STEP 9. Real RolloutBackend for Isaac Lab / Arena (plan.md 3.1, 3.2.2).
This needs the Isaac install and the chosen policy code, which you cannot run. So:
1. FIRST output a verification checklist: every Isaac/Arena/policy API you intend to call and
   where in the installed code to confirm it. Wait for me to confirm before writing the backend.
2. Then implement the backend: apply a scene spec (factor changes, visual-only material swap),
   clean vs incident inference modes, seeds, instrumentation (per-frame state, policy-input
   tensors, action/observation step ids, gripper commanded vs actual, subtask progress), and
   the reference-controller hook that emits actions in the policy's output format.
3. Add scripts I can run: qualify the reference controller on every condition, measure replay
   agreement, verify a material swap changes no physics.
Done when: the scripts exist and run against MockBackend; real runs are mine to do.
```

## Step 10: Data ladder [needs human for Isaac parts]

```
STEP 10. Vision data ladder (plan.md Section 5.2, 5.3, 5.5).
1. Clean-rollout collector (option A: VLA's own successful rollouts), logging full per-frame state.
2. Rung A: augmentation config (jitter, brightness, contrast, blur, crop) applied at train time.
3. Rung B: re-render by replaying logged per-frame state (never open-loop action replay), with a
   pose-tolerance check that drops mismatching episodes. Isaac calls behind the backend interface.
4. Rung C: Cosmos Transfer wrapper as a documented stub, with quality gates (object identity,
   geometry, contact timing) implemented as functions on frames/state.
5. Converter to the policy's training format (TODO(verify): format) and a configurable
   nominal/shifted mix with a validation split. No hard-coded ratio.
6. Rung choice from the router factor via the rule table in plan.md 3.2.10.
Done when: Rung A and the quality gates are tested; Rung B and the converter run on mock data.
```

## Step 11: Fine-tune and validate [needs human]

```
STEP 11. Fine-tuning wrapper + validation (plan.md 5.4, 5.6).
FIRST list the real tune flags and module names in the installed policy code and wait for me to
confirm. Then build: a thin wrapper that launches the policy's official fine-tune with the
default recipe (projector + action head; vision side only as an optional flag), records
hyperparameters, GPU and VRAM; and a validation script that runs staged Rung 1 on held-out
conditions with the failing baseline's paired seeds, plus a nominal-regression check against the
base checkpoint. LoRA only if I ask.
Done when: the wrapper dry-runs with a fake trainer and the validation script runs on MockBackend.
```

## Step 12: Evaluation and write-up

```
STEP 12. Experiments and report (plan.md Sections 6, 7, 10).
Build scripts that consume result files and produce: router confusion matrix with and without an
oracle commit; router ablations (elimination / + Stage 1 / full / VLM-only baseline); repair
success and Rung 1 pass rates; cost per fix; cost and held-out success vs the fixed-recipe
baseline; seeds saved by staged validation with its measured wrong-pass/wrong-fail rate;
data-ladder rung sufficiency; nominal before/after. Output tables (CSV + markdown) and simple plots.
Then draft the Limitations section from plan.md 3.2.15 and Section 9, using only numbers that exist
in the result files. Mark anything not measured as "not measured".
```

---

## Review prompt (fresh session after every step)

```
Review the changes for STEP <N> against plan.md section(s) <...>. You did not write this code.
1. List where the code deviates from plan.md or from the step's "Done when".
2. Find ways the router or ladder could return a wrong verdict silently (no error, wrong route).
3. Find any place a model output can change a route, or ground truth can leak to the repair agent.
4. List untested branches and thresholds that are hard-coded.
5. Run the tests. Report failures only.
Output: a numbered list of concrete problems with file and line, most serious first. No praise,
no rewrites. If there are no serious problems, say so in one line.
```
