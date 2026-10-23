# Automated Repair Ladder for VLA Policies in Isaac Sim

## 1. One-paragraph pitch

A VLA policy fails a task in Isaac Sim. The system decides **why** it failed, then fixes it with the cheapest tool that works:

- **Code, config or control bug** -> the repair agent (hagent: small model + big model) patches the code, and the patch is verified by a ladder of increasingly expensive checks.
- **Vision data gap** -> climb a data-cost ladder (augmentation -> Isaac randomization -> Cosmos Transfer) only as far as needed, then fine-tune the policy on the new data.
- **Eval render settings that differ from the training data** -> a config edit, handled as a small hagent task.
- **Action/skill gap, new sensor modality, simulator/asset/infra issue** -> detected and reported, *not fixed* (too expensive for our budget, or not a policy problem). We state this as a limitation.

The router that makes this decision is the most important component: a wrong route wastes the whole downstream spend. It is built from deterministic checks and simulator replays combined by a fixed rule table, and models never decide the route (3.2).

Headline claim to prove: **"Given a failing rollout, the system correctly routes the cause and restores held-out success, at lower cost than a fixed expensive recipe (always Cosmos + fine-tune, or always fine-tuning on a large randomized dataset)."**

The baseline is deliberately a realistic alternative a practitioner would use, not "retraining from scratch", which nobody does for a VLA and which would make the cost claim trivially true.

## 2. Scope (decide this up front)

| Failure source | In scope? | Why |
|---|---|---|
| Code / config / control bugs (latency, frame transform, gains, channel order, image flip, gripper sign, index mapping, chunk/replan steps) | Yes, fix | Cheap to inject, cheap to verify, hagent is built for this. Real examples exist in public issue threads (see 6.4) |
| Vision gap (glossy objects, lighting, camera shift) | Yes, fix | Cheapest data path: existing rollouts + augmentation / Isaac randomization / Cosmos Transfer |
| New sensor modality (RGB -> RGBD, thermal, fisheye) | Detect only | Needs backbone changes and new data types; out of budget |
| Action/skill gap (clutter planning, unseen geometry, joint limits) | Detect only (stretch: cheap fix via scripted/Mimic data on structured tasks) | RL or Dreams-style generation is too expensive. State as a limitation, not a hidden gap |
| Rendering mismatch (eval render settings differ from training data) | Yes, fix (config edit) | One-line change, verified by Rung 1 |
| Environment / simulator / asset / inference-server issues | Detect only | Flag when a known-good reference controller fails on the same scene, or when results change between serial and parallel inference |
| Sim-to-real | No | Explicitly out of scope |

Honest framing for the paper: *we prove the pipeline end to end on code bugs and vision gaps; other causes are shown to be detectable but are gated by compute budget.*

## 3. Architecture

```
failing rollout (+ scene spec, training spec)
      |
      v
  ROUTER  (deterministic checks + counterfactual replays + fixed decision table, see 3.2)
      |
      +-- code/config/control bug --> REPAIR AGENT (hagent) --> Rung 0 -> Rung 1 -> Rung 2
      |                                                    ^                   |
      |                                                    +---- fail/retry ---+
      +-- rendering mismatch ----> config edit (small hagent task) --> Rung 1
      +-- vision gap ------> DATA-COST LADDER (augment -> Isaac randomization -> Cosmos)
      |                        --> fine-tune --> Rung 1 on held-out
      +-- action gap / perception unresolved / sim, asset or infra issue / undetermined
      |                                                          --> report only
      |
      +-- after ANY fix: re-run the router on the fixed system (capped passes per incident)
```

### 3.1 Environment

- **Isaac Sim / Isaac Lab (+ Arena)** for rollouts, data generation and final validation. Headless and parallel.
- Policy under test: a pretrained VLA (GR00T or the DROID-style pick-and-place example). **First task of the project:** confirm the exact policy and version, read its model code, and list the real module names and tune flags (e.g. visual / LLM / projector / diffusion-model). Do not assume any of this from documentation or chat summaries.
- Fine-tuning data must be in the policy's expected format (for GR00T: LeRobot-style parquet + mp4, plus its modality config). Budget time for the conversion step (e.g. LeIsaac or a custom converter).
- A **known-good reference controller** for the chosen task (scripted IK or planner, see 5.2 option B) is useful beyond data generation: it is how the router tells a simulator or asset problem from a policy problem (3.2.6, 3.2.7). It must emit actions in the policy's own output format, and it must be **qualified** (success measured on nominal and on every shifted condition) before the router relies on it (3.2.2).
- The router's prerequisites (scene spec, training spec, rollout instrumentation, clean vs incident inference modes, rollout cache, replay-agreement measurement) are built alongside the failure map, see 3.2.2 and Section 8.

### 3.2 Router: counterfactual replay plus deterministic checks

The router answers one question: **why did this rollout fail, and which destination should it go to?** It is the most expensive place to be wrong. A wrong route sends the wrong fix (agent tokens, data generation, fine-tuning) at the wrong cause, and nothing downstream can recover that spend. So the design is conservative: experiments and deterministic checks produce evidence, a fixed rule table turns evidence into a route, and models never decide.

#### 3.2.1 Design principles

1. **Interventions decide, models explain.** The route comes from simulator replays and deterministic checks, combined by the fixed decision table in 3.2.8. LLMs and VLMs write reports, map config keys, and propose extra visual factors to test. A proposed factor still has to be confirmed by a replay. Cosmos is a generator, not a classifier, and no model labels the cause from pixels.
2. **Change one factor at a time, and run all of them.** No early stop across hypotheses. If the router stops at the first replay that flips, the verdict depends on test order. Example: a channel swap that only hurts red objects flips both when the code is reverted and when the appearance is restored. Whichever replay ran first would win.
3. **Positive evidence before elimination.** Code bugs are routed by located evidence (a violated contract, a pipeline-stage test that fails) whenever possible. Elimination is the fallback and carries lower confidence.
4. **Three-valued outcomes everywhere.** Each replay ends as restored, partial, or no effect. The router can also end in **undetermined**, which is a reportable result and not a failure of the system.
5. **No last-known-good commit inside the router.** In a real failure there usually is none, and if there were, the fix would be `git revert`. For injected bugs the clean commit may be used only to score the router (6.5). It is never an input and never visible to the repair agent (3.3).
6. **Isolate by pipeline stage, not by commit.** The system is a chain: simulator render, observation preprocessing, policy, action postprocessing, controller, physics. Every bug in 6.4 lives in one link. Testing links directly lets the router localize a bug without any commit history.

#### 3.2.2 Prerequisites (built in build-order steps 1 and 2, see Section 8)

The router only works if each incident carries structured information. Most of this logging is needed for the failure map and the data pipeline anyway, so build it once.

| Item | What it is | Why the router needs it |
|---|---|---|
| **Scene spec** | Every scene is "nominal plus a list of named factor changes", e.g. `lighting=dark_hdr_03`, `material(mug)=glossy_02`, `camera_offset=+4cm x` | Makes one-factor replays possible. Separates an intended test condition from an unintended config change |
| **Training spec** | Extracted once from the policy's training and data config: image size, channel order, normalization stats, camera names and poses, action horizon, replan and denoising steps, control frequency, render settings of the training data, state and action index layout | Anything that differs from the training spec without being a declared factor in the scene spec is a contract violation (3.2.4) |
| **Rollout instrumentation** | Per-frame simulator state; the exact tensors the policy receives, sampled every N steps; per-subtask progress (reach, grasp, lift, place); for every applied action, the step of the observation it was computed from; commanded vs actual gripper width; end-effector pose at the moment the gripper closes | Feeds the contract checks, the staleness probe and the behavior analysis. The per-frame state is also required by data Rung B (5.3) |
| **Qualified reference controller** | Scripted IK or planner using privileged state (5.2 B). It must emit actions in the policy's own output format (chunk shape, frame, units). Measure its success on nominal **and on every shifted condition** before trusting it | It never looks at pixels, so its success should not change across appearance conditions. If it drops on a shifted condition, something besides appearance changed (physics or geometry). Free asset sanity check |
| **Two inference modes** | *Incident mode* is however the eval was run (parallel, shared server). *Clean mode* is serial: one env, no queueing | All router replays except the infra check run in clean mode, so server effects cannot contaminate the other replays |
| **Rollout cache** | Episodes keyed by (code hash, policy checkpoint, scene spec, seed, inference mode) | Router replays, Rung 1 and the data pipeline reuse the same episodes |
| **Replay-agreement measurement** | Run the same seed twice in clean mode, with the policy's sampling seed fixed too, and record how often the outcome matches | GPU physics and sampled action heads may not be deterministic. This number decides whether paired statistics are valid (3.2.9) |

#### 3.2.3 Pipeline overview

```
Incident (+ scene spec, training spec)
  |
  Stage 0  Reproduce (clean mode)                 -> no real failure? stop
  Stage 1  Checks from logs alone (no rollouts)   -> contract / staleness / render+physics evidence
  Stage 2  Branch split: nominal scene, current code
  |            |-- within nominal band --> Branch C (condition-specific failure)
  |            |-- clearly failed -------> Branch G (global failure)
  |            '-- degraded -------------> Branch G first, flagged likely multi-cause
  Stage 3  Decision table with fixed precedence   -> Verdict
  |
  Verdict -> hagent | data ladder | config fix | report
  After any fix: re-run the router on the fixed system (3.2.11)
```

Every check, in every stage, emits the same record, so stages can be tested and ablated independently:

| Evidence field | Meaning |
|---|---|
| `id` | Stable id; the report writer may only cite ids that exist |
| `source` | Which check produced it (`contract_cfg`, `staleness`, `replay_C2`, ...) |
| `signal` | Short machine-readable label |
| `strength` | How far outside the nominal distribution it is, calibrated on no-fault runs (0 to 1) |
| `location` | Config key, pipeline stage, file or symbol it points to, if known |
| `detail` | Human-readable sentence for the report and for hagent |

#### 3.2.4 Stage 0 and Stage 1

**Stage 0, reproduce.** Re-run the failing seeds in clean mode, 10 seeds, extended to 20 if ambiguous. Compare against the reference success rate for that condition from the failure map or the last validated run.
- Within the expected band: verdict "no significant failure", router stops.
- If the failure only appears in incident mode, record it and let Branch G's infra check handle it.
- Reuse this seed set for every replay in the incident so all comparisons are paired across conditions.

**Stage 1, checks from logs alone.** No new rollouts, run on every incident whichever branch follows. They produce evidence with a location attached, and they catch code bugs that only fire in the shifted condition (for example a wrong instruction string for one object, or a preprocessing path taken only at a different exposure).

- **Config contract.** Compare the eval config against the training spec and the scene spec. Flag chunk length, replan steps, denoising steps, control frequency, normalization stats, camera pose, image size, render settings and the instruction string. A difference that is not a declared factor is a violation. A small model can propose how keys in the two config formats map onto each other, but the comparison is a plain rule and the mapping is hand-checked once per policy.
- **Observation contract.** Compare the logged policy-input tensors with training-data statistics: per-channel means (channel swaps), value range (normalization), orientation, resolution and aspect ratio. Stronger version: push a few training-set frames through the *current* preprocessing and measure the policy's open-loop action error against the dataset actions. If it is much worse than with the reference preprocessing, the observation path is broken. This catches swaps, flips, normalization and resize bugs at zero sim cost.
- **Action contract.** Check the policy's raw outputs against the training action space (dimensions, ranges, gripper sign convention). Compare commanded vs actual gripper width and commanded vs tracked joint positions. Systematic tracking error points at gains. A gripper that never opens points at a sign or threshold bug.
- **Staleness probe.** From the logged (observation step, applied step) pairs, compute the age of each applied action. A *constant* extra delay that is also present in clean mode is code latency. A *variable* delay that grows with load and disappears in clean mode is server staleness. This is the split the bug table (6.4) needs.
- **Render and physics sanity.** Black or saturated frames, missing textures, NaN states, interpenetration, an object resting pose that differs from the logged initial state.

#### 3.2.5 Stage 2: branch split

Run the **nominal scene with the current code** in clean mode, and compare it with the historical nominal rate from the failure map (built before any bug was injected) or the last validated run.

| Nominal result | Branch |
|---|---|
| Within the nominal band | **C** (the failure is condition-specific) |
| Clearly failed (near the shifted failure rate) | **G** (the failure is global) |
| Degraded (well below nominal, well above failure) | **G first**, flagged as likely multi-cause; after any fix the router re-runs |

#### 3.2.6 Branch G: global failure

All of the following are run, and none can short-circuit the others.

- **G1, infra.** Compare incident mode against clean mode. If clean mode restores success and the staleness probe shows variable delay, the cause is infra.
- **G2, simulator.** Run the reference controller on the nominal scene. If it fails (and it was qualified), the cause is the simulator or environment: check sim build, assets and physics settings.
- **G3, pipeline isolation.** Send the reference controller's actions **through the current wrapper's action pipeline and controller** and compare with G2.
  - G2 succeeds, G3 fails: the bug is on the action side (gains, gripper, chunk handling, frame transform, latency).
  - Both succeed but the policy still fails: the bug is on the observation side or in policy loading (preprocessing, normalization, checkpoint, index mapping).
- **G4, contract evidence** from Stage 1.

The verdict is **code**, with a location whenever Stage 1 or G3 found one. Code by pure elimination is the last resort, carries lower confidence, and says so in the verdict.

#### 3.2.7 Branch C: condition-specific failure

All of the following are run.

- **C1, solvability.** Run the reference controller on the *shifted* scene. If it fails, the shifted scene itself is broken (the asset's physics or geometry changed, or the task became infeasible). Verdict: **sim or asset issue**. This is never reported as an action gap.
- **C2 to Ck, one-factor restores.** For each declared factor in the scene spec, run the shifted scene with only that factor set back to nominal. Object pose, instruction string, physics material, mesh and seed stay fixed. A material restore must change only the visual material, which is verified once per asset in Isaac. If no single factor restores success, run one replay with all factors restored. If only that works, the factors interact.
- **C_render, rendering.** Run the shifted scene with the training data's render settings, but only when the eval's render settings differ from the training spec. If they match, skip it, because Stage 1 already told you.
- **C_behavior, behavior analysis.** No new rollouts, only the logs:
  - *Implied target:* the end-effector pose when the gripper closes, compared with the true object pose across seeds. Small scatter around the right object means perception is fine. A consistent offset in one direction means a frame or extrinsic bug, even in this branch. Large scatter or the wrong object means perception failed.
  - *Failure phase* from subtask progress: reach, grasp, lift or place.

This gives the final bucket positive evidence. If nothing restores, the implied target is right, and failure is at grasp, lift or place, the evidence points to an action or skill gap. If nothing restores and the implied target is wrong, perception failed in a way no tested factor explains.

#### 3.2.8 Stage 3: decision table with fixed precedence

All evidence is collected first. The verdict is the first row that matches.

| # | Evidence | Verdict | Destination |
|---|---|---|---|
| 1 | Stage 0 shows no real drop | No significant failure | Stop |
| 2 | Render or physics sanity broken, or reference controller fails (G2 or C1), or reference controller unqualified on this condition | Sim / asset / infeasible | Report, with the replay that proved it |
| 3 | Clean mode restores and staleness is variable | Infra | Report |
| 4 | Contract violation, or G3 shows a pipeline failure, or a consistent implied-target offset, or Branch G after rows 2 and 3 were excluded | Code / config (with location if found; elimination only = low confidence) | hagent |
| 5 | Eval render settings differ from training and C_render restores | Rendering mismatch | Config fix (hagent task) |
| 6 | One factor Ci restores | Vision gap, factor i | Data ladder, rung by factor (3.2.10) |
| 7 | Only the all-factors restore works | Vision gap, interacting factors | Data ladder, Rung B |
| 8 | Nothing restores, implied target right, failure at grasp, lift or place | Action / skill gap | Report |
| 9 | Nothing restores, implied target wrong | Perception unresolved (untested factor or new sensor modality; the router cannot tell which) | Report, optional factor search (3.2.12) |
| 10 | Anything else, or conflicting evidence | Undetermined | Report |

Why this order:
- Sim and infra come first because they contaminate every other replay.
- Code beats vision because a code bug can look like a vision gap, and a code fix is cheaper than data generation.
- Rendering beats vision because it is a config fix, not a data fix.

**Partial fixes.** If the winning row's replay only partly restores success, the verdict carries a "likely second cause" flag. After the fix the incident re-runs through the router (3.2.11).

#### 3.2.9 Statistics and cost

- **Three outcomes per replay**, from its success rate with a confidence interval, against two anchors: the failure rate from Stage 0 and the nominal band from Stage 2.
  - *Restored:* interval overlaps the nominal band and excludes the failure rate.
  - *Partial:* interval excludes the failure rate but sits below the nominal band.
  - *No effect:* interval includes the failure rate.
  - If the interval straddles a boundary, extend from 10 seeds to 20-30 before deciding.
- **One shared definition of "clearly better".** The router and Rung 1 both use `ladder/stats.py`.
- **Paired or unpaired.** If the replay-agreement measurement (3.2.2) is high, use a paired test on discordant seeds. Otherwise use unpaired tests and expect to need more seeds.
- **Several replays per incident raise the chance of a false "restored".** Do not correct this with a formula. Calibrate: run the whole router many times on no-fault incidents (nominal policy, nominal scene) and on known cases, then set thresholds so the false-verdict rate on no-fault incidents stays under a target you choose (for example 5%). Report that number.
- **Scale.** With 10 seeds only large effects are detectable. The failure map deliberately picks large drops (about 80% down to under 40%), so 10 is a reasonable start, with extension on ambiguity.

Cost per incident, in simulator episodes:

| Part | Episodes |
|---|---|
| Stage 0 | 10-20 |
| Stage 2 nominal | 10-20 |
| Branch G (incident vs clean mode, reference controller twice) | about 30-50 (reference-controller episodes are cheap: no policy inference) |
| Branch C (C1, 2-4 factor restores, optional render replay) | about 40-100 |

That is roughly 60-150 episodes per incident, and the cache lets Rung 1 reuse some of them. Stage 1 checks and C_behavior cost nothing extra.

#### 3.2.10 Fix selection is a rule table, not a model decision

After routing, the choice of fix is a documented rule, not a model decision:

| Router verdict | Fix |
|---|---|
| Code / config bug | hagent patch + ladder |
| Rendering mismatch | Config edit (hagent task), then Rung 1 |
| Vision gap, brightness / contrast / color shift | Rung A of the data ladder (augmentation), default recipe in 5.4 |
| Vision gap, glossy material, camera shift, distractors | Rung B (Isaac randomization), escalate to Rung C (Cosmos) if needed |
| Vision gap, interacting factors | Rung B |
| Perception unresolved, needs new sensor modality | Report only |
| Action gap | Report only (stretch: scripted/Mimic data) |
| Infra / sim / asset issue | Report only, with the replay that proved it |

The factor that restored success in 3.2.7 is the specification for the data pipeline: it names which factor to randomize or regenerate.

#### 3.2.11 Hand-off and the re-run loop

What each destination receives:
- **hagent:** the verdict as a *hypothesis to verify first*, the location evidence, the repro command and the failing-rollout summary. Nothing from the bug ledger or the clean commit (3.3). Router evidence is fine to share, because it was discovered, not leaked.
- **Data ladder:** the factor name (which picks the rung), the failing seeds, the nominal rollouts to build from, and the held-out conditions to validate on.
- **Report:** the full verdict, the replay table and the evidence list, so a reader can see why the system stopped.

**After any fix, the incident goes back through the router.** The re-run does three jobs:
1. It is the final verification that the cause is gone, not just masked.
2. It catches a second cause. A fix that only partly restores success comes back as "likely second cause" and the next pass can route it elsewhere.
3. It makes Rung 2 simple (Section 4): Rung 2 reads the router's evidence from the re-run and is not a second, separate diagnoser.

When there are two causes, **fix code before vision.** A code bug can make the policy look worse under the shifted condition than it really is, so generating vision data first can mean training on a failure that is partly not visual.

Cap the number of router passes per incident (default two or three). If the cap is hit, the incident is reported as unresolved with all evidence.

#### 3.2.12 Where the models fit

None of them decides a route. Prices below are from the Token Factory price list you captured (per 1M tokens, input / output); check them again before relying on them.

| Model | Job | Price |
|---|---|---|
| Nemotron 3 Super 120B | Turns verdict plus evidence into the hagent task prompt and the incident report, citing evidence ids only (strict JSON, retry once, fall back to evidence ranked by strength). Acts as the Rung 2 judge. When the verdict is "undetermined", picks extra experiments from a fixed menu (another factor restore, more seeds, a state-reset test) | $0.3 / $0.9 |
| Nemotron 3 Nano 30B | Proposes the key mapping between training and eval config formats for the contract check. Summarizes long logs. Batch pricing for offline sweeps | $0.06 / $0.24 |
| Cosmos3-Super-Reasoner | Used only for verdict row 9. Show it nominal vs failing **policy-input** frames (after preprocessing) and ask what visually differs. Each answer becomes a new factor to restore, and the replay decides | $0.1 / $0.3 |
| Nemotron Nano V2 12B | Cheap second opinion on the same frames, if it turns out to be the vision-language variant | $0.07 / $0.2 |

A rough size check: an aggregator call with 20k tokens in and 2k out on Nemotron 3 Super is about one cent. LLM spend per incident is cents, and simulator time is the real cost. That means self-consistency (for example 3 samples with a majority vote) is affordable on any model step.

Things to verify before relying on them:
- Whether the Token Factory endpoint for Cosmos3-Super-Reasoner accepts video, or only images. The public model card lists text, image and video input, with mp4 at about 4 fps recommended. Sampled frames are the fallback.
- Whether the "Nemotron-Nano-V2-12b" product on the price list is the vision-language variant. The price list does not say.

**Optional factor search (row 9 only).** The reasoner proposes one or more candidate visual factors. Each is added to the scene spec and replayed through C2 to Ck. Only a replay can move the verdict out of "perception unresolved".

#### 3.2.13 Localization for the code route

In order, stopping when one is confident:
1. **Stage 1 contract hits** name the exact config key, and G3 narrows the bug to the observation side or the action side. Free, and often enough.
2. **Report writer (Nemotron 3 Super)** merges everything into ranked file and symbol candidates. It may only name files and symbols that exist in the repo or appear in the evidence.
3. **Hunk bisect (optional, only if a real known-good history exists).** Split the diff into hunks, revert half, replay about 5 seeds, recurse. It handles interacting hunks poorly, so cap the depth and report "bisect inconclusive" instead of guessing. This is for localization only and is not used for routing.

#### 3.2.14 Module layout and testing without the simulator

Each check is a module with the same interface: input an incident, output a list of evidence records. The decision table is a pure function of the evidence and the replay table.

```
router/
  schemas        Incident, Evidence, Verdict
  prereq/        scene spec, training spec, rollout cache, replay-agreement tool
  stage0         reproduce
  stage1/        cfg_contract, obs_contract, action_contract, staleness, sanity
  branches/      branch_g, branch_c (replay runners)
  behavior       implied target, failure phase
  stats          shared with ladder/stats
  decide         decision table (pure function)
  report         LLM report writer and hagent prompt builder
  router         orchestrates the stages and the re-run loop
```

Test without Isaac first:
- Unit-test each Stage 1 check on synthetic traces (for example an action trace with a known delay, and check the staleness probe recovers it).
- Test `decide` with mocked replay tables, including noisy, borderline and conflicting ones.
- Test the whole router with a mocked rollout function whose success rates are set by the injected fault, so routing logic is checked in seconds.

#### 3.2.15 Limits to state in the paper

- The router cannot separate an action gap from a new-modality need, nor an action gap from an untested visual factor. It reports "action / skill gap" or "perception unresolved" and stops there.
- Everything rests on the reference controller. Its qualification numbers go in the paper.
- Code verdicts reached by elimination are lower confidence and are reported separately from code verdicts with a located cause.
- Multi-cause incidents need one router pass per cause (3.2.11).
- Thresholds are calibrated per policy and task, so a new policy needs a new calibration run.

### 3.3 Repair agent = hagent (already built)

Mapping the existing hagent to the ladder:

| Ladder role | hagent component |
|---|---|
| Diagnose and plan | Big model, one advisor call (`plan_at_step`) |
| Edit code, run commands | Small model main loop, bash only |
| Rung 0 verification | Existing final reviewer (big model, one call) plus static checks and a smoke run via bash |
| Rung 2 triage | Big model, one call (see 4, Rung 2) |
| Stuck / loop protection | `detector.py` escalation ladder (already done) |
| Budget control | `max_steps`, `max_cost`, `max_big_calls` |
| Safe execution | **To do:** Docker/Podman env implementing the same `execute / git_diff / git_diff_hash` interface |

Things to add for this project:
- A **task prompt template** that includes: failing task, the router verdict **as a hypothesis to verify first** (never as established fact), the location evidence (config key, observation side vs action side), the repro command, a failing-rollout summary (success rate, action trace stats), and the repo path of the controller/wrapper code. It contains nothing from the bug ledger or the clean commit. Router evidence is safe to share because it was discovered, not leaked.
- A **shell tool for the ladder**, e.g. `run_rung1 --seeds 50`, so the agent can trigger simulator validation itself and read a compact result (pass rate, confidence interval, top error mode).
- A small change to the reviewer: besides APPROVE/REJECT, have it emit a **pass probability**. It is only logged next to the real Rung 1 outcome (see 4.3), never used to gate anything.
- Borrow ideas from SWE-agent's agent-computer interface (search, show diff, line edits) only if bash-only proves too clumsy. Benchmark first.

**No leakage from the bug injection.** For every injected-bug incident, the container the agent works in must not reveal the answer:
- Squash the injected change into a single base commit (no `git log` / `git diff` path back to the clean version).
- The bug ledger, the clean commit and the injection scripts live outside the container.
- Strip or neutralise tell-tale comments, flag names and file timestamps that point at the injected line.

Otherwise the agent can find the bug by reading history, and the repair numbers are meaningless.

## 4. Verification ladder (for code fixes)

Cheap checks first, expensive simulator runs last. There are three rungs. (An earlier version had a separate "LLM soft verification" rung between static checks and the simulator. It was cut: see Section 10.)

### Rung 0: static checks + smoke run + reviewer (cheap, no full simulation)
- Lint, tests, diff sanity, "did the patch touch the right file".
- A short smoke run (a few steps) that catches hard crashes such as tensor shape mismatches. This is the easy end of the bug spectrum and does not need the router.
- The hagent reviewer call lives here. It can reject obviously wrong or incomplete patches and send the agent back to work (capped by `max_review_rejections`).
- Never blocks on an unparseable reviewer reply (counts as APPROVE), so this rung cannot loop.

### Rung 1: staged simulator validation in Isaac Sim
The only rung that produces ground-truth labels per episode. To save sim time without a predictor, run seeds in stages and stop early when the answer is clear.

1. **Stage A: small batch** (e.g. 10 seeds, headless, parallel). Compute success rate with a confidence interval.
2. **Decision after Stage A:**
   - clearly at the broken baseline -> **fail**, return to the repair agent
   - clearly at nominal -> **pass**
   - ambiguous -> go to Stage B
3. **Stage B: full batch** (e.g. 50 seeds, or more if the interval is still wide).
4. **Decision after Stage B:** pass or fail as above; if still ambiguous -> **Rung 2**.

Rules:
- Use **paired, fixed seeds** across before/after so the comparison is paired (subject to the replay-agreement measurement in 3.2.2).
- Verdict comes from success rate with a **confidence interval**, not a single number.
- Use the **same statistics module as the router** (`ladder/stats.py`), so there is one definition of "clearly better" across Rung 1 and the router (3.2.9).
- Thresholds must be **calibrated on the nominal, unbroken policy**, not copied from this plan:
  - clear pass: success within error bars of nominal
  - clear fail: near the broken baseline
  - otherwise: ambiguous (Stage B, then Rung 2)
- Because the data are looked at twice (after Stage A and after Stage B), use intervals that account for the early look (e.g. a stricter Stage A threshold or a simple sequential-testing correction), and report the resulting error rates.
- Run Rung 1 with inference settings that do not introduce stale actions (watch policy-server batching when many parallel envs share one server), otherwise infra effects contaminate the verdict.
- Use Modal or the ARES-style harness for scale if local GPUs are not enough.

What the simulator does and does not guarantee: each episode's success label is exact, so there is no classifier error of the kind an LLM predictor would add. The verdict on a patch is still a statistical decision from a limited number of seeds, so a wrong pass or wrong fail is possible, especially on an early stop. That error rate is what the calibration step measures.

### Rung 2: triage the ambiguous middle
**Owner: the big model, one call per incident.** Rung 2 is not a second diagnoser. When Rung 1 stays ambiguous after a patch, the router is re-run on the patched code (3.2.11) and Rung 2 reads its output: the verdict, replay table and evidence list, plus per-subtask progress (reach/grasp/lift/place), action-trace statistics, the Rung 1 success rates with intervals, and a few key frames from failing episodes. Structured evidence, not raw video. It returns one of:
- **simple code bug** -> back to the repair agent with the new evidence
- **hidden data gap** -> hand to the vision data pipeline (Section 5)
- **unclear** -> stop and report the incident as unresolved

This is the one place where a model judges a cause, so:
- the call counts toward `max_big_calls` and appears in the cost metric (Section 7);
- every Rung 2 decision is logged, and a sample is checked by a human **for reporting only** (agreement rate goes in the paper; the human does not sit in the loop);
- a Rung 2 "data gap" verdict must match a router one-factor or all-factors restore (decision rows 6 and 7 in 3.2.8) before any data is generated. Rung 2 cannot create a vision route on its own.

Loop protection comes from hagent's caps and detector, plus a **max number of ladder iterations** per incident.

### 4.3 Optional side analysis: reviewer calibration
Because every patch that reaches Rung 1 gets a real label, the reviewer's logged pass probability can be compared to the outcome at zero extra sim cost. If it shows real signal (e.g. AUC well above 0.5), report it. If not, say so. This is a footnote, not a stage, and it does not need escape sampling.

## 5. Vision data-gap fix (our main data contribution)

### 5.1 Why vision only, and what the evidence is
Per the budget decision: Dreams-style generation and RL for action gaps are expensive, so we implement the vision path fully.

**Where the diagnosis comes from.** The router's counterfactual replay (clean appearance flips the result to success) is the evidence that the gap was perceptual. It is *not* which module we fine-tune: a vision-caused failure can often also be fixed by tuning the action side, because it can learn to map shifted features to correct actions (like fixing a car that pulls left by steering slightly right instead of realigning the wheels). So the choice of which module to tune is an engineering and cost decision, and the report must not claim that "the action head was untouched" proves the policy was fine.

### 5.2 Source of clean nominal rollouts (user-selectable)
The tool offers the options below, with their prerequisites. For the time-boxed version, implement one fully and expose the others as documented stubs.

| Option | Prerequisite | Best when |
|---|---|---|
| **A. The VLA's own successful rollouts** (rejection-sampled) | Nominal success reasonably high (about 80%+) | Default and cheapest. Needs no oracle |
| **B. Scripted policy** (IK / motion planner such as cuRobo) | Structured task with known goal states | Pick, place, stack, bin picking. Doubles as the router's reference controller (3.2) |
| **C. Teleop demos + Isaac Lab Mimic** | A handful of demos (about 5-10) and subtask annotation | Manipulation tasks with clear subtasks |
| **D. Small RL oracle** (privileged state) | Reward design, state-based env, time to train | Tasks where A-C are not possible |

Notes:
- With options B-D, actions come from the oracle/planner/human, which is exact for the simulator. With option A, actions are the policy's own successful actions (filtered behavior cloning), so do not describe them as expert ground truth.
- Oracle policies and pretrained Isaac Lab checkpoints are typically state-based RL policies without cameras. Rollouts must be re-run with cameras enabled and the observations logged.
- **Log full simulator state per frame** (robot joint state, object poses, gripper state) alongside observations and actions. Rung B of the data ladder needs it (5.3).
- Convert the result to the policy's training format (see 3.1) and keep a held-out split.

### 5.3 Data-cost ladder for appearance shifts
Try the cheapest rung first and escalate only if held-out success does not recover. The router tells you which factor to target: the factor whose restore flipped the replay (3.2.7) names what to randomize or regenerate, and the starting rung comes from the rule table in 3.2.10.

| Rung | What | Cost | Good for | Cannot do |
|---|---|---|---|---|
| **A. On-the-fly augmentation** | Color jitter, brightness/contrast/saturation, blur, crop on the existing dataset during training | No new data, no sim cost | Mild lighting / color shifts | Glare, specular materials, camera geometry, new objects |
| **B. Isaac randomization** | Re-render the same rollouts with randomized materials, HDRIs, lights, camera jitter, distractors | Sim time, exact | Glossy/dark materials, harsh lighting, camera shifts | Photorealism beyond the simulator |
| **C. Cosmos Transfer** | Generative re-rendering of clean rollouts to change only the failing visual factor | Heaviest GPU cost | Photorealism where Isaac cannot express the change | Anything that alters geometry or contact timing |

**How Rung B re-renders.** Do not re-simulate by replaying recorded actions open-loop: GPU physics is not guaranteed to be deterministic, so the trajectory can drift away from the recorded one and the actions no longer match what is on screen. Instead, **replay the logged per-frame state** (set robot and object poses each frame, physics stepping off or kinematic) and only change appearance. Check that replayed poses match the logged ones within a tolerance; drop episodes that do not.

For rungs B and C, pair the new frames with the original recorded actions (no inverse-dynamics step), and apply quality gates: reject clips where object identity, geometry or contact timing changed; keep a sample for human spot-check. Be careful with camera-geometry changes: the original actions only stay valid if the shift is in appearance, or the controller's camera assumptions are handled consistently.

### 5.4 Training recipe (time-boxed default)
- **Default:** keep the vision backbone and language backbone frozen; fine-tune the projector and the action head (diffusion/DiT) using the policy's official tune flags, with the Rung A/B/C data and augmentation.
- **Optional ablation (only if the pilot fails or time permits):** also tune the vision side (full unfreeze of the vision encoder, or LoRA on it) and compare held-out success and nominal success.
- **LoRA is optional, not core.** In the public GR00T repo, official LoRA support was reported as dropped between 1.5 and 1.6 with a request to bring it back (issue #568, with a community N1.7 LoRA repo linked and unvetted). If the target policy is 1.6+, LoRA depends on unofficial code or your own PEFT wiring. Use it only if VRAM forces it.
- **Do not fix a LoRA rank from lore.** Public reports are inconclusive: one user's poor result at rank 2 on N1.5 had unclear causes (capacity, training steps, or deployment code; issue #367), and a rank-64 attempt on LIBERO failed until config/evaluation problems were fixed (issue #136). Pick one rank if LoRA is used, label it untuned, and do not run a rank sweep within this project's budget.
- Record the actual hyperparameters, GPU and VRAM used.
- Verify the real module names and flags in the model code before committing to any of the above (see 3.1).

### 5.5 Data mixing
Keep a mix of original nominal data plus the new vision-shifted data so the policy does not forget the nominal case. Start with something like 50/50 and **tune on a validation set**; do not treat any fixed ratio (60/40, loss weights 1.0/1.5) as known-good. Per-sample gradient routing by data source is an optional refinement.

### 5.6 Validation of the vision fix
Use the same staged Rung 1 procedure (Section 4) on **held-out** conditions after fine-tuning, with the same paired seeds as the failing baseline. Also re-run the nominal condition to check for regression (fine-tuning sometimes makes things worse than the base checkpoint).

## 6. Experiment design

### 6.1 Pick a failure that is worth fixing (failure map)
Good target: nominal success high (about 80%+), shifted success low but nonzero (about 10-40%), caused by distribution shift, and cheap to regenerate.

1. Build a grid, e.g. 6 objects x 4 lighting/HDR x 3 camera offsets.
2. Run 10-20 episodes per cell with the pretrained policy.
3. Keep cells where nominal is high and the shifted condition drops below about 40%.
4. Candidate families ranked by cost: lighting extremes (dark and glare), object appearance/material, camera or wrist-camera offset, then clutter. Avoid deformable objects and long-horizon composites in the first version (they drift into skill gaps).
5. Track per-subtask progress (reach, grasp, lift, place) so a failure can be located (for example, grasp succeeds but lift fails), which helps separate vision issues from physical or control ones. The same signal feeds Rung 2 triage and the router's behavior analysis (3.2.7).
6. Build every grid cell as **nominal plus named factor changes** (the scene spec, 3.2.2), and log the rollout instrumentation listed there. Also record the **nominal success rate before any bug is injected**: the router compares against it, and it is the historical reference for the Stage 2 branch split.
7. Qualify the reference controller on every cell. Its success should not change with appearance. A cell where it drops has a physics or geometry change, not just a visual one, and should be dropped or fixed.
8. Measure replay agreement (same seed twice in clean mode) so the router knows whether paired statistics are valid.

### 6.2 Pilot gate (early, before building the rest)
Generate 50-100 episodes for the failing family, fine-tune with the default recipe (5.4), and check **held-out** improvement. If data does not fix it, learn that now. If it does, you already have the first headline number. Run the optional vision-side ablation only if the default fails or time remains.

### 6.3 Held-out evaluation
Train on some failing variants, test on others (different glossy objects, unseen HDRs, unseen camera offsets). Without this, results could just be memorization.

### 6.4 Bug injection suite (for the code-repair side)
Prefer **silent** bugs: the policy runs but fails. Several are modeled on faults reported in public GR00T/LIBERO threads (issue #136, where success of 90%+ was reported only after fixes of this kind), so the suite is grounded in real failures. Verify equivalents in your policy version.

Each bug has **exactly one expected router verdict**, so the confusion matrix is unambiguous.

| Bug | Symptom | Expected verdict | Origin |
|---|---|---|---|
| RGB/BGR channel swap | Looks like a vision failure | Code bug | Planned |
| Image flipped/rotated 180 degrees vs. training preprocessing | Looks like a vision failure | Code bug | Reported in #136 |
| Gripper action sign inverted or threshold error | Reaches but cannot grasp, gripper stuck closed | Code bug | Reported in #136 |
| State/action index slicing or modality mapping wrong | Erratic or off-target motion | Code bug | Reported in #136 |
| Replan steps / action chunk length / denoising steps mismatch | Jitter, drift, long-task failure | Code bug | Reported in #136 (chunk-length sensitivity also reported elsewhere, unverified) |
| Action latency or control-frequency mismatch in the controller code | Hesitation, late corrections | Code bug | Planned |
| Stale actions from a shared, batched policy server (no code change; injected via server config/load) | Hesitation, late corrections, only under parallel eval | **Infra** | Planned |
| Camera-to-wrist frame transform offset | Looks like a vision failure | Code bug | Planned |
| Joint gain decay | Sluggish motion | Code bug | Planned |
| Normalization-stats mismatch | Systematic offset | Code bug | Planned |
| Wrong instruction template for one object only (fires only in the shifted condition) | Looks like a vision failure on that object | Code bug | Planned |
| Preprocessing path taken only at one resolution or exposure | Looks like a vision failure on that condition | Code bug | Planned |
| Mild normalization offset (degrades nominal but does not break it) | Partial drop everywhere | Code bug, flagged multi-cause if combined with a shift | Planned |
| Tensor shape mismatch | Hard crash | Caught at Rung 0 (not routed) | Easy case |

Latency in the controller code and stale actions from the server look the same on video. They are separated by where the fault is injected (code vs. server load), and the router separates them with the staleness probe (constant vs variable delay) and the clean-mode vs incident-mode replay (3.2.4, 3.2.6).

Inject by patching controller/wrapper config flags or code strings (robosuite's `controllers/` module is a good reference design for how to parameterize these). Keep a ledger **outside the agent's container** (see 3.3): bug id, injected diff, expected symptom, expected router verdict.

### 6.5 Router evaluation set

The 6.4 bugs only exercise the code rows of the decision table (3.2.8). To measure the router honestly, every row needs cases, each with **exactly one expected verdict**:

| Class | Example cases | Expected verdict |
|---|---|---|
| Code, global | the 6.4 list | Code (with location where a contract or G3 finds one) |
| Code, condition-specific | wrong instruction template for one object; preprocessing path at one resolution | Code |
| Code, partial | mild normalization offset | Code |
| Vision | dark lighting, glare, glossy material, camera offset, distractors (from the failure map) | Vision, factor named |
| Interacting vision factors | two factors that only fail together | Vision, interacting |
| Rendering mismatch | eval render settings differ from the training data (if the training data allows) | Rendering mismatch |
| Sim / asset | broken collision mesh, wrong friction on the shifted asset, changed physics step | Sim / asset |
| Infra | stale actions from a shared batched server | Infra |
| Action gap | a cell where the reference controller succeeds but the policy cannot grasp (unseen geometry) | Action / skill gap |
| Combined | code bug plus a lighting shift | Code, flagged "likely second cause"; after the fix the re-run returns vision |
| No fault | nominal policy, nominal scene, unlucky seeds | No significant failure |

Protocol:
- Split the set into a **calibration set** (to set thresholds, 3.2.9) and a **test set**.
- **Hold out some bug types entirely**, so the contract checks are tested on bugs they were not written for.
- For injected bugs, hide the injection inside a **multi-commit history with benign changes** when scoring the commit-swap oracle, so the oracle is not trivially one hunk. The oracle is used for scoring only and is never an input to the router (3.2.1).
- Report the same router metrics in two settings, with and without an oracle clean commit available. The no-oracle number is the headline.

### 6.6 Oracle checks (optional, use carefully)
- A **privileged-state baseline** (feed ground-truth object state instead of pixels) is only valid if the policy actually accepts state inputs. Most end-to-end VLAs do not, so this may require a separate teacher policy. Treat it as optional supporting evidence.
- Pretrained Isaac Lab checkpoints (Cartpole, Ant, ANYmal, Franka reach/lift, hands) are state-based RL policies. They are not evidence about the VLA's performance. Only something like Franka lift is a plausible privileged baseline, and only if your task matches it.
- The counterfactual-replay router already gives a cheaper, more direct isolation of vision vs control.

## 7. Metrics to report

| Metric | Why |
|---|---|
| Router confusion matrix on the evaluation set (6.5), reported with and without an oracle clean commit | Shows diagnosis works, including vision-looking code bugs, infra confounders and combined cases |
| Router **undetermined rate** and **accuracy when it commits to a verdict** | A router that abstains honestly is better than one that guesses; both numbers are needed |
| **False-verdict rate on no-fault incidents** (the calibration target, e.g. under 5%) | Controls the multiple-replay problem (3.2.9) |
| Router cost per incident: simulator episodes and LLM spend | Backs the claim that routing is cheap next to a wrong fix |
| Router ablations: elimination only (earlier design) vs plus Stage 1 positive checks vs full router; and a **VLM-only baseline** where a video model guesses the route | Shows each stage earns its place, and why replays are needed |
| Localization hit rate for code verdicts (does the named config key or stage match the injected fault) | Shows the hand-off to hagent is useful, not only correct on route |
| Repair success rate and Rung 1 pass rate after repair | Core code-repair result |
| Cost per fix (tokens by small vs big model, including Rung 2 calls, sim GPU-hours) | Backs the cheap-first design; use hagent's `usage` field |
| **Cost and held-out success vs. a fixed-recipe baseline** (always Cosmos + fine-tune, or always large randomized fine-tune) | Directly tests the headline claim (Section 1) |
| Sim seeds saved by staged validation vs always running the full batch, plus the measured wrong-pass / wrong-fail rate of early stopping | Shows the early-stop design pays off without hiding its error |
| Relative improvement on held-out vision conditions, with error bars (e.g. 25% -> 65%) | Better than absolute 90% thresholds at this scale |
| Nominal success before vs. after each fix, plus the base checkpoint as baseline | Detects forgetting and fine-tuning regressions |
| Which rung of the data-cost ladder was sufficient per failure | Backs the cheapest-tool-first claim |
| Rung 2 decisions: count, and agreement with human spot-check | Makes the one model-judged step auditable |
| Ablations (as time permits): small-only vs big-only vs hybrid agent; default recipe vs vision-side tuning | Justifies each component |
| (Optional) Reviewer pass-probability vs Rung 1 outcome (AUC) | Free side analysis, see 4.3 |

## 8. Build order (time-boxed)

The estimates below were originally scaled to roughly 12 weeks of one person's time. **Replace them with your real deadline**, keep the proportions, and treat steps 1-3 as the likeliest to overrun (policy setup, format conversion and the first fine-tune usually take longer than planned). The router is the part you cannot afford to get wrong, so step 6 and the prerequisites it depends on are not the place to cut first.

**Minimum viable version (must finish):**

| # | Step | Estimate |
|---|---|---|
| 1 | Pick policy + task; read the model code; confirm module names, tune flags and data format (3.1). Get a scripted reference controller running for the task (5.2 B), emitting actions in the policy's output format, and **extract the training spec** (3.2.2) | 2 weeks |
| 2 | Build the failure map (6.1), including a lighting sweep, subtask tracking, per-frame state logging, and the **router prerequisites** (3.2.2): scene spec, rollout instrumentation, clean vs incident inference modes, rollout cache, reference-controller qualification, replay-agreement measurement | 2 weeks |
| 3 | Pilot gate (6.2) with data-ladder rung A, then B if needed, and the default recipe (5.4) | 2 weeks |
| 4 | Run hagent against a live model and benchmark `small` / `big` / `hybrid` on a handful of injected bugs from 6.4 (README says it has only been tested with scripted models so far) | 1 week |
| 5 | Containerize the hagent env (Docker) with the no-leakage setup (3.3), and add the `run_rung1` tool with staged seeds | 1 week |
| 6 | Implement the router (3.2), in this order: the decision table tested on mocked replay results, including borderline and conflicting tables, before any sim time; Stage 0, Stage 2 and Branch C; Branch G with the staleness probe; Stage 1 contract checks; calibration runs on no-fault and known incidents to set thresholds; build the router evaluation set (6.5) | 2 weeks |
| 7 | Implement Rungs 0-2, Rung 2 reading the router re-run, Rung 2 logging, the reviewer pass-probability logging, the router re-run loop and the iteration cap (3.2.11) | 1 week |
| 8 | Full evaluation including router metrics (Section 7), the fixed-recipe baseline, the LLM report layer (3.2.12), write-up, and the limitations section | 2 weeks |

That totals 13 weeks, which is **1 week more than the 12-week assumption**. The router prerequisites in steps 1 and 2 and the calibration in step 6 are what added time compared with a router built only from replays. These are my planning figures, not measured ones, so replace them with your real deadline. If you must stay at 12 weeks, the levers in order are: cut the stretch list entirely, shrink the bug suite and the router evaluation set (6.4, 6.5), run fewer factor restores per incident (3.2.7), and skip the optional factor search (3.2.12). If step 3 slips by more than a week, apply those before touching steps 6-8.

**Stretch (only if time remains):**
- Cosmos Transfer rung with quality gates.
- Additional clean-data source options beyond the one implemented (5.2).
- Vision-side tuning or LoRA ablation.
- Scripted/Mimic data as a cheap action-gap fix on a structured task.
- Reviewer calibration AUC.
- Router extras: the optional factor search with the reasoner (3.2.12), and hunk bisect for localization when a real known-good history exists (3.2.13).

## 9. Risks and open questions

- Does the chosen policy actually fail in a way data can fix? (Pilot gate answers this.)
- Is Cosmos Transfer available and affordable at the scale needed? Fallback: Isaac-only randomization. If Cosmos is not run, the fixed-recipe baseline in Section 7 becomes "always large randomized fine-tune".
- Action-gap detection: the router can say "action / skill gap" or "perception unresolved", but cannot tell *which* skill is missing, nor separate an action gap from a new-modality need or from an untested visual factor. Say so (3.2.15).
- The router leans on the reference controller for sim-vs-policy separation (G2, C1) and for pipeline isolation (G3). If it is flaky on the task, those splits become unreliable. Qualify it on nominal and on every shifted condition first (3.2.2). Code verdicts reached by elimination alone are lower confidence and are reported separately.
- Multi-cause failures: the router flags "likely second cause" when a replay only partly restores success, and the incident re-runs after each fix (3.2.11). Two causes that happen to restore fully on one replay can still be missed. Include combined cases in the evaluation set (6.5) and report the result.
- **Replay determinism.** If the same seed does not reproduce the same outcome in clean mode, paired statistics are invalid and the router needs more seeds. Measure replay agreement before trusting any replay (3.2.2).
- **Visual-only restores.** The one-factor restores are only meaningful if restoring a factor changes the visual material and nothing else (mesh, mass, friction). Verify this once per asset in Isaac, otherwise a "vision" verdict can really be a physics change.
- **Training spec extraction.** The contract checks are only as good as the training spec. If the policy's training config does not expose normalization stats, camera poses or render settings, those checks fall back to weaker evidence. Check this per policy in step 1.
- **Router thresholds are per policy and task.** They are calibrated on no-fault and known incidents (3.2.9). A new policy or task needs a new calibration run. The false-verdict rate on no-fault incidents must be reported.
- **Model endpoints for the router.** Whether the Token Factory endpoint for Cosmos3-Super-Reasoner accepts video, and whether "Nemotron-Nano-V2-12b" on the price list is the vision-language variant, are not verified (3.2.12). Both only affect the optional row-9 path, so the core router does not depend on them.
- Rung 2 is a model judgment. Its error rate is measured, not assumed away.
- Leakage: if any trace of the injection reaches the agent (history, ledger, comments), repair results are invalid. Audit one container by hand before the main evaluation.
- LoRA availability and the tune flags depend on the policy version. Verify against the actual model code before committing to any module-level design (see 3.1 and 5.4).
- The hypothesis "for lighting shifts, tuning the action side (with augmentation) is enough" is untested for this policy. Present it as a hypothesis checked by the pilot, not as a fact.
- Staged validation: the early-stop thresholds and stage sizes need calibrating on the nominal policy, otherwise Stage A can stop too early on noisy results.
- State replay for Rung B assumes the logged state is complete enough to reproduce the scene (including articulated objects and gripper contact). Validate on a few episodes before generating at scale.
- Camera-geometry changes in generated data may invalidate paired actions. Keep the quality gate.
- Multi-task fine-tuning may interfere with single-task performance; prefer per-task fine-tuning or monitor nominal success.
- Facts taken from chat summaries are not evidence. Specific benchmark numbers, issue numbers and thresholds attributed to the repo or Isaac Lab-Arena (other than issues #568, #367, #136, which were read directly) must be checked before they are cited.
- External repos and benchmarks mentioned in earlier notes (EmbodiedSWE, ARES, SERA, LLM4ADSTest, FailBench/MiraBench) were not verified. Check each exists, is maintained, and fits before depending on it. ARES, robosuite, lerobot and Modal are the safest starting points.
- Model names, context sizes and prices in the hagent example config are placeholders.

## 10. What changed, and what was dropped

### Changes in this revision (router redesign, 3.2)
- **The router is now deterministic checks plus counterfactual replays, combined by a fixed decision table with explicit precedence** (3.2.8). Models write reports and propose factors to test; they never choose a route (3.2.1).
- **No early stop across hypotheses.** Stopping at the first replay that flips made the verdict depend on test order (a code bug could be labeled vision). All single-factor replays now run, and precedence is fixed: sim, then infra, then code, then rendering, then vision.
- **Added Stage 1 checks from logs alone** (config, observation and action contracts, staleness probe, render and physics sanity). They give positive, located evidence for code bugs, including ones that fire only in the shifted condition, which the earlier two-branch router would have called "vision gap".
- **Branch split has three outcomes** (within nominal band, clearly failed, degraded), not yes/no. Degraded goes to the global branch first and is flagged as likely multi-cause.
- **Pipeline isolation (G3)**: reference-controller actions are sent through the current wrapper to say whether a bug is on the observation side or the action side, so "code" comes with a location instead of only "by elimination".
- **One-factor restores replace the single "clean appearance" replay**, so the router names which factor caused the failure. That factor picks the data-ladder rung and specifies the data to generate.
- **Solvability check (C1)**: the reference controller also runs on the *shifted* scene, so a broken asset is reported as sim/asset instead of "action gap". "Nothing restores" no longer defaults to action gap: it needs a behavior-analysis signal (implied target, failure phase), otherwise it is "perception unresolved" or "undetermined".
- **Reproduce step (Stage 0)** so a noisy draw from a 70% policy does not open an incident.
- **Three-valued statistics, replay-agreement measurement, shared stats module with Rung 1, and calibration on no-fault incidents** instead of a fixed flip rule at 10 seeds (3.2.9).
- **Re-run loop and Rung 2 simplification**: after any fix the router re-runs, and Rung 2 reads that output instead of acting as a separate diagnoser (3.2.11, Section 4). Code is fixed before vision when two causes are present.
- **Router evaluation set with a verdict for every row of the table**, calibration/test split, held-out bug types, and an oracle-free headline number (6.5). New router metrics in Section 7.
- **Prerequisites pulled into build-order steps 1-2** (scene spec, training spec, instrumentation, qualified reference controller, clean vs incident inference modes, rollout cache), and the schedule now totals 13 weeks with the levers to get back to 12 stated openly (Section 8).
- **Model roles fixed** (3.2.12): Nemotron 3 Super for reports and Rung 2, Nemotron 3 Nano for config mapping and log summaries, Cosmos reasoner only for the optional row-9 factor search.

### Changes in the previous revision
- **Router restructured into two branches** keyed on "does the nominal scene also fail?" (3.2). The old table routed any failure on a known-good scene to "simulator issue", which would have misclassified most code bugs. Simulator issues now require a known-good reference controller to fail too.
- **Code bugs are routed by elimination**, not by swapping in a last-known-good commit. A commit swap would make the repair a `git revert`; it is kept only as an evaluation-only oracle for scoring the router.
- **No-leakage setup for injected bugs** (3.3, 6.4): squashed history, ledger outside the container, no tell-tale comments.
- **Rung 2 has an owner**: one big-model call on structured evidence, counted in cost, logged, spot-checked by a human for reporting only, and a "data gap" verdict re-confirmed by the router before data is generated (4).
- **Headline baseline changed** from "retraining from scratch" to a realistic fixed expensive recipe, with a matching metric (1, 7).
- **Rung B re-renders by replaying logged state**, not by re-simulating recorded actions, because GPU physics may not be deterministic (5.2, 5.3).
- **Rung 1 wording made consistent** (ambiguous after Stage A -> Stage B; still ambiguous -> Rung 2), and the "no false-negative problem" claim softened: per-episode labels are exact, but patch verdicts from limited seeds can still be wrong, so the early-stop error rate is measured (4, 7).
- **One expected verdict per injected bug**: controller latency (code bug) and shared-server stale actions (infra) are now separate entries (6.4).
- **Build order has time estimates** and a cut rule if the pilot slips (8).

### Changes in earlier revisions
- The causal claim rests on the router, not on which module is tuned (5.1).
- The data fix is a cost ladder (augmentation -> Isaac randomization -> Cosmos) instead of Isaac/Cosmos only (5.3).
- Clean-data source is a user-selectable menu with prerequisites, plus conversion to the policy's format (5.2, 3.1).
- Default training recipe uses official tune flags on the projector and action head; LoRA is optional and unofficial in newer versions (5.4).
- Router gained infra/latency and rendering-mismatch replays and a "new modality" verdict; fix selection is a rule table (3.2, now 3.2.10).
- Bug suite grounded in documented silent faults; shape mismatch recast as an easy Rung 0 case (6.4).
- Metrics gained nominal-regression and per-rung sufficiency (7). Build order split into MVP and stretch (8).

### Dropped
- **Separate LLM soft-verification rung ("will this succeed?")**: overlapped with the Rung 0 reviewer, could only reorder candidates, and measuring its false negatives needed simulator runs that cost what it was meant to save. Replaced by staged seeds inside the simulator rung, plus an optional logged reviewer pass probability.
- **"Cosmos (or a model) decides if vision is the problem"**: replaced by counterfactual replay.
- **Last-known-good commit swap as a router step**: kept only as an evaluation oracle.
- **2x2 replay grid (code swap x appearance swap) as the router**: replaced by stage-wise isolation and one-factor restores, because it needs a clean commit and cannot say which factor or which stage.
- **Aggregator-ordered replays with early stopping**: removed, since the model's ordering could decide the verdict.
- **VLM triage as a prior or a classifier**: a video model can no longer shape the route. It is used only for the optional row-9 factor proposal, and each proposal is tested by a replay.
- **"Fails on a known-good baseline scene means simulator issue"**: replaced by a reference-controller check, because most global code bugs also fail the baseline scene.
- **"Nothing flips means action gap or sim"**: replaced by explicit rows that need positive evidence, plus "perception unresolved" and "undetermined".
- **Hunk bisect as a routing step**: demoted to optional localization when a real known-good history exists.
- **"Retraining from scratch" as the cost baseline**: not a realistic alternative.
- **Three-arm module experiment and LoRA rank sweep**: too expensive for the time budget; default recipe plus one optional ablation instead.
- **"Action head frozen proves the policy was fine"**: removed as unsupported (5.1).
- **Dreams / RL / Cosmos-generated actions for action gaps**: too expensive; mentioned only as future work.
- **`|X_true - X_pred|` as the main router**: assumes an exposed coordinate output.
- **Fixed 60/40 data split and 1.0/1.5 loss weights**: unsupported numbers; replaced by tuning on validation.
- **Hard 90% / 20% pass-fail thresholds and SNR > 2**: replaced by thresholds calibrated on the nominal policy.
- **"Cosmos IDM extracts exact actions from generated video"** as the main data source: actions come from the simulator or the clean rollout.
- **Unverified claims** (late-2026 benchmark details, speedups, rank-2 failure explanations, repo capabilities): kept only as items to verify.
