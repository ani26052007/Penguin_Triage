# Task: build the simulation side of Penguin_Triage (vla-isaac + policy-wrapper)

You are working in `~/comp_and_research/Penguin_Triage`. The router (`vla-router/`) and the repair agent (`hybrid-agent/`) already exist and run on mocks. Your job is to build everything that connects them to Isaac Lab Arena and GR00T N1.6:

- **`vla-isaac/`**: the rollout backend, traces, reference controller, experiment tools, data pipeline and GPU runbook.
- **`policy-wrapper/`**: the repair target.

## 0. Read first. Do not write code until this is done

Read these in full:
- `docs/plan.md`: the design.
- `docs/PLAN_UPDATE.md`: the proposed changes to the plan. Treat its replacement texts as accepted.
- `docs/PILOT_AND_ORDER.md`: the pilot choice, the step order and the five decisions.
- `docs/facts_gr00t_n16.md`: verified code facts.
- `DECISIONS.md`, `ISSUES.md`, `vla-router/DECISIONS.md`, `vla-router/README.md`.

In the docs, citation prefixes mean:

| Prefix | Path |
|---|---|
| `G:` | `external/IsaacLab-Arena/submodules/Isaac-GR00T` |
| `A:` | `external/IsaacLab-Arena` |
| `R:` | `vla-router` |
| `P:` | `docs/plan.md` |

Then read the vla-router interfaces you must implement or call. **Never guess their signatures.**
- `router/schemas.py`: `SceneSpec`, `TrainingSpec`, `ActorRef`, `EpisodeResult`, and related types.
- `ladder/backend.py`: `RolloutBackend`, `MockBackend`, and the placeholder at lines ~35-56.
- `router/prereq/trace.py`: `EpisodeTrace`, `TraceSource`, and the `code_key` rules.
- `router/prereq/behavior_trace.py`: `BehaviorTrace`, `GraspRecord`, `SubtaskProgress`, `PHASES`, `BehaviorSource`.
- `router/stage1/cfg_contract.py`: `ConfigMapping`.
- The workspace manager: `WorkspaceManager`, `materialize`, `file_config_loader`.
- `ladder/rung0.py`.
- `tests/workspace_cases.py`.
- The existing contract tests for `MockBackend`, `TraceSource` and `BehaviorSource`.

Then read the Arena GR00T client you will copy:
- `A:isaaclab_arena_gr00t/policy/*`, `utils/*`, `embodiments/gr1/*` and `tests/*`;
- `A:isaaclab_arena/policy/action_scheduling/*`;
- `A:isaaclab_arena/evaluation/policy_runner.py` and `policy_runner_cli.py`;
- `A:isaaclab_arena/recording/*`;
- `A:isaaclab_arena/variations/*`;
- `A:isaaclab_arena/terms/recorders.py`.

Write `docs/IMPL_PLAN.md`. It must contain:
1. The exact vla-router signatures you will implement, copied from the code.
2. The file list below, adjusted to what you found.
3. Every place where the docs and the code disagree.
4. Every Arena or Isaac Lab API you will depend on that you could not read. `submodules/IsaacLab` is not checked out, so everything from Isaac Lab is in this list.

**Then STOP and wait for my approval.**

## 1. Accepted decisions

1. **Compute.** All Isaac and GR00T runs happen on a rented GPU: an RTX 6000-class card with ≥48 GB, plus L4s for grid sweeps later. The laptop (RTX 3050, 6 GB) runs only GPU-free tests. Never assume a GPU, Isaac Sim or a GR00T install exists on this machine.
2. **Pilot.** GR1 `put_item_in_fridge_and_close_door`, object `ranch_dressing_hope_robolab`, checkpoint `nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge`. Failure family is dome-light lighting. **Keep task-specific values in config, not code**, because the pilot may switch to DROID `pick_and_place_maple_table` after the first measurements.
3. **Reference controller.** Isaac Lab Mimic runs RAW for G2 and C1. G3 is deferred: any request for the reference through the wrapper raises `NotImplementedError`, and those decision rows stay undecided.
4. **Rung B.** Mimic regeneration under the shifted factor. State-replay re-render is not built.
5. **Bug suite.** Client-side bugs in `policy-wrapper/`, plus denoising and normalisation bugs through `serve.py`. The injected async "infra" mode is built **last** and clearly labelled as injected.

**Design rule the docs miss.** Actuator and gain overrides from `policy-wrapper/config/eval.yaml` belong to the code under test. They are applied **only** when the actor is the policy. They are **never** applied when the RAW reference runs. Otherwise a gain bug makes G2 fail and gets routed as sim/asset instead of code. Write a test for this.

## 2. Hard rules

- **Do not modify these:**
  - `external/IsaacLab-Arena`, which is read-only;
  - `hybrid-agent/`;
  - vla-router, except for the two changes listed in Phase 1 and Phase 3. Explain each one in `vla-router/DECISIONS.md` in the existing style.
- **Do not download models, datasets or Docker images.** Do not install Isaac Sim, GR00T or flash-attn. Do not run anything that needs a GPU.
- **GPU-free imports.** Every module must import without Isaac Sim, torch-cuda, h5py-with-real-files or GR00T installed. Put heavy imports inside functions, or behind a small adapter that tests replace with a fake.
- **No invented APIs.**
  - If code depends on an API you could not read, mark the line `# VERIFY(<id>): <what to check>`.
  - Add the item to `docs/VERIFY_ON_GPU.md` with the exact command that settles it.
  - Never present a guessed API as fact.
- **Configuration lives in YAML.** No task names, joint counts, camera names or thresholds are hard-coded in Python.
- **Tests.** Every phase ends with `pytest` green, both in `vla-isaac` and in `vla-router`. Show me the test output. Follow vla-router's existing test style and fixtures. Reuse its backend, trace and behaviour contract test suites rather than duplicating them.
- **Commits.** One commit per phase in each repo you touch, with a clear message.
- **Licence.** Files copied from Arena keep their Apache-2.0 headers. Add a `NOTICE` file to `policy-wrapper/` listing the source paths and the Arena commit `7632eb7d5e`.

## 3. Phases

Do the phases in order. After each phase, append a section to `docs/BUILD_STATUS.md` with:
- what was built;
- the test command and its result;
- the new `VERIFY` items;
- anything you deviated from, and why.

Continue to the next phase unless a STOP condition (section 5) hits.

### Phase 1. `vla-isaac` package and task-agnostic backend
- Create `vla-isaac/` with a `pyproject.toml` that installs vla-router as an editable path dependency.
- **`vla_isaac/config.py`**: loads a backend config YAML holding the Arena root, GR00T root, Python executables, output root, server host and port, the default number of incident envs, and timeouts.
- **`vla_isaac/scene/factors.py`**: maps `SceneSpec` factors to Arena Hydra overrides through a YAML factor table, for example `lighting.intensity: {asset: light, variation: intensity, kind: build_time}`.
  - A fixed value is `enabled=true` with sampler `low == high`, or a one-element `hdr_names`.
  - The mapping must be **total and injective**. An unknown factor raises an error; it never falls back silently to nominal.
  - Restoring factor i removes exactly that factor's overrides.
- **`vla_isaac/scene/list_variations.py`**: parses the output of `--list_variations` into the factor table, so the table can be regenerated on the GPU box.
- **`vla_isaac/seeds.py`**: router seed = (build seed, episode index).
  - Group seeds by build seed.
  - For one build seed, the episodes are 0..k-1. Episode k's run-time variation draws do not depend on `num_envs` (`A:isaaclab_arena/variations/variation_base.py:124-131`).
  - Results come back in the caller's seed order.
- **`vla_isaac/runner/arena_cmd.py`**: builds the `policy_runner.py` command line. It sets:
  - policy type, policy YAML, remote host and port;
  - `--num_episodes`, `--num_envs` (clean mode = 1, incident mode = N), `--seed`, `--enable_cameras`, `--output_base_dir`;
  - the env sub-command, then the overrides.

  Unverified flags such as `--viz none` get `VERIFY` markers.
- **`vla_isaac/runner/subprocess_runner.py`**: runs the command with a timeout and captures the logs.
  - A non-zero exit raises a typed error, which vla-router turns into UNRESOLVED (the F3 behaviour; check how).
  - Include a `FakeArenaRunner` that writes fixture `episode_results_rank*.jsonl` with the real field names from `A:isaaclab_arena/recording/common_terms.py` and `progress_terms.py`.
- **`vla_isaac/results/episode_jsonl.py`**: JSONL → `EpisodeResult`. It keeps `variations`, `placement` and `progress` in `logs`, together with paths to the raw files.
- **`vla_isaac/backend/server.py`**: start, ping, stop and restart of the GR00T server. It speaks the ZMQ/msgpack `ping` endpoint (`G:gr00t/policy/server_client.py`). Its tests use a fake REP server.
- **`vla_isaac/backend/isaac_backend.py`**: `IsaacBackend` implements `RolloutBackend.run(scene, code_ref, actor, seeds, mode)`. In order, it:
  1. materialises `code_ref` through `WorkspaceManager`;
  2. starts that copy's `serve.py`;
  3. runs the policy runner with `PYTHONPATH` pointed at the copy;
  4. parses the results;
  5. stamps `code_key`.

  `code_key` is stamped by writing a sidecar `vla_meta.json` next to the outputs. Do not use an Arena recorder term.
- **vla-router change 1** (ISSUES #19): the rollout cache key gains the backend id. Make it backward compatible, and add a test.
- **Tests:**
  - run the shared `RolloutBackend` contract suite against `IsaacBackend` with the fake runner and fake server;
  - same inputs → same command;
  - factor mapping is total and injective, and unknown factors raise;
  - a crashed subprocess raises;
  - restore(i) removes exactly one override;
  - seed order is preserved across builds;
  - actuator overrides are absent for an `ActorRef` of kind REFERENCE.

### Phase 2. `policy-wrapper/` repo
- Create `policy-wrapper/` as a **new git repo** with the layout in `docs/PILOT_AND_ORDER.md` ("Where `policy-wrapper/` comes from").
- Copy the listed Arena files. Rewrite `isaaclab_arena_gr00t.*` imports to `policy_wrapper.*`. Keep the imports of `isaaclab_arena.policy.policy_base` and `register_policy`, which come from the installed Arena. Make those imports lazy, so the tests run without Arena installed. If that is impossible, say so in `BUILD_STATUS.md`.
- Make the client read `config/eval.yaml`, the `client:` section, as its single source of truth.
- **`config/eval.yaml`**: use exactly the pilot values in `docs/PILOT_AND_ORDER.md`. Put the HF revision hash as `TODO_PIN` plus a `VERIFY` item.
- **`serve.py`** starts `Gr00tPolicy` from the `server:` section. It:
  - sets `action_head.num_inference_timesteps` from `denoising_steps`;
  - seeds torch per request (`seed_per_request`). The seed comes from a request field or a counter that resets on `reset`. Document which you chose;
  - applies `norm_stats_override` when set;
  - knows that `--modality-config-path` is ignored when a model path is given (facts §8), and does not pretend otherwise.
- `instrument.py` is a stub in this phase. It is filled in Phase 3.
- **Tests:**
  - the copied fake-client tests, re-pointed;
  - a fake ZMQ REP server returning a fixed chunk: check the observation dict structure, chunk replay and per-request seeding;
  - `WorkspaceManager.materialize` plus `file_config_loader("config/eval.yaml")` round-trip;
  - `cfg_contract` with the real `ConfigMapping` from `docs/PILOT_AND_ORDER.md` on base vs edited `eval.yaml`, expecting exactly one violation per edited key;
  - Rung 0 on the repo.
- **Commit** the base as a single snapshot commit. No comments anywhere that point at future injected bugs.
- Register it as a `WorkspaceManager` base in the way vla-router expects.

### Phase 3. Instrumentation → `TraceSource` / `BehaviorSource`
- **`policy-wrapper/policy_wrapper/instrument.py`** writes, per episode, a JSON file with:
  - policy-input `FrameStats`, taken after `resize_rgb_for_policy` and sampled every N steps (N in config);
  - `StepPair`s: observation step at fetch, applied step = fetch step + chunk index, using the scheduler's `current_action_index`;
  - raw server actions.

  It costs nothing when disabled.
- **`vla_isaac/traces/hdf5_reader.py`** reads Arena's HDF5 recorder groups through an adapter interface, so tests use synthetic h5py files. It extracts:
  - `joint_cmd` and `joint_pos`;
  - EE pose;
  - object poses;
  - `SimHealth`: NaN fraction, object initial pose vs logged.

  Any group or key name you could not confirm gets a `VERIFY` marker.
- **GR1 hand-closure scalar**: a configurable function of the hand joints. The default is the mean normalised closure of the finger joints listed in YAML. It is used for `gripper_cmd_width` / `gripper_width` and for the grasp moment.
- **Reach and grasp events are computed offline** from the HDF5 data:
  - reach = EE within d of the object;
  - grasp = closure above threshold and the object moving with the EE.

  Do not add Arena `TrackedEventCfg`. Lift and place come from `progress.events`.
- **`IsaacTraceSource._load` / `IsaacBehaviorSource._load`** read only the files the backend wrote. They refuse a missing or wrong `code_key`.
- **vla-router change 2: the door subtask.**
  - Add a configurable phase mapping so that a composite task's extra subtask maps to an existing phase or to a new optional phase.
  - Recommend the smallest change, implement it, and keep every existing `PHASES` test passing.
  - Note it in `vla-router/DECISIONS.md`.
- **Tests:**
  - synthetic HDF5 and JSONL → traces;
  - `StepPair` arithmetic on a scripted chunk schedule, recovering a constant extra delay;
  - `FrameStats` on known images: RGB vs BGR channel means, and a flip is visible in the profiles;
  - wrong `code_key` is refused;
  - the hand-closure scalar on synthetic joint trajectories.

### Phase 4. Training spec and reference controller
- **`vla_isaac/specs/extract_training_spec.py`** builds a `TrainingSpec` JSON from:
  - the checkpoint's `config.json`, `processor_config.json`, `statistics.json` and `embodiment_id.json`;
  - Arena's closed-loop YAML and modality config.

  **The checkpoint's modality config wins over Arena's file** (facts §8). Every field is either filled or explicitly marked missing. Tests use small fixture JSONs that you write, shaped like the real files; mark them as fixtures.
- **`vla_isaac/reference/mimic.py`** runs Isaac Lab Mimic `generate_dataset.py` (`--generation_num_trials N`, the pilot task, `gr1_pink`) on a scene with the scene's variation overrides and **without** the code's actuator overrides. It:
  - parses the trial success rate;
  - produces `reference_unqualified` against a configurable threshold (ISSUES #5);
  - raises `NotImplementedError` for a WRAPPER-pipeline request.
- Wire RAW reference requests in `IsaacBackend` to it.
- **Tests:** parse fixture Mimic output; check the threshold rule; check that G3 raises; check that actuator overrides are absent.

### Phase 5. Experiment tools
- **`vla_isaac/experiments/grid.py`**: a failure-map grid runner over any `RolloutBackend`.
  - It takes a grid YAML of factor levels.
  - It runs one build per build-time cell.
  - It resumes from completed cells.
  - It can shard cells across several GPU hosts given in config, for the L4s.
  - It writes a CSV and JSON table with Wilson intervals, computed by vla-router's `ladder/stats.py`; do not reimplement them.
- **`vla_isaac/experiments/replay_agreement.py`** (ISSUES #15): same seeds twice in clean mode, then the per-seed agreement rate with an interval.
- **`vla_isaac/experiments/mode_gap.py`**: clean vs incident nominal rates.
- **`vla_isaac/replay/recorded_variations.py`**: strips one key from an Arena recorded-variation JSONL, so that Arena accepts replay with that factor disabled.
- **Tests:** everything runs on `MockBackend` with known rates and agreement. Check resume and sharding.

### Phase 6. Data pipeline and fine-tuning
- **`vla_isaac/data/rung_a.py`**: produces the Rung A fine-tune configuration, a colour-jitter and rotation argument set.
- **`vla_isaac/data/rung_b_mimic.py`**:
  1. runs Mimic regeneration under the failing factor (50-100 episodes);
  2. converts the result with Arena's `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py` and a copied YAML that keeps `action_name_sim: processed_actions`.
- **`vla_isaac/data/mixer.py`**: a deterministic mix of nominal and shifted LeRobot episodes at a configurable ratio, default 50/50.
- **`vla_isaac/data/heldout.py`**: a deterministic held-out split of conditions.
- **`vla_isaac/train/finetune_launcher.py`** builds the `G:gr00t/experiment/launch_finetune.py` command for the 5.4 default recipe:
  - `--no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model` plus the jitter arguments;
  - the Arena tune-visual recipe as an ablation preset.

  Before launching, it reads `tune_top_llm_layers` from the base checkpoint's `config.json` and records it in the run manifest. After the run, it parses the trainable-parameter count and `final_model_config.json` into the manifest.
- **Tests:**
  - conversion on fixture HDF5, reusing the approach of Arena's `tests/test_convert_hdf5_to_lerobot.py` if it runs without Isaac; otherwise test only the YAML and command construction;
  - mixer ratio and determinism;
  - split determinism;
  - exact launcher command strings for both presets.

### Phase 7. GPU runbook
Create `vla-isaac/scripts/gpu/`. These scripts are **written but not run** here.
- `00_check_gpu.sh`:
  - the GPU must be Ampere or newer, because flash-attn is required;
  - it must have RT cores, so an A100 or H100 is rejected with a message;
  - it prints VRAM.
- `R0_setup.sh` to `R8_vram.sh`: exactly the commands in `docs/PILOT_AND_ORDER.md`, made idempotent. Every script writes into `measurements/<timestamp>/`. **R4 runs 50 episodes, not 20**, because a 19/20 result straddles 0.8.
- `R7_dark_cell.sh` plus `R7b_hdr_and_colour_temp.sh`: the shifted-lighting probes that decide whether the pilot family exists.
- `mimic_override_check.sh`: 2 Mimic trials with a lighting override. It checks from the recorded variations that the override took effect. This decides decisions 3 and 4, so it runs on day one.
- `check_client_tests.sh`: runs Arena's GR00T client tests to learn whether they import without Isaac Sim.
- `collect_measurements.py`: turns everything into one `measurements/summary.json`, plus an applied/unapplied list against `docs/VERIFY_ON_GPU.md`.
- `run_order.md`: the order to run them on the rented box, the expected outputs, and the decision each one settles. In particular:
  - pilot stays or switches to DROID;
  - Mimic is usable or not;
  - mode gap is real or not.

### Phase 8. Injected bugs (last)
- **`vla-isaac/bugs/`**: this is the injection tooling. It lives **outside** `policy-wrapper/` and outside anything the agent container will mount.
  - It contains a ledger YAML with: bug id, target file, diff, expected symptom and expected verdict, following the 6.4 table in `docs/PLAN_UPDATE.md`.
  - It contains a script that applies one bug to a fresh copy and **squashes it into a single base commit**, with no tell-tale comments, flag names or timestamps.
- **Cover:**
  - RGB/BGR swap;
  - image flip;
  - hand sign;
  - one swapped joint-space YAML line;
  - chunk/replan mismatch;
  - denoising steps;
  - normalisation mismatch;
  - mild normalisation offset;
  - latency buffer;
  - wrong instruction for one object;
  - single-resolution preprocessing path;
  - gain decay via `env.actuator_overrides`.
- The injected async infra mode comes last and is clearly labelled.
- **Tests:**
  - each bug applies cleanly and changes exactly the intended file;
  - the squashed repo has one commit;
  - a grep for ledger ids and suspicious words finds nothing in the repo.

## 4. When everything is done

Update `docs/BUILD_STATUS.md` with:
- a table of every phase;
- the final test counts for `vla-isaac`, `policy-wrapper` and `vla-router`;
- the complete `docs/VERIFY_ON_GPU.md` list;
- the first three commands I should run on the rented GPU.

## 5. STOP conditions: stop and ask me

Stop and ask me if any of these happens:
- A vla-router interface does not match the docs in a way that changes the design.
- You would need to edit vla-router beyond the two changes allowed above.
- You would need to edit `external/IsaacLab-Arena` or `hybrid-agent/`.
- A copied Arena file cannot be made importable without Isaac Sim, and that blocks GPU-free tests for a whole phase.
- Any test in vla-router that passed before your change now fails and you cannot fix it without changing its intent.
