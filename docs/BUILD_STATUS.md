# Build status: simulation side

One section per phase of `docs/PROMPT_sim_side.md`, with decisions from the Phase 0 approval
(`docs/IMPL_PLAN.md` Q1-Q5 and the follow-ups). Test environment: conda env `vla-sim` (a clone
of `vla`, Python 3.12.15, CPU-only torch 2.14.1). Run with `unset PYTHONPATH`, because the shell's
ROS `PYTHONPATH` loads pytest plugins that break collection.

## Phase 1: `vla-isaac` package and task-agnostic backend (2026-10-27)

### Built
- `vla-isaac/pyproject.toml` depends on `vla-repair-ladder` (vla-router), installed editable in the same pip command as vla-router's own README does. The `[test]` extra pins the versions used here. `README.md` has the setup.
- `vla_isaac/config.py` holds two strict YAML loaders; an unknown or missing key raises `ConfigError`.
  - **Backend config** (`config/backend.example.yaml`): Arena and GR00T roots, the two Python executables, output root, workspace store and bases, server host/port/timeouts, run timeouts, incident env count, seed stride, runner kind, app flags, policy device, and `trajectories.camera_stride`.
  - **Task config** (`config/pilot_gr1_fridge.yaml`): every task-specific value. That covers the env block, the env factory class, the actuated env name, the eval-config consistency checks, the policy type, the serve module and the eval-config keys. A DROID pilot is a new YAML file.
- `vla_isaac/scene/factors.py` maps factors through a YAML table (`config/factors_gr1_fridge.yaml`) to `<asset>.<variation>` overrides.
  - Fixed values: `uniform` sets `sampler_cfg.low == high`, `choice` sets a one-element `hdr_names`, `bernoulli` sets `sampler_cfg.probability`.
  - Total and injective: an unknown factor raises `UnknownFactor`, and two factors on one variation raise `FactorTableError`.
  - Overrides are serialised exactly as Arena's `overrides_from_dict` does.
- `vla_isaac/scene/list_variations.py` parses `--list_variations` output, following the format in the code, not the docstring (IMPL_PLAN D6).
  - It infers uniform, choice and bernoulli samplers, and reports anything else as unsupported instead of guessing.
  - CLI: `python -m vla_isaac.scene.list_variations --input --output --names`.
- `vla_isaac/seeds.py` maps a router seed to `build_seed * stride + episode_index`, groups seeds by build and restores the caller's order. A build runs `max(k)+1` episodes.
- `vla_isaac/runner/arena_cmd.py` builds:
  - the one-run experiment YAML;
  - the experiment-runner argv (`--experiment_config`, `--experiment_output_directory`, `--enable_cameras` when the env asks for cameras, then the app flags);
  - the actuated launcher argv;
  - the `policy_runner.py` argv for the runbook (policy flags, env sub-command, overrides).

  They are pure functions.
- `vla_isaac/runner/subprocess_runner.py`:
  - `SubprocessRunner` runs with a timeout and keeps the log. A non-zero exit, a timeout or a failed start raises `ArenaRunError`.
  - `FakeArenaRunner` writes `episode_results_rebuild0.jsonl` with Arena's real field names: the core record, `variations`, `placement` and `progress.events`.
- `vla_isaac/results/episode_jsonl.py` reads strictly (duplicate keys, a missing core field or a non-bool `success` raise).
  - Episode index: clean mode uses `episode_in_env`; incident mode ranks by `(episode_in_env, env_id)` and sets `paired: False` (D10).
  - `logs` keep `variations`, `placement`, `progress`, the episode fields and the file paths.
- `vla_isaac/backend/server.py` starts, pings, stops (`kill` endpoint, then terminate, then kill) and restarts the policy server. It speaks GR00T's msgpack/ZMQ wire format with its own code (D15) and refuses to start if something already answers on its port.
- `vla_isaac/backend/isaac_backend.py`: `IsaacBackend.run(scene, code_ref, actor, seeds, mode, *, record_trajectories=False)`.
  - Order of work: materialise, start the copy's `serve.py`, run the experiment runner per build with `PYTHONPATH` set to the copy, parse the results, write the `vla_meta.json` sidecar with `code_key`, and stamp `code_key` into every result's logs.
  - Episodes beyond the requested seeds go into `cache` under this backend's `backend_id`.
  - `plan_run` is the pure planning step, which keeps commands deterministic.
  - Reference actors: WRAPPER raises `NotImplementedError` (G3, decision 3; since router change 4 `ReplayUnavailable`, below). RAW goes to a `reference` runner (Phase 4) and receives the scene's variations only.
- `vla_isaac/arena_ext/{actuator_env,launch_experiment}.py` run in Isaac's Python only (D7). The launcher registers a subclass of the task's env factory that applies `env.actuator_overrides`, then calls Arena's `experiment_runner.main()`.
  - `IsaacBackend` uses this path only for a POLICY actor with non-empty overrides.
  - Reference runs never get it, so a gain bug cannot make G2 / C1 fail.
- **vla-router change 1** (ISSUES #19): `CacheKey.backend` and `RolloutBackend.backend_id`, backward compatible. Entry added to `vla-router/DECISIONS.md`.
- `docs/VERIFY_ON_GPU.md` lists every VERIFY item; a test enforces that every marker has a row.
- `policy-wrapper/` is added to the top-level `.gitignore`.

### Tests
```
cd vla-isaac  && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 92 passed in 12.95s
cd vla-router && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 754 passed, 1 skipped in 27.60s
```
(vla-router before Phase 1: 752 passed, 1 skipped. The 2 new tests are for change 1.)

What the vla-isaac tests cover:
- the shared `RolloutBackend` contract suite (`tests/contracts/backend_contract.py`), run on `MockBackend` and on `IsaacBackend` over the fake runner and fake server;
- same inputs give the same command (plan, argv, env, YAML);
- the factor mapping is total and injective, unknown factors raise, and `restore(i)` removes exactly one factor's overrides (one test per factor);
- a crashed subprocess raises, both through the real `SubprocessRunner` (exit 139, timeout, missing executable) and through the fake; a crash inside `IncidentLoop` ends the incident UNRESOLVED with `error_stage="router"`, `error_type="ArenaRunError"`;
- seed order is preserved across three builds;
- actuator overrides are absent for REFERENCE actors (RAW and WRAPPER) and present for the policy;
- server start, stop, restart, timeout, early exit and busy port, against a fake ZMQ REP server process;
- extra episodes land in the cache and are served from it;
- the JSONL edge cases;
- config errors;
- every module imports with torch, numpy, h5py, zmq, msgpack, cv2, Isaac and GR00T blocked.

### New VERIFY items
V-viz, V-exp-yaml, V-exp-remote-policy, V-runner-parity, V-instruction, V-timestamped-dir, V-jsonl-success, V-var-record, V-list-var, V-actuator, V-ext-env, V-traj-size. The scripts for V-runner-parity (`R4b_runner_parity.sh`) and V-instruction (`R4c_instruction_ab.sh`) are written with the rest of the runbook in Phase 7.

### Deviations and findings
- **The task config gained `environment_class`** (`module:Class`). The actuated launcher needs the factory class, and hard-coding the pilot's class in Python would break the "task-specific values in config" rule.
- **The eval config must name the scene's environment.** `env.task`, `env.embodiment` and `env.object` in the code's `eval.yaml` are checked against the task config. A mismatch raises `ConfigError` instead of running a scene the `SceneSpec` does not name.
- **The camera stride is applied by the reader, not by Arena.** Arena records every step and has no stride option. The stride is in the backend config and in every `vla_meta.json`.
- **`record_trajectories` may not record cameras at all (V-traj-size).** Its terms (`A:isaaclab_arena/terms/recorders.py:277-289`) have no camera term; camera frames are only in the demo recorder config. If so, Phase 3's `FrameStats` come from the wrapper's `instrument.py` (policy input) and the render stats need another source; Phase 3 will say which.
- **Not yet copied: the runner-parity and instruction GPU scripts.** They are Phase 7 runbook items; their VERIFY rows exist now.
- **G3 will crash the router (question below, blocking).** `router/branches/branch_g.py:37-38` always runs the G3 replay through the backend. With decision 3, `IsaacBackend` raises `NotImplementedError` for it, so every Branch G routing on the real backend ends UNRESOLVED (`error_stage="router"`) instead of leaving the G3-dependent result undecided. `decide` does not require G3: in Branch G, row 4 is "code by elimination" without it. Fixing it means a router edit beyond the approved ones, so this is a STOP condition. **Resolved:** the user chose option (a) as router change 4 (next section).

## Router change 4: G3 unavailable (2026-10-27, between Phases 1 and 2)

Approved as option (a) with six conditions; each is met as follows.

### Built
1. `ladder.backend.ReplayUnavailable(NotImplementedError)` and `wrapper_reference(actor)`. `IsaacBackend.run` raises it for REFERENCE + WRAPPER as its first statement: no materialise, no server, no subprocess, no output directory, nothing cached (tested with spies on `materialize`, `subprocess.run` / `Popen`, the fake runner and the server factory).
2. Call-site audit, listed in `vla-router/DECISIONS.md`: G3 (`router/branches/branch_g.py:37`) is the only router request for the reference through the WRAPPER, and it is the only place that catches `ReplayUnavailable`. G2 and C1 are RAW; every other replay uses the policy. Outside `route()`, `scripts/confusion_matrix.py` also builds G3 on a mock that never raises it. No other WRAPPER use, so no STOP.
3. G3 gets no `ReplayResult`; it is recorded as unavailable (`ReplayRunner.unavailable`, `RouterResult.unavailable`, verdict notes) and `decide(..., unavailable={"G3"})` gives it its own state. It is read as neither a pass nor a fail.
4. Row 4 by elimination with G3 unavailable sets `Verdict.g3_unavailable=True` (with `by_elimination=True`, `confidence=low (elimination only, G3 unavailable)`). It is shown by `verdict_text` (report / hagent prompt) and carried in `PassRecord` / `LoopResult.to_dict()`.
5. `MockBackend(..., wrapper_reference_unavailable=True)`, and the test `QuotaBackend` honours it. Tests in `vla-router/tests/test_g3_unavailable.py` (16) cover:
   - a located contract hit gives the same verdict;
   - no contract hit gives code by elimination with the flag;
   - a plain `NotImplementedError` (G3 or G2) and `ReplayUnavailable` for G2 end UNRESOLVED;
   - Branch C gives identical results;
   - `decide` semantics and validation.
6. `vla-router/DECISIONS.md` "Router change 4" entry, referencing decision 3.

Side fix: `CachedBackend` counts hits and misses after the backend returns, so a refused run adds no misses.

### Tests
```
cd vla-router && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 770 passed, 1 skipped in 29.52s
cd vla-isaac  && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 92 passed in 13.02s
```
(vla-router: 754 + 16 new. vla-isaac: the G3 test was replaced by the stricter no-side-effect test.)

### Deviations and findings
- `code_partial_offset` with G3 unavailable loses the location `observation_side_or_policy_loading` (it needs G3 restored); the verdict is otherwise unchanged.
- Until Phase 4 wires the Mimic RAW reference, G2 on `IsaacBackend` still raises a plain `NotImplementedError`, so a Branch G incident on the real backend still ends UNRESOLVED; that is intended (only G3 is optional).

## Phase 2: `policy-wrapper/` repo (2026-10-27)

### Built
- **`policy-wrapper/`** is a new git repo with one snapshot commit (`23fa9de`, branch `main`). It is in the top-level `.gitignore` and has its own `NOTICE` (the source paths, Arena commit 7632eb7d5e, and the changes made to the copied files) and `LICENSE` (Apache-2.0, the text of Arena's `LICENSE.md`). The copied files keep their headers.
- **Copied from Arena:**
  - `policy/` (core, remote closed-loop client, config, task mode);
  - `scheduling/` (Arena core's three action schedulers);
  - `utils/` (image conversion, io, joints conversion, robot joints);
  - `embodiments/gr1/*`;
  - the fake-client test.

  `isaaclab_arena_gr00t.*` and `isaaclab_arena.policy.action_scheduling` are rewritten to `policy_wrapper.*`.
- **`policy_wrapper/_arena.py`** imports `PolicyBase`, `PolicyCfg` and `register_policy` from Arena when it can, and otherwise uses stand-ins with the same surface. A base class cannot be imported lazily, so this is the fallback the prompt allows; it does not block the tests (D5). The `isaaclab_arena_g1` constants are imported only in the G1 branch.
- **`config/eval.yaml`** holds exactly the pilot values. `server.checkpoint_revision: TODO_PIN` is V-hf-rev. None of the pilot doc's comments are copied.
- **The client's single source of truth** is `config/eval.yaml` `client:`, read by `policy_wrapper/eval_config.py`:
  - unknown keys raise;
  - relative paths are resolved against the repo root;
  - the scheduler is selected here (the `scheduler` field moved off `Gr00tRemoteClosedloopPolicyCfg`);
  - `bgr_conversion` is honoured by `resize_rgb_for_policy`.
- **Per-request seed (D16):**
  - The client sends `options={"seed": request_seed(cfg.seed, resets, request index)}` (`policy_wrapper/seeding.py`).
  - `serve.py` seeds torch with that seed. If a request has none, it uses its own counters, which restart on `reset`.
  - The seed used is returned in the reply's info dict.
- **`serve.py`** builds `Gr00tPolicy` from `server:`:
  - `model.action_head.num_inference_timesteps = denoising_steps`;
  - `processor.set_statistics(override_file, override=True)` when `norm_stats_override` is set;
  - the checkpoint comes from `--checkpoint-dir`, a local path, or the local HF cache with `local_files_only` (V-ckpt-cache). It is never downloaded at serve time.

  Its docstring says that GR00T ignores `--modality-config-path` when a model path is given, and that `serve.py` has no such option.
- **`instrument.py`** is a no-op stub.
- **vla-isaac:**
  - `vla_isaac/workspace.py` holds `eval_config_loader` / `make_workspaces`, which register policy-wrapper as a `WorkspaceManager` base with the Q3 loader. The loader reads `config/eval.yaml` and adds `client.target_image_hw = target_image_size[:2]` and `server.effective_norm_stats`. That value comes from the override file when it is set, otherwise from the checkpoint's `statistics.json`. If neither can be found, the loader raises `WorkspaceError`.
  - `vla_isaac/workspace.py` also holds `config_mapping(task)`.
  - `vla_isaac/specs/norm_stats.py` turns GR00T statistics into `TrainingSpec.norm_stats` (flat per modality, group order from YAML). Phase 4's `extract_training_spec` uses the same function.
  - Task config: `eval_config:` (key paths, derived keys, norm-stats group order) and `config_mapping:` (the pilot `ConfigMapping`, with `image_size → client.target_image_hw` and `norm_stats → server.effective_norm_stats`).
  - Backend config: `checkpoints:` maps checkpoint id to local dir.
  - `IsaacBackend` passes `--device` and, when the checkpoint is mapped, `--checkpoint-dir` to `serve.py`.

### Tests
```
cd policy-wrapper && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 25 passed in 3.20s
cd vla-isaac      && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 114 passed in 19.66s
cd vla-router     && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 770 passed, 1 skipped in 31.06s
```
- policy-wrapper:
  - Arena's fake-client tests (9), re-pointed to GR1: 54-DoF state, 26-DoF policy, 36-DoF action, 512² POV camera, 16-step chunks.
  - `test_wire.py`, a real ZMQ REP server returning a fixed chunk, checks three things:
    - the observation dict as decoded on the server;
    - chunk replay: 16 steps from one chunk, then a new request;
    - per-request seeds: sent, used by `torch.manual_seed`, reproducible across servers, restarted by `reset`.
  - `test_serve.py` covers the strict server section, checkpoint resolution, the denoising and stats hooks, and the `serve.py` process end to end (ping, `get_action`, `reset`, `kill`).
  - `test_eval_config.py` checks that the pilot values are exact, that the client config comes from `client:`, and that `bgr_conversion` works.
  - `test_imports.py` checks that every module imports without Arena.
  - The `gr00t` modules are stand-ins written for the tests (`tests/fake_gr00t`). No GR00T code is copied.
- vla-isaac `tests/test_policy_wrapper_base.py`, on the real checkout:
  - the `materialize` + `file_config_loader("config/eval.yaml")` round trip, and `code_ref_of` gives the same name;
  - the derived keys;
  - `make_workspaces`;
  - a missing checkpoint raises;
  - **cfg_contract with the pilot mapping:**
    - nominal gives zero violations;
    - each of the 7 edited mapped keys gives exactly one (render gives `render_settings_differ`);
    - a `norm_stats_override` with modified stats gives exactly one `norm_stats` violation;
    - an override with identical stats gives zero;
  - **Rung 0** on a working copy (hagent `LocalEnv`, `pytest` of the copy): passes on a harmless edit and fails on a syntax error.

### New VERIFY items
V-ckpt-cache, V-stats-format, V-server-seed. V-exp-remote-policy now names policy-wrapper's class.

### Deviations and findings
- **The policy is registered as `policy_wrapper_gr00t_remote_closedloop`.** Under Arena's name `gr00t_remote_closedloop`, it would collide with Arena's own GR00T policy if both were imported: `register_policy` would then skip ours, and the typed-config lookup would fail.
- **`scheduler` moved from `Gr00tRemoteClosedloopPolicyCfg` to `eval.yaml` `client:`**, which makes eval.yaml the single source. The experiment YAML never sets it.
- **`cv2` is still imported at module level** in `utils/image_conversion.py` (IMPL_PLAN said to move it). `opencv-python-headless` is in the test environment, so copying the file unchanged was the smaller change.
- **The training spec in the cfg_contract tests is built by hand from the pilot values**, with norm stats from a FIXTURE `statistics.json`. Phase 4 extracts the real one.
- **A laptop without the checkpoint cannot name policy-wrapper code with the Q3 loader.** `WorkspaceError` names the backend `checkpoints:` key. This is intended: the norm-stats contract must not pass for lack of the statistics.
- **Arena's own client tests may still not run without Isaac Sim.** policy-wrapper's tests show they can be made to; Phase 7's `check_client_tests.sh` checks Arena's.

## Router change 3: Rung 1 `kind: isaac` from an import path (2026-10-27, after Phase 2)

### Built
- `ladder/rung1.py`: `"backend": {"kind": "isaac", "factory": "pkg.module:attr[.attr]", "args": {...}}`. The factory is called with `args` and must return a `RolloutBackend`. Every error is a clean exit-2 `ValueError`.
- vla-isaac: `IsaacBackend.from_config(config, **kw)` builds the backend from a backend YAML: task config, policy-wrapper bases, and the Q3 eval-config loader. The Rung 1 factory string is `vla_isaac.backend.isaac_backend:IsaacBackend.from_config`.
- `vla-router/DECISIONS.md` has a "Router change 3" entry (before change 4).

### Tests
```
cd vla-router && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 780 passed, 1 skipped in 29.42s
cd vla-isaac  && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 115 passed in 19.38s
```
- vla-router `tests/test_rung1_backend_factory.py` (10):
  - a dummy import path through the whole CLI;
  - a class-attribute path;
  - seven bad configs;
  - a scan showing that vla-router never imports `vla_isaac`.
- vla-isaac: its factory path builds an `IsaacBackend`.

## State before Phase 3 (handoff, 2026-10-27; superseded by "Current state" at the end)

Done: Phase 1, router change 4, Phase 2, router change 3 (and router change 1 in Phase 1). **Next: Phase 3** (instrumentation, `TraceSource` / `BehaviorSource`, router change 2). The decisions to follow are in `docs/IMPL_PLAN.md` §8, "Accepted decisions".

| Repo | Test command (run inside the repo) | Result |
|---|---|---|
| vla-router | `env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q` | 780 passed, 1 skipped in 29.56s |
| vla-isaac | same | 115 passed in 19.08s |
| policy-wrapper | same | 25 passed in 3.03s |

Commits (top-level repo `Penguin_Triage`, branch `fixes`; vla-router and vla-isaac are directories in it):

| Commit | What |
|---|---|
| `08a2006` | Phase 1: vla-isaac package, `IsaacBackend`, router change 1 |
| `e0c88d6` | Router change 4: G3 unavailable |
| `24052ae` | Phase 2: vla-isaac side (eval-config loader, norm stats, `make_workspaces`, policy-wrapper base tests) |
| `1ae2385` | Router change 3: Rung 1 `kind: isaac` from an import path |
| policy-wrapper `23fa9de` | Separate repo `policy-wrapper/` (branch `main`, gitignored at top level): the single base snapshot |

Repo hygiene:
- The top level still has the user's staged `.gitmodules` / `external/IsaacLab-Arena`, the deleted root `PLAN_*.md` / `plan.md` / `prompts.md` (moved to `docs/`), the modified `hybrid-agent`, and untracked docs.
- Never stage any of them. Commit with `git commit --only -- <own paths>`.

## Phase 3: instrumentation, trace sources, router change 2 (2026-10-27, finished 2026-10-28)

### Built
- **policy-wrapper `instrument.py`**: five hooks (`steps`, `on_request`, `on_reply`, `on_step`, `on_reset`/`close`), configured only by `VLA_INSTRUMENT_*` env vars set by the harness (never `eval.yaml`); `NullInstrument` when unset (no counters, stats or files).
  - Boundaries (correction 2): `FrameStats` of the dict actually sent to the server (`policy_input`) and of the env camera frame (`render`); raw chunks exactly as returned.
  - Tags (correction 1): the schedulers return `TaggedAction(action, obs_step, chunk_offset)`; the tag is set when the chunk arrives and travels with the action; `(obs_step, applied_step, offset)` is logged in `on_step`, at the return to `env.step`. `get_action` stays as a wrapper.
  - One JSON per episode, with `start_index` (incident-mode start order, D10) and observation keys (`policy.*_eef_pos`).
- **vla-isaac `traces/`**: `layout.py` (task YAML `traces:`), `hdf5_reader.py` (adapter `TrajectoryStore`, `H5pyStore`), `instrument_reader.py`, `hand_closure.py`, `events.py` (offline reach/grasp, events → subtasks, `phase_map`), `sources.py` (`IsaacTraceSource`, `IsaacBehaviorSource`: read only the files `vla_meta.json` names, inside its directory; code_key from meta).
- `IsaacBackend`: traced runs set the instrument env vars; meta gains `instrument_dir`, `seed_stride`, `profile_buckets`; `logs.paths.run_root` is the log ref. `trajectories.profile_buckets` added; `camera_stride` is the instrument's frame interval.
- **Protected evidence (correction 3)**: task config `rung0.forbidden_prefixes` includes `policy_wrapper/instrument.py` (`TaggedAction` lives there); `vla_isaac.workspace.rung0_config`. Phase 8 must never target it.
- **Router change 2** (correction 4): optional `post_place`; `DECISIONS.md` entry. `config/gr1/*` joint-space copies + `vla-isaac/NOTICE`.

### Tests
```
vla-router     786 passed, 1 skipped   (780 + 6: tests/test_post_place_phase.py; golden pick-only verdicts unchanged)
vla-isaac      176 passed              (115 + 61: test_traces.py, test_trace_parts.py, test_source_contract.py, Rung 0 protection)
policy-wrapper  38 passed              (25 + 13: tests/test_instrument.py)
```
- The source contract suite (`tests/contracts/source_contract.py`: template `load` not overridden, traces in the mode asked for with the code key, deterministic and round-tripping, other code refused, a missing reference unavailable, a reference to the other mode's logs a `ValueError`) runs on the router's `InMemoryTraceSource` / `InMemoryBehaviorSource` and on `IsaacTraceSource` / `IsaacBehaviorSource` (IMPL_PLAN D1).
- Mutation checks (made by hand, then reverted): logging `applied = obs_step + chunk index` instead of the step at the return to `env.step` fails the delay-buffer tests in policy-wrapper and vla-isaac; stamping the incident's code key instead of the run's fails four refusal tests.

Staleness: a 3-step buffer → `estimate_extra_delay` 3.00 and `staleness_constant`; no buffer → 0, no evidence. Also tested: a client preprocessing change (BGR, flip) shows in `policy_input` stats.

### New VERIFY items
V-hdf5-joint-order, V-hand-limits (placeholder 0.0/1.5 rad, `width_hand: right`); Phase 3 usage added to V-hdf5-keys, V-hdf5-export, V-ee-frame, V-obj-state, V-progress, V-rgb-order, V-reset-calls.

### Deviations and findings
- **Changed router test**: `tests/test_router.py:110` listed the four phase signals literally; now `{V.failure_phase(p) for p in V.PHASES}` (intent unchanged, verdicts unchanged).
- **GR1 has no `ee_frame` transformer** at Arena 7632eb7d5e, so EE positions come from the policy's own obs terms logged by the instrument (V-ee-frame).
- **Not logged**: contact penetration (`SimHealth.max_penetration` stays 0.0, which the sanity check reads as healthy) and missing-texture fraction. Fixing needs a router change; not done. *(Penetration: fixed by router change 5, below.)*
- **Not visible**: a delay on the observation side (a stale observation reused) is not in the step pairs; only action-side delay is.
- Arena's copied `ActionChunkScheduler` clears `env_requires_new_chunk[mask]` in place, so its fetch-count stats read the zeroed mask (Arena quirk, left as is; tags set before it).

### Also in this phase
- policy-wrapper `NOTICE` lists the changes to the copied schedulers and client.
- `VERIFY_ON_GPU.md`: the P0-P3 legend is below the table, which is now one table (the legend used to split it).

## Decisions from the Phase 3 review, and the fixes before Phase 4 (2026-10-28)

Recorded in `docs/IMPL_PLAN.md` §8, "Decisions from the Phase 3 review" (commit `5151b38`): the changed `test_router.py:110` is accepted; hand open/closed values stay V-hand-limits; router change 5 is approved; stale observations reused by the client are out of scope (no 6.4 bug), with a frame hash logged per request.

**Known limitation (stale observations).** The staleness checks see only action-side delay (the step pairs). An observation reused by the client is not evidence for any router check. Since fix c, every request's images carry a content hash in the instrument files, so it is detectable later (two requests with the same hash sent the same image).

### Fix a: the grasping hand is chosen per episode (`cc54314`)
- `traces.hand_closure.width_hand` is gone from the task config and `TraceLayout`.
- `vla_isaac/traces/events.py` `choose_hand`: the grasping hand is the hand whose closure first crosses the grasp threshold while it is within `reach_distance_m` of the object (crossings at one step: nearer hand first), and the grasp needs the object to move with that hand, as before. Without a grasp, the hand of the first closure within reach is the grasping hand (`hand_basis: attempt`). Without any closure within reach there is no grasping hand, and the nearest hand gives the gripper width only (`nearest`). The chosen hand's closure is the episode's `gripper_cmd_width` / `gripper_width`.
- Recorded in the behaviour trace: the router's `BehaviorTrace` has no logs or notes field, so `IsaacBehaviorSource` returns `IsaacBehaviorTrace`, a subclass with `notes = {grasp_hand, width_hand, hand_basis}` that round-trips through `to_dict` / `from_dict`. The router reads it as a plain `BehaviorTrace`. No router change.
- Tests: left-hand grasp, right-hand grasp, no grasp, an attempt that does not move the object, a closure out of reach, a tie (nearer hand wins), and end to end through `IsaacBehaviorSource` / `IsaacTraceSource` (notes per episode, gripper width from the chosen hand).
- **Deviation:** when an earlier closure within reach does not move the object and a later one by the other hand does, the hand that moved it is the grasping hand. The literal rule ("first crossing within reach") would name the first hand while the grasp is the second hand's.

### Fix b: router change 5, `SimHealth` fields not measured (`50dc31d`)
- vla-router: `SimHealth.max_penetration: Optional[float]`, `None` = not measured. It is the only `SimHealth` field vla-isaac cannot measure. `nan_fraction` and the initial position come from the HDF5, and an unlogged rest position is already `()`.
  - `to_dict` writes `null`. `from_dict` reads `null` as `None`, numbers as before, and a missing key as 0.0 (the old format). The dataclass default stays 0.0.
  - `SanityCheck` skips `None`. A fired evidence's detail ends with `; sanity/max_penetration not measured (k of n episodes)`, and the check also returns a note. A measured value scores exactly as before.
  - `DECISIONS.md` "Router change 5".
- vla-isaac: the traces carry `max_penetration=None`.
- Tests: `vla-router/tests/test_sim_health_not_measured.py` (9), plus vla-isaac `SanityCheck` on Isaac traces (no evidence, the note).
- Rejected: a `PartialSkip` (it would make row 2 UNKNOWN on every Isaac incident), and changing the default to `None`.

### Fix c: frame content hash (policy-wrapper `1b66477`)
- `instrument.py` `content_hash` = blake2b-128 over dtype, shape and bytes. At every request (not only at the sampled frame steps), each image of the dict sent to the server is hashed:
  - the chunk record carries `sent_hashes` (per video key, oldest frame first);
  - each frame sample carries `policy_input_hash` next to its `FrameStats`.
- Nothing is hashed when instrumentation is off. The format string stays `/1`, because vla-isaac's reader ignores the added keys. No router change.
- Tests: hash properties; every sent image hashed at every request; a stale frame shows as a repeated hash; the disabled path makes no hash calls.

## Phase 4: training spec and reference controller (2026-10-28)

### Built
- **`vla_isaac/specs/norm_stats.py` `training_norm_stats`**: the one norm-stats function. Both `extract_training_spec` (`TrainingSpec.norm_stats`) and the Q3 eval-config loader (`server.effective_norm_stats`) call it (point 1).
  - The group order comes from the checkpoint's `processor_config.json` (its modality keys). It falls back to the task config's `norm_stats_groups` only when the checkpoint has no processor config.
  - The statistics are the checkpoint's `statistics.json`, or the override file.
  - The loader now always resolves the checkpoint, also when an override is set, so the override is flattened in the checkpoint's group order.
- **`vla_isaac/specs/arena_data_config.py`** reads Arena's GR00T data config module through its syntax tree, without importing GR00T.
  - The result has the shape a checkpoint's `processor_config.json` holds: `ModalityConfig` / `ActionConfig` become dicts with the dataclass defaults, and enums become their names.
  - Anything it cannot read raises an error.
- **`vla_isaac/specs/extract_training_spec.py`** builds a `TrainingSpec` and writes it as JSON. CLI: `python -m vla_isaac.specs.extract_training_spec --backend backend.yaml --output spec.json`.
  - Inputs: the checkpoint's four files; Arena's closed-loop YAML, data config, `modality.json` and converter YAML (task config `training_spec:`, paths relative to the Arena root); and the local LeRobot dataset's `meta/` (backend config `datasets:`, never downloaded).
  - **The checkpoint wins** over Arena's data config (point 2). Every differing field is listed in `extra["disagreements"]` as `{field, used, values: {checkpoint, arena_data_config, ...}}`. The fields compared are: every field of every modality config, the embodiment tag, the action horizon, the norm-stats group order, and the state and action layouts. The frame rate and image size are compared between the dataset and Arena's converter (the dataset wins).
  - **Every field is filled or explicitly missing.** `extra["sources"]` gives each value's file and key, and `extra["missing"]` gives each reason. A test checks that the two sets partition the spec's fields.
    - Pilot: `action_horizon` from the checkpoint's action `delta_indices`, `replan_steps` from Arena's `action_chunk_length`, `denoising_steps` from `config.json` `num_inference_timesteps`;
    - `control_hz` / `image_size` from the dataset's `meta/info.json`, else Arena's converter;
    - `norm_stats` and the layouts from the checkpoint;
    - `channel_order: RGB` and `render_settings: {}` declared in the task config (V-rgb-order, V-render-defaults);
    - `camera_poses` missing (no file records them).
  - **`instruction_templates` only from `meta/tasks.jsonl`** (point 3). Without a local dataset it is missing with a V-train-instr reason. It is never taken from the eval string, nor from either Arena YAML. The two Arena strings are kept in `extra["arena_instructions"]` for the reader.
  - Also in `extra`: the processor settings (crop, resize, jitter, ...), the model's tune flags and `tune_top_llm_layers`, and the embodiment id.
- **`vla_isaac/reference/mimic.py` `MimicReference`** (point 4) is the default reference runner of `IsaacBackend` when the task config has a `reference:` section.
  - **RAW only.** WRAPPER (G3) raises `ReplayUnavailable` (still `NotImplementedError`). A policy actor is a `ValueError`. Another reference checkpoint is a `MimicConfigError`.
  - **Never the actuated launcher.** The command is `generate_dataset.py` with Arena's stock registration callback and `--task <env>` plus the task's env args (`--embodiment gr1_pink`). The scene's variation overrides come last. There is no `--external_environment_class_path`, no `vla_isaac`, no actuator variables and no `PYTHONPATH`. The config loader refuses a section that names the launcher.
  - **No downloads.** The annotated source dataset is a backend `datasets` path. Absent, the run is refused with `MimicConfigError` (V-mimic-source).
  - **Trials.** One `EpisodeResult` per requested seed, in trial order: trial i belongs to `seeds[i]`, with `logs["paired"] = False`.
    - Outcomes come from the cumulative "successes/attempts" progress lines in Mimic's log (pattern in the task config, PLACEHOLDER, V-mimic-out).
    - The first `len(seeds)` attempts are used. `generation_guarantee` may add attempts (V-mimic-guarantee); they are counted in `attempts_total`.
    - A log that cannot order the attempts, or has too few, raises `MimicOutputError`.
    - The pairing rule is stated in `vla-router/DECISIONS.md` ("Backend note"). No router code change.
  - **Qualification** (ISSUES #5): `qualify(results, rule, condition)` uses `ladder.stats.wilson_interval`.
    - The reference is unqualified when the **lower bound** is below `reference.qualification.threshold` (0.8 at 95 % in the pilot config), not when the raw rate is.
    - `Qualification.evidence()` is `reference_unqualified` (source `reference_qualification`, strength 1.0), for `route(..., extra_evidence=...)`.
    - A test shows that `decide` puts it in row 2.
- Task config: `training_spec:` and `reference:` sections, both strict. Backend config: `datasets:` (id → local path).

### Tests
```
cd vla-router     && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 795 passed, 1 skipped in 30.66s
cd vla-isaac      && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 213 passed in 28.66s
cd policy-wrapper && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 41 passed in 5.24s
```
(Before the fixes: 786 + 1 skipped / 176 / 38. Fix a: vla-isaac +7. Fix b: vla-router +9, vla-isaac +1. Fix c: policy-wrapper +3. Phase 4: vla-isaac +29, in `tests/test_training_spec.py` (14) and `tests/test_mimic.py` (15).)

- **Training spec** (FIXTURE checkpoint and dataset files; Arena's real files from the checkout):
  - pilot values with sources; the fields partition into filled and missing;
  - a JSON and CLI round trip;
  - a changed checkpoint modality config wins, and both values are listed;
  - the checkpoint's group order sets the layout and both norm-stats sides;
  - without a dataset, or without `tasks.jsonl`, the instruction is missing and never the eval string;
  - **the nominal pilot gives zero `cfg_contract` violations end to end**: the spec is extracted from the fixture checkpoint files, `eval.yaml` goes through the loader, and every mapped field is compared;
  - the loader's `effective_norm_stats` equals `spec.norm_stats`;
  - the converter's "ranch dressing" string as the training string gives exactly one instruction violation, and an edited `denoising_steps` gives exactly one;
  - bad inputs and strict sections;
  - the AST reader on Arena's file and on an unreadable one.
- **Mimic:**
  - parsing the fixture log (order, totals, repeated lines, mixed or non-cumulative counts);
  - one result per seed, in trial order and unpaired;
  - extra attempts;
  - too few attempts, and a crash;
  - the exact command, with the scene's variations last;
  - **no actuated launcher and none of the code's gains** (a backend whose eval config sets gains; the policy plan does get them);
  - a missing source dataset is refused before running;
  - only RAW runs (WRAPPER raises `ReplayUnavailable` / `NotImplementedError`);
  - the backend serves the reference through `CachedBackend`;
  - strict sections;
  - the Wilson rule: 20/20 qualifies at 0.8; 19/20 does not, although its raw rate 0.95 ≥ 0.8; 48/50 does; the threshold and confidence are configurable;
  - the unqualified evidence routes to row 2.

### New VERIFY items
V-train-instr (new row; was in IMPL_PLAN §4 only), V-mimic-source, V-mimic-out, V-mimic-guarantee, V-mimic-seed. Phase 4 usage was added to V-mimic-override, V-ckpt-cfg, V-rgb-order and V-render-defaults.

### Deviations and findings
- **No checkpoint file and no Arena file records the frame rate or image size.** Arena's `embodiments/gr1/info.json` is a template with `fps: null`. They come from the dataset's `meta/info.json` when it is local, else from Arena's converter YAML (fps 50, 512²), which produced the training data.
- **`channel_order`, `render_settings` and `camera_poses` are in no file.** The first two are declared in the task config with VERIFY ids, and `camera_poses` is missing. `training_spec.declared` accepts only these three.
- **`replan_steps` comes from Arena's closed-loop YAML.** The checkpoint has no replan setting; it is an evaluation choice.
- **A missing spec field is silently skipped by the router's `cfg_contract`** (`expected is None`: `continue`, no partial skip). So without a local dataset the instruction is not compared and nothing in the router result says so; only `extra["missing"]` records it. Changing that is a router change; not done.
- **The loader now needs the checkpoint even with a stats override**, because the group order comes from the checkpoint. A laptop without the checkpoint could already not name policy-wrapper code (Phase 2), so nothing new is blocked.
- **Mimic's semantics are unread** (Isaac Lab is not checked out):
  - the progress-line pattern is a placeholder;
  - with `generation_guarantee`, `--generation_num_trials` probably counts successes;
  - the pilot env fixes the datagen seed (`seed: 1`). If that seed cannot be changed per run, a G2 extension may repeat the same trials (V-mimic-seed; `vla-router/DECISIONS.md` known limit).

  All three are settled by the day-one `mimic_override_check.sh` run (Phase 7).
- **19/20 is unqualified at threshold 0.8 and 95 %.** With 20 trials only 20/20 qualifies. The threshold and the trial count are config, and the plan's N=20 may need raising to 50 (as for R4).
- **`IsaacBehaviorTrace` subclasses a router schema** (fix a), instead of adding a field to the router's `BehaviorTrace`.
- Known limitation carried over: `FrameStats.missing_texture_fraction` is still 0.0 from vla-isaac (not measured). It is a `FrameStats` field, outside router change 5.

## Decisions from the Phase 4 review, and the changes before Phase 5 (2026-10-28)

Recorded in `docs/IMPL_PLAN.md` ("Decisions from the Phase 4 review"). Baseline before any change: 795 passed, 1 skipped / 213 / 41.

### Router change 6: `cfg_contract` says which fields it did not compare (`b319ae2`)
- `ConfigContractCheck.notes` gives one line per **mapped** field whose `TrainingSpec` value is `None`: `contract_cfg: not compared: <field> (missing in spec)`, in `FIELD_ORDER`. The lines go through `Stage1Result.notes` into `RouterResult.notes`, the same route as change 5.
- It is a note, never a skip. The evidence is unchanged: same ids, strengths, locations and details. Unlike change 5, the line is not appended to fired evidence, because a `cfg_contract` evidence is about one other field.
- **Verdicts unchanged.** All 11 router cases go through `route()`, compared against the same mapping with the spec-less fields unmapped (the old silent skip). The verdicts are equal, and the router notes differ only by the not-compared lines. 9 of the 11 cases lack both mapped fields, so the test is not vacuous.
- `vla-router/tests/test_cfg_contract_not_compared.py` (+19), and a `vla-router/DECISIONS.md` entry. Its own commit.

### Mimic: 50 trials and the datagen seed (`7955552`)
- `reference.qualification.trials` (required, 50 in the pilot) and `qualification_seeds(rule)` = `0 .. trials-1`. With 20 trials only 20/20 clears 0.8 at 95 % (19/20 has a lower bound of 0.764). 48/50 clears it, with a lower bound of 0.865.
- New strict section `reference.mimic.seed` (`MimicSeed`: `mode`, `flag`, `max_trials`), VERIFY(V-mimic-seed), now a **day-one check**.
  - **`fixed`** (pilot default, the safe assumption until the check runs): the reference is deterministic. Router seed s **is** trial s of the one sequence. A run makes `max(seeds)+1` trials and returns those indices, so a G2 extension to seeds 10..19 gets trials 10..19, never a repeat of 0..9. Seeds outside `0 .. max_trials-1` are refused. Results carry `logs["deterministic"] = True` and `run_seed = None`.
  - **`per_run`**: each run passes its own seed (`seed.flag`, PLACEHOLDER `--seed`), derived from (scene, actor, seeds). A new request is a new draw, and the same request gives the same seed, so caching stays consistent. Trial i ↔ `seeds[i]` as before. `logs["deterministic"] = False`.
- **Qualification.** `qualify()` reads the flag from the results.
  - **Deterministic: a single pass/fail.** `Qualification.deterministic = True`, `interval()` is None, and `to_dict()` has `lower` / `upper` = None. The evidence says "deterministic reference (one fixed trial sequence, single pass/fail)". Repeated trials (the same `trial_index` twice) are refused: a repeated run is not more data. Mixing deterministic and per-run results is refused.
  - **Per-run**: runs are pooled and the Wilson interval is reported, as before.
- Tests: `tests/test_mimic.py` +9 (both modes, the extension, the bounds, the seed flag before the scene overrides, deterministic pass/fail without an interval, refusing repeats and mixing, pooled per-run interval, strict sections).
- `vla-router/DECISIONS.md` "Backend note": the known limit about G2 extensions repeating trials is marked handled (doc only, no router code).

## Phase 5: experiment tools (2026-10-28)

### Built
- **`vla_isaac/experiments/grid.py`**: the failure-map grid over any `RolloutBackend` (plan 6.1).
  - **Grid YAML** (`config/grid_gr1_fridge.yaml`, strict): factor → levels (`null` = factor absent), episodes, mode, policy checkpoint, workspace base label, `reference`, confidence, candidate rule, `hosts`. The pilot levels are PLACEHOLDERS until R7 / R7b.
  - Cells are every combination of levels, with the nominal cell always first.
  - **One build per cell.** One `backend.run` call per cell for the policy, seeds `0 .. episodes-1` (build seed 0 on `IsaacBackend`).
  - **Reference qualification per cell** (point 1). The RAW reference only (`ReferenceSpec` refuses WRAPPER and policy actors), on `qualification_seeds(rule)` in clean mode. It is qualified with Phase 4's `qualify()` and the task config's rule. The table holds the reference's k/n/rate/interval next to the policy's, and the cell is flagged `reference_unqualified`. A deterministic reference shows `reference_deterministic: true` and no interval.
  - **Candidate flag**: the nominal policy rate ≥ `nominal_min`, the cell rate ≤ `shifted_max`, and the reference not unqualified. These are point rates, a screening aid.
  - **Resume**: one JSON file per finished cell (`<out>/cells/<id>.json`), written atomically (temporary file plus rename). Each carries a fingerprint: grid, scene, code key, policy, seeds, mode, confidence, backend id and reference rule. A rerun skips matching cells and **refuses** a cell file with another fingerprint.
  - **Sharding**: cell i runs on `hosts[i % len(hosts)]`. `collect(spec, dirs)` merges the cell files of several hosts. It refuses missing cells, foreign cells and two different results for one cell.
  - **Table**: `GridTable` writes `grid_table.json` and `grid_table.csv`. Intervals come from `ladder.stats.wilson_interval` (`experiments/tables.proportion`).
  - **CLI**: `python -m vla_isaac.experiments.grid run --grid G --backend B --out D [--host H]` and `... merge --grid G --out D --from D1 D2`.
- **`vla_isaac/experiments/replay_agreement.py`** (ISSUES #15, point 2): the same seeds twice in clean mode, per cell.
  - It reports the agreement k/n with its Wilson interval (the rate cross-checked with `ladder.stats.agreement_rate`), the both-success, both-failure and both discordant counts, the overall success rate over both runs and each run's rate, plus a pooled agreement over all cells.
  - **Visibility**: a cell where every agreement is a failure (or every one a success) has `informative: false` and a note, for example "agreement comes only from failures (overall success rate 0.00): it says nothing about whether successes reproduce". The note appears in the JSON and CSV next to the number.
  - It refuses a `CachedBackend` (it would read the first run back as the second) and unpaired results (incident mode, Mimic trials). On `IsaacBackend` the second call is a real Arena run.
  - CLI on a grid YAML (`--nominal-only` to agree on the nominal cell only).
- **`vla_isaac/experiments/mode_gap.py`** (point 3): the nominal scene in clean and incident mode.
  - Both rates come with Wilson intervals, the gap is clean minus incident, and `differ` = two-sided `ladder.stats.fisher_exact` p < alpha.
  - **Always unpaired**: `pairing: "unpaired"` with the D10 reason. What the backend said (`incident_paired_in_logs`) is reported but does not change the method.
  - CLI on a grid YAML.
- **`vla_isaac/replay/recorded_variations.py`** (point 4): `strip_key(src, dst, key, enabled_keys)` and `strip_factor(src, dst, table, scene, factor)`.
  - It writes a copy without `variations[key]`; everything else is unchanged.
  - **It refuses**, before writing anything:
    - any remaining key without an enabled variation on the replay build, **naming each key and the first record holding it** (Arena asserts there);
    - a key that no record holds;
    - a key that is still enabled;
    - duplicate JSON keys (read as Arena reads them);
    - writing over the source.
  - `strip_factor` derives the enabled keys from `scene.restore(factor)` and the factor table.
  - CLI.

### Tests
```
cd vla-router     && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 814 passed, 1 skipped
cd vla-isaac      && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 264 passed
cd policy-wrapper && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 41 passed
```
(vla-router +19 from change 6. vla-isaac +9 from the Mimic changes, then Phase 5 +42: `tests/test_grid.py` 21, `tests/test_agreement_and_mode_gap.py` 14, `tests/test_recorded_variations.py` 7. The GPU-free import test covers the new modules.)

Everything runs on `MockBackend` with known rates (point 5):
- **Grid.** The rates are policy nominal 1.0, dark 0.0, dark+HDR 0.0 and 0.9 elsewhere; the reference is 1.0 everywhere except 0.5 on dark+HDR.
  - Exact counts. The 0.9 cells are checked against an independent run of the same mock.
  - The intervals equal `ladder.stats`.
  - Dark is a candidate, and dark+HDR is flagged `reference_unqualified` (and is not a candidate). The flag shows in the CSV.
  - One policy and one reference call per cell, with 20 seeds and 50 trials.
  - A deterministic reference gives pass/fail with no interval in the table.
  - JSON and CSV output.
- **Resume after a crash mid-grid.**
  - The backend crashes on the 4th cell. 3 cell files survive, and a stray half-written temporary file is ignored.
  - The rerun runs only the 3 missing cells, and the table equals the uninterrupted run.
  - A third run runs nothing.
  - Changing the episodes, confidence, code or reference rule is refused, never reused.
- **Sharding.** Two fake hosts with separate backends run disjoint cells (0, 2, 4 and 1, 3, 5). The merged JSON and CSV equal the one-host table. A partial merge, an unknown host, conflicting results and a foreign grid are refused. The merge CLI works.
- **`IsaacBackend` (fake runner).**
  - Each grid cell is one Arena build: seed 0, `num_envs` 1.
  - The agreement tool makes two real Arena runs.
- **Agreement.**
  - Exact counts on a scripted pair (4 / 2 / 2 / 2).
  - A deterministic mock agrees 1.0.
  - `replay_noise = 0.25` on 400 seeds gives an interval containing 0.75, the known agreement.
  - Always failing and always succeeding are both noted and not informative.
  - A cache and unpaired results are refused.
- **Mode gap.**
  - Clean 1.0 vs incident 0.2: `differ`, p equal to `fisher_exact` on the same counts, intervals from `wilson_interval`.
  - No gap: p = 1.
  - Unpaired whatever the logs say.
- **Recorded variations.**
  - The strip keeps everything else.
  - The refusals name the key (`'ranch_dressing_hope_robolab.mass' (first in record 0)`).
  - **Arena's own validator** (`recorded_variation_samples.py`, pure Python, run in an isolated subprocess) rejects the original recording and accepts the stripped one.

### New VERIFY items
None new. V-mimic-seed is rewritten as a day-one check with both outcomes handled (P5). V-var-record now also covers the strip tool.

### Deviations and findings
- **Deterministic qualification still applies the Wilson rule once.** "A single pass/fail" is implemented as the Phase 4 rule (lower bound ≥ threshold) applied to the one fixed trial sequence, with no interval reported. So 19/20 still fails and 50/50 passes. The alternative, the raw rate ≥ threshold, would make 20 trials enough again; I did not choose it. *(Confirmed in the Phase 5 review: the Wilson rule stays.)*
- **Fixed mode reads router seed s as trial index s.** It is not stated in the decision. It is what keeps extensions from repeating trials. The cost: a request for seeds 10..19 reruns trials 0..9 too (20 trials). `max_trials` (200) bounds it.
- **The grid always adds the nominal cell**, even if no factor lists `null`, because the candidate rule and the router need it.
- **The candidate rule uses point rates.** It is a screening aid, not a decision; the intervals are next to it.
- **The second agreement run deletes the first run's raw files on `IsaacBackend`.** It reuses the same output directory (`build0_n<k>`, which is wiped before each run). The outcomes are kept in the report, but the first run's Arena log and JSONL are gone. A follow-up could add a run tag to the output path. *(Fixed before Phase 6, below.)*
- **The mode gap calls `fisher_exact` directly** rather than `compare()`. `compare()` chooses paired or unpaired from the agreement, and D10 makes incident mode unpaired whatever the agreement is.
- **The strip tool also refuses an absent key and a still-enabled key**, beyond the point-4 rule. Replaying a stripped file through `IsaacBackend` (`recorded_variation_samples_path`) is not wired; Phase 5 asked only for the tool.
- **The `run` CLIs of grid, agreement and mode gap** build `IsaacBackend.from_config` and are not exercised here (GPU box). The merge CLI and every function they call are tested.

## Decisions from the Phase 5 review, and the fix before Phase 6 (2026-10-28)

Recorded in `docs/IMPL_PLAN.md` ("Decisions from the Phase 5 review"): a deterministic reference keeps the Wilson lower-bound rule; fixed mode reads router seed s as trial s, with the `max_trials` cap of 200; replaying a stripped recording through `IsaacBackend` stays unwired (known limitation; live restore covers it). Baseline before any change: 814 passed, 1 skipped / 264 / 41.

### Fix: replay agreement keeps both runs' raw files
- `IsaacBackend.run(..., run_tag=None)`. A tag joins the output key and names the directory: `runs/<hash>_<tag>/`. Without a tag the path is unchanged. A tag that is not a plain name (letters, digits, `-`, `_`, `.`) raises `ConfigError`.
- `measure_agreement` passes `run_tag="agreement_1"` / `"agreement_2"` to a backend whose `run` takes `run_tag` (`IsaacBackend`), so each run keeps its own Arena log, experiment YAML, episode JSONL and `vla_meta.json`. Other backends (`MockBackend`) are called as before.
- Each cell reports both runs' directories (`run_roots.first` / `.second`, from `logs["paths"]["run_root"]`). `compare_runs` refuses two runs that report the same directory.
- Tests (+3, vla-isaac 267): both runs' log, YAML, meta and JSONL exist after the measurement, in different directories; an untagged run keeps its old path and a bad tag is refused; a shared directory is refused. Mutation check: calling `run` without the tag fails 2 tests.

## Phase 6: data pipeline and fine-tuning (2026-10-28)

Requirements from `docs/IMPL_PLAN.md` ("Decisions from the Phase 5 review", Phase 6 points 1-6).

### Built
- **`config/finetune_gr1_fridge.yaml`** (`vla_isaac/data/config.py`, strict): `rung_a`, `rung_b` (episodes 100 in 50-100; the shifted conditions, PLACEHOLDERS until R7 / R7b; the converter), `heldout` (seed, fraction), `mix` (ratio 0.5, seed, nominal dataset id), `finetune` (launcher, modality config, embodiment tag, base checkpoint id and revision `TODO_PIN`, `allow_unknown_top_llm_layers: false`, the two presets, hardware per GPU count, `timeout_s`).
- **`config/arena/gr1_manip_ranch_bottle_config.yaml`**: Arena's converter YAML, copied unchanged (Apache header kept; `vla-isaac/NOTICE`). A test checks it is byte-identical to Arena's.
- **`vla_isaac/data/rung_a.py`** (point 5): `--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08` (tyro's dict form, Arena's values) and `--random-rotation-angle N`. A `None` value passes no flag, and the manifest says, per field, that GR00T then keeps the checkpoint's own setting (facts §5, `processing_gr00t_n1d6.py:513-522`), with the checkpoint's value when its `processor_config.json` is readable.
- **`vla_isaac/data/heldout.py`** (point 2): the split is over **conditions** (`condition_key`: sorted `factor=value` pairs, or `nominal`). The shifted keys are sorted, shuffled with `random.Random(heldout.seed)`, and `round(fraction * n)` are held out (at least one each side). The nominal condition is never held out or trained through the split: its episodes are the separate **regression** set. `split_episodes` puts every episode where its condition goes and refuses an episode id under two conditions.
- **`vla_isaac/data/mixer.py`** (point 3):
  - `mix_counts` / `select_mix`: ratio = shifted / total (config, default 0.5). All shifted episodes are used when the nominal pool is large enough, otherwise all nominal ones and the shifted are sampled down. Exact rational arithmetic, half up. Deterministic for `mix.seed` and independent of input order. An empty source, or an episode in both, is refused.
  - `write_mixed_dataset`: one GR00T LeRobot v2 directory, because `launch_finetune.py` takes a single `--dataset-path`. GR00T finds an episode's files through `episodes.jsonl` `episode_index` and the `info.json` patterns (`G:gr00t/data/dataset/lerobot_episode_loader.py`), so parquet and video files are copied under new indices without rewriting the parquet. Tasks map, modality, features and patterns must match across sources, or the mix is refused (the parquet's task indices are not rewritten). `meta/stats.json` is **not** copied: GR00T's dataset factory computes it when it is missing and keeps a stale one. `meta/vla_mix.json` maps every new index to its source episode, condition and role.
- **`vla_isaac/data/rung_b_mimic.py`** (point 4):
  - `regenerate` runs the existing `MimicReference` (RAW, the scene's variation overrides, no actuator overrides) for trials 0..N-1 on the shifted scene. In **fixed** seed mode the report says `deterministic: true`, `trajectory_sequence: "single fixed sequence"`, and warns that the N episodes are one fixed trajectory sequence whose diversity comes only from the scene variations. In `per_run` mode it reports the run seed and no warning. The nominal condition, a count outside 50-100, and zero successes are refused. The demo count is Mimic's `successes_total` (V-mimic-guarantee).
  - `plan_conversion` writes a per-run copy of the converter YAML: `data_root` and `hdf5_name` point at the generated HDF5, and the Arena-relative paths are made absolute. A template whose `action_name_sim` is not `processed_actions` is refused. The command is `<arena python> <arena>/isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py --yaml_file <yaml>` from the Arena root. An existing output directory is refused, because the converter would ask on stdin. `convert` returns a `LeRobotSource` tagged with the condition.
- **`vla_isaac/train/finetune_launcher.py`** (points 1 and 6):
  - Presets `default` (`--no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model`) and `tune_visual` (Arena's recipe, the ablation), both followed by Rung A's flags. Argument order as in Arena's docs. 1 GPU: `CUDA_VISIBLE_DEVICES=0 <python> gr00t/experiment/launch_finetune.py ...`. N GPUs: `<python> -m torch.distributed.run --nproc_per_node=N --standalone ...`. Batch, steps and workers come from `finetune.hardware[N]` (1: 16 / 30000 / 4; 8: 96 / 20000 / 16, from the sequential-task doc). Runs from the GR00T root.
  - **Refuses** to build a command when `tune_top_llm_layers` cannot be read from the base checkpoint's `config.json` (file missing, not JSON, key absent, not an integer >= 0), unless `allow_unknown_top_llm_layers` is set in config or passed. The manifest records the value, its source, the reason, and the allow flag either way. An unmapped base checkpoint is always refused (never downloaded).
  - **Manifest** (`vla_finetune_manifest.json`, written before the run): both presets' flags and args, the chosen preset, `tune_top_llm_layers`, checkpoint id / revision / `revision_pinned` / local dir, hardware, Rung A, the data mix (from `meta/vla_mix.json`, or `mixed: false`), the held-out split, the seeds (mix, held-out, and GR00T's training seed 42, which is not on the fine-tune CLI), cwd, argv, env and the command string.
  - **After the run**, `record_run` adds the trainable / total parameter counts and percentage from the log (ranks reporting different counts are an error), `experiment_cfg/final_model_config.json`, its `tune_top_llm_layers` and whether it matches the value read before, and `complete` / `missing`. CLI: `python -m vla_isaac.train.finetune_launcher plan|run --backend B --pipeline P --dataset D --out O [--preset] [--num-gpus] [--heldout split.json] [--allow-unknown-top-llm-layers]`.

### Tests
```
cd vla-router     && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 814 passed, 1 skipped
cd vla-isaac      && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 331 passed, 1 skipped
cd policy-wrapper && env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q   -> 41 passed
```
vla-isaac: 267 after the agreement fix, then Phase 6 +64 (`tests/test_data_pipeline.py` 38, `tests/test_rung_b.py` 11 + 1 skipped, `tests/test_finetune_launcher.py` 15). The skip: `V-converter-noisaac: Arena's converter does not import in this environment (ModuleNotFoundError: No module named 'tqdm'); it needs no Isaac module, but its third-party imports (tqdm, imageio, pandas) are not installed here.`
- **Config**: the pilot loads; ratio defaults to 0.5; 12 bad configs refused.
- **Rung A**: exact flags; a `None` field passes no flag, and the manifest note says the checkpoint's setting is kept (with the FIXTURE checkpoint's value, or `unread`).
- **Held-out**: deterministic and independent of input order; the seed changes the split; at least one condition on each side for fractions 0.1-0.9. No episode id or condition is on both sides, and nominal episodes are only in the regression set (on a shuffled pool of 25 episodes). Duplicates, fewer than two shifted conditions, and an episode under two conditions are refused.
- **Mixer**: exact counts for 7 cases (e.g. 1000/100 at 0.5 → 100/100; at 0.25 → 300/100; 30/100 at 0.5 → 30/30; 1000/3 at 0.4 → 5/3). Deterministic for a seed and independent of input order; another seed differs; empty sources refused. The merged directory: every new file's bytes are the source episode's, the meta is renumbered, the totals are right, there is no `stats.json`, and an existing output is refused. Different tasks or modality are refused, and nothing is written.
- **Rung B**: fixed mode is deterministic, with the warning, and a regeneration repeats the same command. The Mimic command has the lighting override, no actuator variables and `--generation_num_trials 50`. Per-run mode has the seed flag and no warning. The default count comes from config; the range, nominal and zero successes are refused. The template is byte-identical to Arena's. The written YAML changes exactly `data_root`, `hdf5_name` and the five paths, which resolve to real files in the Arena checkout. Other actions are refused. The exact converter command, the output dir rule and the stdin-prompt refusal; `convert` with a fake.
- **Launcher**: the exact command strings for both presets on 1 and 8 GPUs. The `tune_visual` flags and the jitter string are found in Arena's documented recipe. The manifest has every required field and is JSON-serialisable. 4 unreadable-config cases are refused, and accepted with the flag (from config or the argument), recorded in the manifest. Unmapped checkpoint, unknown preset and unconfigured GPU count are refused. `record_run` on the FIXTURE run (`tests/fixtures/finetune_run/`) parses the counts and the final config; missing outputs give `complete: false`; ranks reporting different counts raise. `run_finetune` writes the manifest before and after, with the env, cwd and timeout.

### VERIFY items
No new ids. Extended (P6): V-converter-noisaac (the import chain was read; the missing modules here), V-finetune-log (both lines and the final config file), V-ckpt-cfg (the launcher's refusal), V-hf-rev (`finetune.base_revision`), V-mimic-seed and V-mimic-guarantee (Rung B).

### Deviations and findings
- **The converter does not run here.** It needs no Isaac module, but tqdm, imageio, pandas and pyarrow are not in vla-sim. I did not install them (the env is pinned by the `[test]` extra), so the run test skips with that reason.
- **The mixer writes one merged LeRobot directory.** It does not use GR00T's multi-dataset `mix_ratio`, because `launch_finetune.py` hard-codes one dataset with `mix_ratio: 1.0`; that would need `launch_train.py`. The merge relies on the loader code read above. Whether training accepts the merged directory is checked by the first fine-tune.
- **Sources must share one instruction string.** The parquet's task indices are not rewritten, so the mix is refused when `tasks.jsonl` differs. The copied converter YAML says "ranch dressing bottle". If the original dataset's `tasks.jsonl` (V-train-instr) says something else, the Rung B YAML's `language_instruction` must be set to it before converting.
- **The held-out fraction is of the shifted conditions in `rung_b.conditions`.** Held-out conditions are regenerated too only if they are listed. Their episodes are set aside by `split_episodes`, never mixed in. The plan's evaluation on them is Rung 1, which is not part of this phase.
- **8-GPU hardware values** come from the sequential-task doc (batch 96). facts §3 quotes batch 24 for the static-manipulation doc.
- **Training seed**: GR00T's `data.seed = 42` is not on the fine-tune CLI, so every fine-tune uses it. The manifest says so. Run-to-run variance (README: 5-6 points) therefore needs `launch_train.py`, or repeated runs with that caveat.
- **Fine-tune timeout** is its own config value (`finetune.timeout_s`, 48 h). The backend's `timeouts.run_s` (2 h) is too short for 30k steps.
- **The `plan` / `run` CLI of the launcher** reads a backend YAML and is not exercised here; every function it calls is tested.

## Current state (2026-10-28)

Done: Phases 1-6, router changes 1-6, the three fixes before Phase 4, the Mimic changes before Phase 5, and the replay-agreement fix before Phase 6. **Next: Phase 7** (GPU runbook).

| Repo | Test command (run inside the repo) | Result |
|---|---|---|
| vla-router | `env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q` | 814 passed, 1 skipped |
| vla-isaac | same | 331 passed, 1 skipped (V-converter-noisaac) |
| policy-wrapper | same | 41 passed |

Commits since Phase 5:

| Commit | What |
|---|---|
| `1c77108` | IMPL_PLAN: decisions from the Phase 5 review |
| `642ee28` | Replay agreement keeps both runs' raw Arena files |
| (this commit) | Phase 6: data pipeline and fine-tuning |

Known limitations carried forward: a stripped recorded-variation file is not replayed through `IsaacBackend` (live restore covers it); the staleness checks see only action-side delay; `FrameStats.missing_texture_fraction` is not measured.

Repo hygiene is unchanged: never stage `.gitmodules`, `external/IsaacLab-Arena`, `hybrid-agent` or the moved `PLAN_*.md`; commit with `git commit --only -- <own paths>`.
