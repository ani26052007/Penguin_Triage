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

## Current state (handoff, 2026-10-27)

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

## Phase 3: instrumentation, trace sources, router change 2 (2026-10-27, partial: see "Not done")

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
vla-isaac      152 passed              (115 + 37: test_traces.py, test_trace_parts.py, Rung 0 protection)
policy-wrapper  38 passed              (25 + 13: tests/test_instrument.py)
```
Staleness: a 3-step buffer → `estimate_extra_delay` 3.00 and `staleness_constant`; no buffer → 0, no evidence. Also tested: a client preprocessing change (BGR, flip) shows in `policy_input` stats.

### New VERIFY items
V-hdf5-joint-order, V-hand-limits (placeholder 0.0/1.5 rad, `width_hand: right`); Phase 3 usage added to V-hdf5-keys, V-hdf5-export, V-ee-frame, V-obj-state, V-progress, V-rgb-order, V-reset-calls.

### Deviations and findings
- **Changed router test**: `tests/test_router.py:110` listed the four phase signals literally; now `{V.failure_phase(p) for p in V.PHASES}` (intent unchanged, verdicts unchanged).
- **GR1 has no `ee_frame` transformer** at Arena 7632eb7d5e, so EE positions come from the policy's own obs terms logged by the instrument (V-ee-frame).
- **Not logged**: contact penetration (`SimHealth.max_penetration` stays 0.0, which the sanity check reads as healthy) and missing-texture fraction. Fixing needs a router change; not done.
- **Not visible**: a delay on the observation side (a stale observation reused) is not in the step pairs; only action-side delay is.
- Arena's copied `ActionChunkScheduler` clears `env_requires_new_chunk[mask]` in place, so its fetch-count stats read the zeroed mask (Arena quirk, left as is; tags set before it).

### Not done (usage limit)
- The reusable `TraceSource`/`BehaviorSource` contract suite (`tests/contracts/source_contract.py`, IMPL_PLAN D1); the refusals are tested directly in `test_traces.py`.
- policy-wrapper `NOTICE` lines for the scheduler / client changes in this phase; the P2/P3 legend in `VERIFY_ON_GPU.md`.
