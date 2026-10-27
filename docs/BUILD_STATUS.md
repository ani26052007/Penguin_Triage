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
  - Reference actors: WRAPPER raises `NotImplementedError` (G3, decision 3). RAW goes to a `reference` runner (Phase 4) and receives the scene's variations only.
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
- **G3 will crash the router (question below, blocking).** `router/branches/branch_g.py:37-38` always runs the G3 replay through the backend. With decision 3, `IsaacBackend` raises `NotImplementedError` for it, so every Branch G routing on the real backend ends UNRESOLVED (`error_stage="router"`) instead of leaving the G3-dependent result undecided. `decide` does not require G3: in Branch G, row 4 is "code by elimination" without it. Fixing it means a router edit beyond the approved ones, so this is a STOP condition.
