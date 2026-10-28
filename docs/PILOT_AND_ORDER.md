# Pilot choice and order of work to the pilot gate

Proposal only. Citation prefixes as in `docs/facts_gr00t_n16.md` (`G:` GR00T pin, `A:` Arena, `R:` vla-router, `P:` `docs/plan.md`). Companion files: `docs/facts_gr00t_n16.md` (Part 1) and `docs/PLAN_UPDATE.md` (Part 2).

---

# Part 3. Pilot choice

## Facts: the three candidates that have a GR00T N1.6 path in Arena

| | **A. GR1 `put_item_in_fridge_and_close_door`, ranch bottle** | **B. GR1 `gr1_open_microwave`** | **C. DROID `pick_and_place_maple_table`** |
|---|---|---|---|
| Checkpoint | `nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task`, subfolder `ranch_bottle_into_fridge/` (`A:docs/pages/example_workflows/sequential_static_manipulation/step_5_evaluation.rst:30-37`) | `nvidia/GN1x-Tuned-Arena-GR1-Manipulation`, `--revision gn1_6` (`A:.../static_manipulation/step_5_evaluation.rst:30-33`) | `nvidia/GR00T-N1.6-DROID`, zero-shot (`A:docs/pages/quickstart/running_a_real_policy/gr00t.rst:4-7,33-36`) |
| Documented success | 1.0 over 5 episodes, 1 env; **0.98 over 50 episodes, 10 parallel envs** (`step_5_evaluation.rst:135,206`; the page shows a 1-GPU and a 2-GPU command before that output, `:150-187`, and does not say which produced it); 0.9 over 10 in a multi-job run (`:363-368`) | 0.88 over 17 episodes, 1 env; **0.605 over 200 episodes, 10 envs** (`static_manipulation/step_5_evaluation.rst:127,194`) | Example output 0.0 to 0.67 over **3** episodes per cell (`gr00t.rst:159-177`): "zero-shot deployment ... remains challenging" (`:182`) |
| Training data in Arena | Annotated and Mimic-generated HDF5 on HF (`sequential_static_manipulation/step_3_data_generation.rst`, steps 1-2) | HDF5 and LeRobot on HF (`static_manipulation/step_4_policy_training.rst:36-66`) | None referenced |
| Lighting factors | Scene has a `light` dome light (`A:isaaclab_arena_environments/gr1_put_and_close_door_environment.py:141,191`), which carries HDR, intensity, colour and colour temperature variations (`A:isaaclab_arena/assets/object_library.py:468-471`) | No light asset in the scene (`gr1_open_microwave_environment.py:41-43,70`) | `light` and `directional_light` (`pick_and_place_maple_table_environment.py:78-83`) |
| Camera factors | None (GR1 registers no camera variations; only DROID and Franka do, `A:isaaclab_arena/embodiments/droid/droid.py:161`, `franka/franka.py:104`) | None | Extrinsics and intrinsics |
| Progress signals | Composite: pick-and-place (settled → lift → place) then close door (`gr1_put_and_close_door_environment.py:195-210`; `A:isaaclab_arena/tasks/pick_and_place_task.py:167-191`) | Door moved → door open (`A:isaaclab_arena/tasks/open_door_task.py:64-80`) | Settled → lift → place |
| Gripper | 6-DoF hands, absolute joint actions (`A:isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py:34-80`; `gr00t_26dof_joint_space.yaml`) | same | Binary gripper, recorded by `GripperStateRecorder` (`droid.py:200`; `A:isaaclab_arena/terms/recorders.py:210-275`) |
| Reference controller available | Mimic (existing) | Mimic (existing) | None (scripted state machine to build) |
| Episode length | 10 s at 50 Hz = 500 steps (`gr1_put_and_close_door_environment.py:210,220`; `A:isaaclab_arena/environments/isaaclab_arena_manager_based_env_cfg.py:212-222`) | 5 s at 50 Hz = 250 steps (`gr1_open_microwave_environment.py:77,80`) | 70 s at the 15 Hz default = 1050 steps (`pick_and_place_maple_table_environment.py:29`; `isaaclab_arena_manager_based_env_cfg.py:185-190`) |

## Decision (proposed): Candidate A

**Pilot: GR1 `put_item_in_fridge_and_close_door`, object `ranch_dressing_hope_robolab`, checkpoint `nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge`. Failure family: dome-light lighting (intensity first, then HDR and colour temperature).**

Reasons, each tied to a fact above:
1. It is the only candidate with a documented nominal rate ≥80% over ≥50 episodes (0.98 / 50).
2. Its training data and the Mimic pipeline that made it are downloadable. That gives Rung A data (the original set), Rung B data (Mimic regenerated under the shifted light), the training spec (Arena's modality config and converter YAML) and a RAW reference controller (Mimic).
3. Lighting variations attach to its scene with no code change.
4. It has pick and place progress events, so "failure phase" exists for C_behavior.

## Opinion: the costs of A, stated plainly
- It is a **composite** task. Plan 6.1 says to avoid long-horizon composites in the first version (`P:439`).
- Its close-door subtask has no slot in `BehaviorTrace.PHASES` (`R:router/prereq/behavior_trace.py:36`).
- **No camera family** is available.
- **Hands instead of a gripper**: `GraspRecord` and the gripper bugs need a hand-closure definition.
- The doc does not say whether the 0.98 / 50 result came from the 1-GPU or the 2-GPU command (`step_5_evaluation.rst:150-206`), so treat single-GPU 10-env behaviour as unmeasured.

**What would change the choice:**
- If A measures < 80% nominal in clean mode, fall back to C on whichever (object, destination, HDR) cell measures ≥80%. C has every variation family and a binary gripper, but no data and no reference controller.
- If the camera family is a must-have, choose C from the start.
- B is not recommended: no light asset, and its documented 1-env vs 10-env gap is unexplained.

## NEEDS A RUN: cheapest measurement of nominal success (Candidate A)

All on a remote GPU with ≥24 GB. Paths are relative to `external/IsaacLab-Arena`. Commands are from the Arena docs, with `--num_episodes` (`A:isaaclab_arena/evaluation/policy_runner_cli.py:84`) and the variation-override syntax (`A:docs/pages/concepts/variations/variations.rst:110-131`).

```bash
# R0. Setup (once)
git submodule update --init submodules/IsaacLab submodules/Isaac-GR00T
uv sync --extra dev --group gr00t-client && source .venv/bin/activate      # gr00t.rst:72-74
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
(cd submodules/Isaac-GR00T && uv sync --python 3.10 && uv pip install -e .)  # G:README.md:107-108

# R1. Checkpoint (sequential_static_manipulation/step_5_evaluation.rst:30-37)
export MODELS_DIR=$HOME/models/pilot && mkdir -p $MODELS_DIR/ckpt
hf download nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task --include "ranch_bottle_into_fridge/*" \
  --repo-type model --local-dir $MODELS_DIR/_dl && mv $MODELS_DIR/_dl/ranch_bottle_into_fridge/* $MODELS_DIR/ckpt/
grep -n "tune_top_llm_layers\|num_inference_timesteps\|action_horizon" $MODELS_DIR/ckpt/config.json   # training spec + 5.4 check

# R2. Server (terminal 1)
(cd submodules/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \
   --model-path $MODELS_DIR/ckpt --embodiment-tag GR1 --device cuda --host 127.0.0.1 --port 5555)

# R3. Factor list for this env (no server needed)
python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --list_variations \
  put_item_in_fridge_and_close_door --embodiment gr1_joint --object ranch_dressing_hope_robolab

# R4. Nominal, CLEAN mode: 20 episodes, extend to 50 if the interval straddles 0.8
python isaaclab_arena/evaluation/policy_runner.py --viz none --seed 0 \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
  --remote_host 127.0.0.1 --remote_port 5555 --num_episodes 20 --num_envs 1 --enable_cameras \
  --output_base_dir outputs/pilot_nominal_clean \
  put_item_in_fridge_and_close_door --embodiment gr1_joint --object ranch_dressing_hope_robolab
#   -> outputs/pilot_nominal_clean/<run>/episode_results_rank0.jsonl   (policy_runner.py:216)

# R5. Nominal, INCIDENT mode: same command with --num_envs 10 --num_episodes 50 --output_base_dir outputs/pilot_nominal_incident

# R6. Replay agreement: R4 again with the same --seed into outputs/pilot_nominal_clean_rep;
#     compare `success` per (env_id, episode_in_env) between the two JSONL files.

# R7. One shifted cell (dark): R4 plus these overrides after the env sub-command
  light.intensity.enabled=true "light.intensity.sampler_cfg.low=[100.0]" "light.intensity.sampler_cfg.high=[100.0]"

# R8. VRAM while R4 runs
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 10
```

Gate: R4 gives ≥ 0.8 with its interval reported. If R4 < 0.8, run the same with Candidate C (DROID: `droid_pnp_gr00t_experiment.yaml`, `gr00t.rst:84-86`, `rollout_limit.num_episodes: 20`) per cell before going further.

Not verified about these commands:
- whether `--viz none` and `--seed` are accepted by `policy_runner.py` in this exact form (`--seed` is defined at `A:isaaclab_arena/cli/isaaclab_arena_cli.py:53`; `--viz` comes from outside Arena's own CLI files);
- whether `--num_envs 1` is the default.

---

# Part 4. Order of work to the pilot gate

## Where `policy-wrapper/` comes from

### Facts
- The Arena GR00T client is `A:isaaclab_arena_gr00t/`, Apache-2.0 licensed per the file headers. These are the code paths the 6.4 bugs live in (see `docs/PLAN_UPDATE.md` §6.4):
  - **observation preprocessing**: `policy/gr00t_core.py:162-296` (`extract_obs_numpy_from_torch`, `resize_rgb_for_policy`, `build_gr00t_policy_observations`) and `utils/image_conversion.py:14-` (`resize_frames_with_padding`, `bgr_conversion` flag);
  - **action postprocessing**: `policy/gr00t_core.py:298-378`;
  - **joint remaps**: `utils/joints_conversion.py` plus the YAML joint spaces in `embodiments/gr1/`;
  - **policy client and hold action**: `policy/gr00t_remote_closedloop_policy.py:75-237`;
  - **config**: `policy/config/gr00t_closedloop_policy_config.py` and the per-task YAML;
  - **chunk scheduling**: Arena core, `A:isaaclab_arena/policy/action_scheduling/{action_scheduler,action_chunk_scheduler,synced_batch_action_scheduler}.py`.
- These files import `isaaclab_arena.*` (`PolicyBase`, `register_policy`, the schedulers) and `isaaclab_arena_g1` constants (`policy/gr00t_remote_closedloop_policy.py:20-32`; `policy/gr00t_core.py:30-39`).
- Server-side values (denoising steps, normalisation stats, policy RNG) sit in the checkpoint and in `G:gr00t/policy/gr00t_policy.py`, not in the client (facts §8).
- Arena already has client tests that use a fake client, "so no server is required" (`A:isaaclab_arena_gr00t/tests/test_gr00t_remote_closedloop_policy.py:9-17,147-269`). Whether they import without Isaac Sim is **UNVERIFIED**.
- vla-router reads the eval config from `config/eval.yaml` in the working copy (`R:DECISIONS.md:275,286`; test fixture `R:tests/workspace_cases.py:59-60`). It compares `action_horizon`, `replan_steps`, `denoising_steps`, `control_hz`, `norm_stats`, `camera_poses`, `image_size`, `render_settings` and `instruction_templates` through a hand-checked `ConfigMapping` (`R:router/stage1/cfg_contract.py:38-52`).

### Proposal (decision for you)
`policy-wrapper/` is a **new git repo**, built from one snapshot commit. vla-router's `WorkspaceManager` names it `<label>@<tree12>`.

```
policy-wrapper/
  config/eval.yaml                      # new: the one file the contract check reads (below)
  policy_wrapper/
    policy/gr00t_core.py                # copy of A:isaaclab_arena_gr00t/policy/gr00t_core.py
    policy/gr00t_remote_closedloop_policy.py   # copy; reads config/eval.yaml client section
    policy/config/gr00t_closedloop_policy_config.py, task_mode.py   # copies
    scheduling/action_scheduler.py, action_chunk_scheduler.py, synced_batch_action_scheduler.py  # copies of A:isaaclab_arena/policy/action_scheduling/*
    utils/image_conversion.py, joints_conversion.py, io_utils.py, robot_joints.py               # copies
    embodiments/gr1/gr00t_26dof_joint_space.yaml, 36dof_joint_space.yaml, 54dof_joint_space.yaml, gr1_arms_only_data_config.py  # copies
    serve.py                            # new: starts G:Gr00tPolicy from config/eval.yaml server section,
                                        #      sets action_head.num_inference_timesteps, seeds torch per request
    instrument.py                       # new: logger hook (policy-input FrameStats, step pairs, raw actions) -> per-episode JSON
  tests/                                # copies of the fake-client tests, re-pointed at policy_wrapper
```

- Imports of `isaaclab_arena_gr00t.*` are rewritten to `policy_wrapper.*`. `isaaclab_arena.policy.policy_base` and `register_policy` stay imported from Arena, which is installed and read-only.
- **What hagent may edit:** everything under `policy-wrapper/`. Arena's environments, embodiments and tasks are not part of the repair target. Gains enter only through `config/eval.yaml` → vla-isaac actuator overrides.
- The squashed single-commit base and the "no tell-tale comments" rule (`P:323-328`) apply when bugs are injected.

`config/eval.yaml` (pilot values copied from `A:isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml` and the env facts above):

```yaml
client:
  language_instruction: "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."
  embodiment_tag: GR1
  action_horizon: 16
  action_chunk_length: 16            # = replan steps
  scheduler: chunk
  modality_config_path: policy_wrapper/embodiments/gr1/gr1_arms_only_data_config.py
  policy_joints_config_path: policy_wrapper/embodiments/gr1/gr00t_26dof_joint_space.yaml
  action_joints_config_path: policy_wrapper/embodiments/gr1/36dof_joint_space.yaml
  state_joints_config_path: policy_wrapper/embodiments/gr1/54dof_joint_space.yaml
  task_mode_name: gr1_tabletop_manipulation
  pov_cam_name_sim: robot_pov_cam_rgb
  original_image_size: [512, 512, 3]
  target_image_size: [512, 512, 3]
  bgr_conversion: false
server:
  checkpoint: nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge   # pin the HF revision hash
  embodiment_tag: GR1
  denoising_steps: 4
  seed_per_request: true
  norm_stats_override: null          # path; home of the normalisation bugs
env:
  task: put_item_in_fridge_and_close_door
  embodiment: gr1_joint
  object: ranch_dressing_hope_robolab
  control_hz: 50
  actuator_overrides: {}             # home of the gain bug
  render: {}                         # render preset; empty = Arena default (UNVERIFIED what that is)
```

`ConfigMapping.keys`: `action_horizon→client.action_horizon`, `replan_steps→client.action_chunk_length`, `denoising_steps→server.denoising_steps`, `control_hz→env.control_hz`, `image_size→client.target_image_size`, `instruction_templates→client.language_instruction`, `render_settings→env.render`, `norm_stats→server.norm_stats_override`. Check it by hand once (`R:router/stage1/cfg_contract.py:7-9`).

## Steps

Each step has four parts: what is built, done when, how it is tested **without a GPU** (fakes or mocks; contract tests against `RolloutBackend` `R:ladder/backend.py:25-32`, `TraceSource` `R:router/prereq/trace.py:260-281`, `BehaviorSource` `R:router/prereq/behavior_trace.py:135-154`), and how it is tested **with Isaac**.

**Step 0: Compute and installs.**
- *Built:* nothing in our repos. A remote GPU (≥24 GB) with Arena, Isaac Lab, GR00T and the checkpoint (R0-R2).
- *Done when:* a headless zero-action rollout of the pilot env runs; the server answers `ping`; VRAM numbers are recorded (R8 and facts §9).
- *Without GPU:* `pytest isaaclab_arena_gr00t/tests/test_gr00t_remote_closedloop_policy.py` on the laptop, to learn whether Arena's client tests import without Isaac Sim (UNVERIFIED). The result decides whether step 2's tests can run in laptop CI.
- *With Isaac:* R3, plus `--policy_type zero_action --num_steps 200`.

**Step 1: Pilot confirmation and training spec.**
- *Built:* `vla-isaac/specs/extract_training_spec.py`. It reads the checkpoint's `config.json`, `processor_config.json` and `statistics.json`, plus Arena's modality config and closed-loop YAML, and emits a vla-router `TrainingSpec` (`R:router/schemas.py:173-196`) as JSON.
- *Done when:* the R4 nominal is ≥0.8 with its interval, R5 and R6 are recorded, and the training-spec JSON is committed with every field filled or marked missing.
- *Without GPU:* unit tests on the downloaded JSON files (download needs no GPU): `action_horizon=16`, `replan_steps=16`, `denoising_steps` from config, `control_hz=50`, layouts from the modality config, `norm_stats` present. Also a round-trip through `TrainingSpec`.
- *With Isaac:* R4-R7.

**Step 2: `policy-wrapper/` base checkout.**
- *Built:* the repo above (copies plus `serve.py` plus `config/eval.yaml`, no `instrument.py` yet), registered as a `WorkspaceManager` base.
- *Done when:* Arena's policy runner with `--policy_type policy_wrapper.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy` and `serve.py` gives a nominal rate whose interval overlaps the stock R4 result on the same seeds.
- *Without GPU:*
  - the copied fake-client tests;
  - a fake ZMQ REP server returning a fixed chunk, checking the obs dict structure, chunk replay and per-request seeding;
  - `WorkspaceManager.materialize` plus `file_config_loader("config/eval.yaml")` round-trip;
  - `cfg_contract` with the real `ConfigMapping` on base vs edited `eval.yaml` (one violation per edited key);
  - Rung 0 (`ladder/rung0.py`) on the repo: lint, tests, smoke.
- *With Isaac:* the parity run (20 episodes, clean).

**Step 3: `vla-isaac` `IsaacBackend` v0.**
- *Built:* a class implementing `RolloutBackend.run(scene, code_ref, actor, seeds, mode)`. It does the following in order:
  1. `materialize(code_ref)`;
  2. start `serve.py` from that copy;
  3. run `policy_runner.py` as a subprocess with `PYTHONPATH` set to the copy;
  4. map `SceneSpec` factors to variation overrides;
  5. set `num_envs` = 1 for clean mode or N for incident mode;
  6. map seeds to (build seed, episode index);
  7. parse `episode_results_rank*.jsonl` into `EpisodeResult(seed, success, logs={code_key, paths})`.
- Replaces the placeholder at `R:ladder/backend.py:35-56`. The cache key gains the backend id (ISSUES #19).
- *Done when:* the contract suite passes, and a real 10-seed run returns 10 results in seed order with `code_key` set.
- *Without GPU:* a `FakeArenaRunner` subprocess that writes fixture JSONL. Contract tests shared with `MockBackend`:
  - one result per seed, in order;
  - same (scene, code, actor, mode, seeds) gives the same call;
  - factor → override mapping is total and injective, and an unknown factor is an error, not a silent nominal;
  - a crashed subprocess raises, and the loop ends UNRESOLVED (F3 behaviour);
  - restore(i) removes exactly one override.
- *With Isaac:* nominal plus one lighting cell, 10 seeds each.

**Step 4: Instrumentation → `TraceSource` / `BehaviorSource`.**
- *Built:*
  - `instrument.py` in the wrapper (policy-input `FrameStats`, `StepPair` from the scheduler index, raw actions);
  - the HDF5 recorder → `joint_cmd/joint_pos`, `SimHealth`, `GraspRecord` (hand-closure scalar for GR1);
  - Arena `TrackedEventCfg` for reach and grasp;
  - `IsaacTraceSource._load` / `IsaacBehaviorSource._load`, which read files the backend wrote;
  - a vla-router change to map the close-door subtask (decision 4).
- *Done when:* `route()` on a real nominal incident runs Stage 1 and C_behavior with no `PartialSkip` for the observation, action and staleness signals.
- *Without GPU:*
  - synthetic HDF5 (h5py) and JSONL fixtures → traces;
  - `load()` refuses a wrong or missing `code_key` (inherited template test);
  - `StepPair` arithmetic on a scripted chunk schedule (constant extra delay recovered);
  - `FrameStats` on known images (RGB vs BGR channel means, a flip shows in the profiles).
- *With Isaac:* nominal plus an RGB/BGR swap injected in `config/eval.yaml` (`bgr_conversion: true`). The observation contract fires and names the image path.

**Step 5: Reference controller (Mimic) as `ActorRef(kind=REFERENCE, pipeline=RAW)`.**
- *Built:* a backend branch that runs Isaac Lab Mimic `generate_dataset.py` (`--generation_num_trials N`, pilot task, `gr1_pink`) on the scene and returns success over trials. Also the producer of `reference_unqualified` (ISSUES #5).
- *Done when:* there is a qualification table for nominal and each candidate cell.
- *Without GPU:* parse fixture Mimic output; the threshold rule produces or omits `reference_unqualified`; `WRAPPER` pipeline requests for the reference raise `NotImplementedError` (G3 deferred, decision 3) and leave the G3 rows undecided.
- *With Isaac:* N=20 trials, nominal plus 2 cells. Whether Mimic accepts variation overrides is **UNVERIFIED**; this step settles it.

**Step 6: Failure map, replay agreement, clean vs incident gap.**
- *Built:* a grid runner over `IsaacBackend`; the replay-agreement tool (ISSUES #15).
- *Done when:* at least one cell has nominal ≥0.8, shifted ≤0.4, and an unaffected reference; replay agreement is measured; the mode gap is measured.
- *Without GPU:* grid runner and agreement tool on `MockBackend` with a known agreement rate.
- *With Isaac:* grid of about 12 cells × 10-20 episodes.

**Step 7: Pilot gate (6.2).**
- *Built:*
  - Rung A: fine-tune on the existing Mimic LeRobot set plus jitter.
  - Rung B: Mimic regeneration of 50-100 episodes under the failing light, converted with Arena's converter, mixed about 50/50 with nominal.
  - A fine-tune launcher with the 5.4 default recipe.
  - Held-out cells.
- *Done when:* held-out success improves with intervals reported and nominal does not regress, or a documented negative result.
- *Without GPU:* conversion on fixture HDF5 (Arena has `isaaclab_arena_gr00t/tests/test_convert_hdf5_to_lerobot.py`); a deterministic held-out split; the mixer ratio; the launcher builds the expected `launch_finetune.py` command.
- *With Isaac:* fine-tune (≥24 GB), then Rung 1 on held-out and nominal.

---

# (1) The five decisions you must make

| # | Decision | My recommendation | What would change it |
|---|---|---|---|
| 1 | **Where simulation and training run.** The laptop has 6 GB; Arena's fine-tune minimum is 24 GB | Rent one ≥24 GB GPU (48 GB preferred, so the server and Isaac Sim share it) for steps 0-7; keep the laptop for mocks | VRAM runs showing GR00T inference plus Isaac fit in 6 GB (only the eval half would move local; training still cannot) |
| 2 | **Pilot task** | GR1 fridge, ranch bottle, lighting family (Candidate A) | R4 < 0.8, or camera offset being mandatory → DROID maple-table cell with ≥0.8 measured |
| 3 | **Reference controller scope** | Mimic as RAW reference (G2/C1) only; defer G3 (no joint-space reference exists); rows that need G3 stay undecided and are reported | If G3 localisation proves essential for the code-route numbers, build a dataset-action replay through the wrapper (needs Isaac Lab `reset_to`, UNVERIFIED) |
| 4 | **Rung B method** | Mimic regeneration under the shifted factor (exists, exact actions, no drift) instead of building state-replay re-render | If Mimic ignores variation overrides, or the paper must claim "same seeds re-rendered", build state-replay on Isaac Lab APIs |
| 5 | **Bug-suite scope** | Client-side bugs in `policy-wrapper/` first, plus the two server-side ones (denoising, normalisation) via `serve.py`; one injected async mode for the infra case, stated as such | Schedule slip (cut to client-side only), or a decision that an injected async mode is too artificial for an "infra" claim (then drop the infra row from the evaluation and say so) |

Lesser choices taken as defaults (override if you disagree):
- the router seed is (build seed, episode index);
- `PHASES` gains `door`, or the door subtask maps to "place", decided in step 4;
- the default fine-tune recipe freezes vision, and Arena's tune-visual recipe is the ablation.

# (2) Everything I could not verify

Facts that need a run:
- GR00T-N1.6-3B and pilot checkpoint values: `tune_top_llm_layers`, `processor_config.json` (jitter, crop, resize), `action_horizon`, `num_inference_timesteps`, and the DROID checkpoint's modality config and relative/absolute action mode. (R1 `grep`; `hf download nvidia/GR00T-N1.6-3B config.json processor_config.json`.)
- Inference VRAM; Isaac Sim with cameras VRAM; both together (facts §9, R8).
- Pilot nominal success in clean and incident mode; replay agreement; the dark-cell drop (R4-R7).
- Loaded parameter count after LLM truncation (first fine-tune log, `setup.py:105-110`).
- Whether Mimic `generate_dataset.py` accepts Arena variation overrides (step 5).
- Whether Arena's GR00T client tests run without Isaac Sim (step 0).
- Whether `policy_runner.py` writes the HDF5 demo recorder during policy runs (Rung A data, option A).
- Exact behaviour of `--viz none`, `--seed` and the default `--num_envs` in `policy_runner.py`.

Facts I could not settle from the files:
- Whether "Cosmos-Reason-2B variant" (README) and `nvidia/Eagle-Block2A-2B-v2` (code) are the same weights.
- Whether the Eagle LoRA keys survive checkpoint loading or train under `tune_visual=False`.
- Isaac Sim 6.0.x hardware requirements (external link only).
- Isaac Lab 3.0.0 APIs: `reset_to`, writing per-step state, rendering without physics, `replay_demos.py` options, and the recorder classes. `submodules/IsaacLab` is not checked out.
- What GR1's hand action terms are (only `BinaryJointAction` grippers are recorded by `GripperStateRecorder`).
- The cause of Arena's documented 0.88 (1 env) vs 0.605 (10 envs) gap on open-microwave.
- Arena's default render settings (for the render-mismatch row).
- Whether GR00T-N1.6-DROID's checkpoint modality config matches the pre-registered `oxe_droid` config (relative joints) or Arena's "absolute joint positions" wording (`gr00t.rst:55`).
- The license implications of copying Arena files into `policy-wrapper/` (headers say Apache-2.0; not reviewed).
