# Proposed edits to `docs/plan.md`, based on the N1.6 and Arena facts

Status: proposal only. `docs/plan.md` is not edited. Citation prefixes are the same as in `docs/facts_gr00t_n16.md`: `G:` is the pinned GR00T, `A:` is Arena, `R:` is vla-router, `P:` is `docs/plan.md`. "Facts" sections hold only cited facts. "Decisions" sections hold proposals for you to accept or reject. "Opinions" sections hold my judgment.

Part 1 is the four checks (a)-(d) the request asked for, because the section edits depend on them. Part 2 is the per-section verdicts with replacement text.

---

# Part 1. Checks

## (a) Which Arena variations support one-factor restores and replay of logged state

### Facts

| Variation (class) | Kind | Can it hold one fixed value? | Visual only? | On which assets | Source |
|---|---|---|---|---|---|
| `HDRImageVariation` | build-time | yes: `hdr_names=[<one>]` | yes (dome-light HDR) | `light` (DomeLight) | `A:isaaclab_arena/variations/hdr_image_variation.py:27-33,37,60`; attached at `A:isaaclab_arena/assets/object_library.py:468` |
| `LightIntensityVariation` | build-time | yes: `sampler_cfg.low == high` (default U[100, 2000]) | yes | `light`, `directional_light` | `light_intensity_variation.py:21-26,28,48-50`; `object_library.py:469,523` |
| `LightColorVariation`, `LightColorTemperatureVariation` | build-time | yes, through the sampler | yes | `light`, `directional_light` | `light_color_variation.py:32`; `light_color_temperature_variation.py:30`; `object_library.py:470-471,524-525` |
| `LightDirectionVariation` | build-time | yes | yes | `directional_light` only | `light_direction_variation.py:73`; `object_library.py:522` |
| `CameraExtrinsicsVariation` / `CameraIntrinsicsVariation` | run-time (`mode="reset"`) | yes, through the sampler | changes the camera only, not the scene | registered **only** for DROID and Franka embodiments | `camera_extrinsics_variation.py:54,85-93`; `camera_intrinsics_variation.py:48,88-96`; `A:isaaclab_arena/embodiments/embodiment_base.py:281-288`; called at `embodiments/droid/droid.py:161`, `embodiments/franka/franka.py:104` |
| `ObjectMassVariation` | run-time | yes | **no** (physics) | every object | `object_mass_variation.py:45,70-78`; `assets/object_base.py:77` |
| `ObjectDisappearVariation` | run-time | yes (Bernoulli) | yes, removes a distractor | every object | `object_disappear_variation.py:74,97-103`; `object_base.py:78` |
| `RigidAssetSelectionVariation` | build-time | yes | **no**: `changes_object_geometry = True` | objects | `rigid_asset_selection_variation.py:50,65` |
| Material / texture | — | — | — | NOT FOUND (no class in `A:isaaclab_arena/variations/`) | directory listing |

- **Restoring a factor means disabling its variation.** Variations "start disabled" (`variation_base.py:8-11,38`). A fixed value is set with Hydra overrides after the environment sub-command, for example `light.intensity.enabled=true "light.intensity.sampler_cfg.low=[100.0]" ...`. The syntax is documented at `A:docs/pages/concepts/variations/variations.rst:85-131`. `--list_variations` prints them per environment (`variations.rst:60-70`).
- **Each variation has its own random stream.** The seed is `sha256(f"{seed}:{variation_key}")`, so "with one shared generator, a variation's values change when others are enabled, disabled or reordered" does not happen. The generator lives on CPU, so the k-th episode gets the same run-time value for any `num_envs` (`variation_base.py:124-131`). **This is what a one-factor restore needs:** turning factor i off leaves every other factor's draws unchanged for the same seed.
- **Build-time variations are one value per environment build** (`variation_base.py:184-199`; `hdr_image_variation.py:60`; `light_intensity_variation.py:48-50`). Each lighting condition is therefore a separate build ("rebuild"), not a per-episode draw.
- **Recording.**
  - Each episode's variation values go to the episode JSONL under `"variations"`, and the relation placement goes under `"placement"` (`A:isaaclab_arena/recording/common_terms.py:38-59`).
  - The core record carries `seed` (= `env.cfg.seed`, the build seed), `env_id`, `episode_in_env`, `success`, `episode_length` and `language_instruction` (`common_terms.py:18-35`).
  - The policy runner writes it to `<output_dir>/episode_results_rank<k>.jsonl` (`A:isaaclab_arena/evaluation/policy_runner.py:216-218`). The experiment runner writes `episode_results_rebuild<i>.jsonl` (`run_execution.py:117-119`).
- **Replay** (`A:isaaclab_arena/variations/recorded_variation_replay.py:31-126`):
  - Recorded values are bound to enabled variations. Enabled variations absent from the record are sampled live (`:91-115`).
  - **Every recorded key must belong to an enabled variation**, or loading asserts (`recorded_variation_samples.py:122-142`). Build-time values must be constant across a record (`:86-104`).
  - Arena's own workflow checks that replayed `placement` and `variations` match the source rows exactly, at a different `num_envs` (`A:docs/pages/example_workflows/analysis/recorded_variations.rst:127-168`).
- **"Logged state" here means variation values and relation placement, not per-frame simulator state.** Nothing in the recorded-variation path records robot or object state per step. That lives in the HDF5 trajectory recorders, see (b).

### What this means for the plan's restores (opinion)
- Lighting (HDR, intensity, colour, colour temperature, direction) and distractor presence support clean one-factor restores, both live (same seed, variation disabled) and from a recording (strip one key from the JSONL, then replay).
- Camera offset supports them only on DROID and Franka. GR1 has no camera variations registered.
- "Glossy material" has no Arena variation. It would need new code, and a check that only the visual material changed (`P:562`).
- A small vla-isaac helper is required to "replay this recording with factor i restored", because Arena refuses a record that carries a key whose variation is disabled.

## (b) Can Arena re-render a recorded episode with a changed variation (Rung B), or only re-simulate?

### Facts
- **Only re-simulation exists in Arena.**
  - `ReplayActionPolicy` replays an HDF5 episode's actions open-loop (`A:isaaclab_arena/policy/replay_action_policy.py:32-84`).
  - It has `get_initial_state()` (`:89-90`), but nothing in Arena calls it (`grep get_initial_state|reset_to` finds only the definition).
  - Recorded-variation replay reproduces conditions and then simulates with whatever policy runs, see (a).
- **Arena warns that action replay is not reproducible:** "physics in Isaac Lab is not deterministically reproducible across `env.reset()` calls. A demonstration may therefore fail during replay even though it succeeded during generation" (`A:docs/pages/example_workflows/sequential_static_manipulation/step_3_data_generation.rst`, "Validate Generated Data" note). The docs replay with `--device cpu` to reduce this.
- **Per-step state is recordable.** `ArenaEnvRecorderManagerCfg` extends Isaac Lab's `ActionStateRecorderManagerCfg` and adds camera frames and policy actions (`A:isaaclab_arena/terms/recorders.py:66-70`). The trajectory terms add `InitialStateRecorderCfg`, `PostStepStatesRecorderCfg`, pre-step actions, processed actions and episode id (`terms/recorders.py:277-289`), plus end-effector poses (`:98-205`) and binary-gripper opening (`:210-275`). The builder attaches the demo recorder only when cameras are enabled (`A:isaaclab_arena/environments/arena_env_builder.py:474`).
- Isaac Lab's own state-setting API (`reset_to`, scene state write, rendering without a physics step) is in the **uninitialised** `submodules/IsaacLab`: **UNVERIFIED** for Isaac Lab 3.0.0.
- **Isaac Lab Mimic is available for Arena tasks.** `--mimic` CLI flag; GR1 sequential task annotated demos and a 100-demo generated set can be downloaded; `generate_dataset.py --generation_num_trials 100 --num_envs 10` (`A:docs/pages/example_workflows/sequential_static_manipulation/step_3_data_generation.rst`, steps 1-2). Mimic "generates additional demonstrations ... by using rigid body transformations" (same file).

### Opinion
- Plan 5.3's kinematic state-replay re-render (`P:413`) has to be **built** in vla-isaac on Isaac Lab APIs I could not read. The order would be: rebuild the environment with the new build-time lighting, reset to the recorded initial state, write the recorded per-step states with physics off, render, and check the poses.
- For GR1 there is a cheaper Rung B that already exists: **regenerate with Mimic under the shifted factor.** Mimic is closed-loop on object poses, ignores pixels, records exact actions, and only keeps successful trials. That is "Isaac randomization" with exact actions and no pose drift. The catch is that its trajectories are new ones, not re-renders of the failing seeds. Whether `generate_dataset.py` accepts Arena variation overrides is **UNVERIFIED**.

## (c) What a reference controller needs, and what already exists

### Facts
- **Plan requirements:** privileged state, actions in the policy's own output format, qualified on nominal and on every shifted condition (`P:57,82`). It runs RAW (G2, C1) and through the wrapper (G3) (`R:router/schemas.py:44-53`).
- **GR1 (pilot candidate).**
  - Mimic data generation exists (above). It drives `gr1_pink` (end-effector targets through the Pink IK controller), while the policy runs on `gr1_joint` (`A:.../static_manipulation/step_5_evaluation.rst:196-203`).
  - The HDF5 datasets carry `processed_actions`. Arena's converter uses those as the policy action (`action_name_sim: processed_actions`, `A:.../static_manipulation/step_4_policy_training.rst:94-97`). That is joint space, which the client remaps with `joints_conversion` (`A:isaaclab_arena_gr00t/policy/gr00t_core.py:298-378`).
- **GR00T `ReplayPolicy`** serves dataset actions through the normal server API (`G:gr00t/eval/run_gr00t_server.py:73-86`). The wrapper cannot tell it apart from the model.
- **DROID.**
  - `droid_differential_ik` embodiment (relative pose IK, `A:isaaclab_arena/embodiments/droid/droid.py:243-293,488-500`).
  - `PoseTargetMotionGenerator`, "Track absolute pose targets and grippers using live controller and scene state" (`A:isaaclab_arena_vlm_agent/pose_target_motion_generator.py:21`).
  - cuRobo IK feasibility only, Docker only (`A:docs/pages/quickstart/installation.rst` uv note).
  - No DROID Mimic source demos are referenced in Arena docs. The only HF datasets are GR1, G1, Franka-lift and a DROID camera-sensitivity sample (grep of `nvidia/Arena-*` in `A:docs`).
- **No scripted pick-and-place state machine exists** in Arena (grep for `scripted|state_machine|StateMachine` finds none outside the VLM agent).
- vla-router has **no qualification code** (`ISSUES.md:55-57`, #5).

### Opinion: what to build
- **Scripted controller vs Mimic.**
  - Mimic for GR1: existing code and data. Its trial success rate per condition is the qualification number.
  - A scripted state machine (pose targets from privileged object poses, then IK) for DROID: new code, about 1-2 weeks.
- **RAW (G2/C1):**
  - GR1: run Mimic generation on the scene and report success over N trials.
  - DROID: run the state machine on `droid_differential_ik`.
- **Through the wrapper (G3):** the reference must emit actions in the policy's format (GR1: 26 policy joints; DROID: 7 joints + gripper) and send them through the wrapper's action postprocessing and chunk scheduler. The plan's G3 assumes this exists. Nothing does.
  - The cheapest version is a `ReplayPolicy`-style server that returns recorded `processed_actions` remapped to policy joints, starting from the recorded initial state. It needs Isaac Lab `reset_to` (**UNVERIFIED**) and suffers the replay non-determinism Arena warns about.
  - A closed-loop version means a GR1 joint-space controller, which does not exist.
- My recommendation is in the decisions list in `docs/PILOT_AND_ORDER.md`: keep G2/C1 with Mimic for the pilot and defer G3.

## (d) How Arena's progress tracker and recorders map onto `EpisodeTrace` / `BehaviorTrace`

### Facts: Arena sources
- **Episode JSONL** (one row per finished episode): `job_name`, the core fields, `variations`, `placement` (`A:isaaclab_arena/recording/common_terms.py:18-59`; manager `A:isaaclab_arena/recording/episode_recorder_manager.py:80-103`) and `progress` (`A:isaaclab_arena/recording/progress_terms.py:18-67`).
  - `progress` contains `overall_score`, `all_complete`, per-criteria score and completion, **`events` with `step`, `criteria_name`, `predicate_index`, `predicate_name`**, tracked events, and `control_step_dt`.
  - Extra terms are allowed: a term is `func(env, env_id) -> flat JSON dict` (`episode_recorder_manager.py:21-31`). The builder takes `extra_terms` (`arena_env_builder.py:281-299`).
- **HDF5 trajectory recorders** (`A:isaaclab_arena/terms/recorders.py`):
  - initial and per-step states;
  - pre-step actions and processed actions;
  - camera frames (`:29-41`);
  - EE poses and velocities from `ee_frame` (`:98-205`);
  - binary-gripper opening (0 open, 1 closed) and `is_commanded_open`, only for `BinaryJointAction` terms (`:210-275`).
- **Progress predicates for pick-and-place:** `objects_settled` → `object_is_above_height` → `object_on_destination` held for N steps (`A:isaaclab_arena/tasks/pick_and_place_task.py:167-191`). There is no reach or grasp predicate. `TrackedEventCfg` can add one (`A:isaaclab_arena/progress_tracking/tracked_event_cfg.py:13-38`) from existing predicates such as `gripper_released`, `gripper_not_grasping` (`tasks/predicates/gripper.py:19,48`) and `gripper_distance_from_object_exceeds_threshold` (`tasks/predicates/spatial.py:26`).
- **GR1 sequential task:** a `PutAndCloseDoorTask(CompositeTaskBase)` with subtasks `[pick_and_place_task, close_door_task]` (`A:isaaclab_arena_environments/gr1_put_and_close_door_environment.py:74,195-210`).
- **vla-router side:**
  - `EpisodeTrace` fields (`R:router/prereq/trace.py:181-201`).
  - `BehaviorTrace` / `GraspRecord` / `SubtaskProgress` (`R:router/prereq/behavior_trace.py:43-101`). `PHASES = ("reach","grasp","lift","place")` is fixed (`:36`), and any other name raises (`:80-82`).

### Mapping

| vla-router field | Arena source today | Gap |
|---|---|---|
| `EpisodeTrace.seed` | `seed` = build seed, plus `env_id`, `episode_in_env` (`common_terms.py:24-26`) | No per-episode seed. Define router seed = (build seed, k-th episode); variation draws are stable per k (`variation_base.py:129-130`). Policy sampling seed: none (`G:gr00t_n1d6.py:310`) |
| `.mode` | not recorded | vla-isaac knows it: clean = `--num_envs 1` |
| `.frames[].render` | HDF5 camera frames (`terms/recorders.py:29-41`), or `--record_camera_video` mp4 (`policy_runner_cli.py:111-118`) | Compute `FrameStats` offline from them |
| `.frames[].policy_input` | nothing | Wrapper hook after `resize_rgb_for_policy` (`gr00t_core.py:201-224`) |
| `.step_pairs` | nothing; the scheduler knows `current_action_index` (`A:isaaclab_arena/policy/action_scheduling/action_chunk_scheduler.py:54-100`) | Wrapper hook: obs step at fetch, offset = chunk index |
| `.raw_actions` | nothing; the server reply is `robot_action_policy` (`gr00t_remote_closedloop_policy.py:197`) | Wrapper hook |
| `.joint_cmd` / `.joint_pos` | HDF5 processed actions and post-step states | Extract joint subset |
| `.gripper_cmd_width` / `.gripper_width` | `GripperStateRecorder` (binary grippers only) | DROID: yes (`droid.py:200,470-476`). GR1 hands are not binary: **gap**, define a hand-closure scalar |
| `.sim` (`SimHealth`) | object initial pose from placement or initial state; "settled" exists as a predicate | NaN fraction and penetration not recorded: new term |
| `.code_key` | nothing | Extra episode-recorder term or vla-isaac stamping. Mandatory (`R:router/prereq/trace.py:246-257`) |
| `BehaviorTrace.success` | `success` (`common_terms.py:20-22`) | — |
| `.subtasks` reach | nothing | Tracked event (EE within d of object) |
| `.subtasks` grasp | nothing | Tracked event (gripper closed and object in contact or moving with EE) |
| `.subtasks` lift | `object_is_above_height` event (`pick_and_place_task.py:175-179`) | Read from `progress.events` |
| `.subtasks` place | `object_on_destination` event (`:180-190`) | Same |
| GR1 "close door" subtask | composite subtask 1 | **No slot in `PHASES`.** Extending `PHASES` is a vla-router change |
| `.grasp` (`GraspRecord`) | EE pose (`terms/recorders.py:98-205`) + object pose (states) at the gripper-close transition | DROID: derivable. GR1: needs a "hand closed" definition |
| `.grasp.distractor_pos` | per-step states of other rigid objects | Extract |

### Opinion
- Most of `EpisodeTrace` must come from a **wrapper-side logger**: policy-input stats, step pairs, raw actions. Arena records none of these.
- Most of `BehaviorTrace` can be derived from the HDF5 recorders and the progress events, plus two tracked events (reach, grasp).
- The GR1 composite task does not fit `PHASES`, which is one more reason the pilot choice matters (see `docs/PILOT_AND_ORDER.md`).

---

# Part 2. Per-section verdicts

| Section | Edit? | Why (fact) |
|---|---|---|
| 3.1 | **Yes** | Policy and version are now known (N1.6, Eagle backbone); module names differ from the plan's guesses; the clean/incident mode mapping and the compute limit are concrete |
| 5.2 | **Yes** | Mimic data and generation exist for GR1; Arena's policy runner does not write training datasets by default |
| 5.3 | **Yes** | Rung A = GR00T's built-in jitter and rotation; Rung B state replay does not exist; Mimic regeneration is an alternative; no material variation |
| 5.4 | **Yes** | "projector" means action-head MLPs in this code; `tune_top_llm_layers` may train LLM layers by default; Arena's recipe tunes vision; LoRA is vendored only; 24 GB minimum |
| 6.1 | **Yes** | Grid factors come from Arena variations; build-time means one build per cell; physics non-determinism is documented |
| 6.4 / 6.5 | **Yes** | Bugs now have concrete homes; some move server-side; the stale-server infra fault cannot happen in Arena's synchronous client without an injected mode; frame-transform does not apply to a joint-space policy |
| 8 | **Yes** | Steps 4-7 of the original order are already built on mocks; the hardware gate comes first |
| 9 | **Yes** | New risks from the facts |

The replacement texts follow. Each block replaces the whole subsection named.

---

## 3.1 Environment: replacement text

> ### 3.1 Environment
>
> - **Simulator:** Isaac Lab Arena at `external/IsaacLab-Arena` (Isaac Sim 6.0.1, Isaac Lab 3.0.0, `A:docs/pages/quickstart/installation.rst:32`), headless. Arena's `submodules/IsaacLab` must be initialised before anything runs.
> - **Policy under test:** GR00T **N1.6** from Arena's pinned submodule `submodules/Isaac-GR00T` at `e29d8fc` (in `origin/n1d6`). Code facts are in `docs/facts_gr00t_n16.md`. The ones the design depends on:
>   - Backbone `EagleBackbone`: SigLIP2 `vision_model`, `mlp1` VL projector, Qwen3-1.7B `language_model` truncated to 16 layers.
>   - Action head `Gr00tN1d6ActionHead`: a 32-layer flow-matching DiT, 4 denoising steps.
>   - Tune flags `tune_llm`, `tune_visual` (also covers `mlp1`), `tune_projector` (action-head state and action encoders and decoder), `tune_diffusion_model`; `tune_vlln` and `tune_top_llm_layers` are not on the CLI.
>   - The pilot checkpoint is chosen in 6.2 and `docs/PILOT_AND_ORDER.md`.
> - **Policy serving:** GR00T's ZMQ REQ/REP server (`gr00t/eval/run_gr00t_server.py`), one request at a time. The Arena client is `isaaclab_arena_gr00t`, synchronous: the sim waits for every chunk. The client code we evaluate and repair is a copy of that package in `policy-wrapper/` (Section 8, step 3).
> - **Inference modes:**
>   - *Clean* = `--num_envs 1`, one server, default chunk scheduler.
>   - *Incident* = how the eval was run, `--num_envs N` sharing one server.
>   - The client is synchronous, so incident mode adds no action staleness unless a fault is injected (6.4).
>   - Arena's own docs show a nominal gap between the two modes on GR1 open-microwave: 0.88 over 17 episodes in one env vs 0.605 over 200 episodes in 10 envs (`A:docs/pages/example_workflows/static_manipulation/step_5_evaluation.rst:127,194`). Measure this gap on the pilot task before the infra row is trusted.
> - **Seeds:** Arena seeds a build (`--seed`), not an episode. Run-time variation draws are stable for the k-th episode at any `num_envs`. A router "seed" is (build seed, episode index). The policy's flow-matching noise uses the global torch RNG and is never seeded by GR00T. The wrapper must seed it per request for replay agreement (3.2.2).
> - **Fine-tuning data format:** GR00T-flavoured LeRobot v2 (parquet + mp4 + `meta/modality.json`). Arena's `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py` converts Isaac Lab HDF5 recordings.
> - **Reference controller** (3.2.2, 5.2 B):
>   - For GR1 tasks, Isaac Lab Mimic generation (existing; source demos downloadable) is the RAW reference for G2/C1. Its trial success rate per condition is its qualification number.
>   - For DROID tasks, a scripted pose-target state machine on `droid_differential_ik` would be new code.
>   - G3 (reference actions through the wrapper) needs actions in the policy's joint format. No existing controller provides them; see 3.2.6.
> - **Compute:** the development laptop has one RTX 3050 Laptop GPU with 6 GB. GR00T fine-tuning needs ≥24 GB per Arena's low-hardware recipe. Isaac Sim with cameras plus the 3B policy server will not fit in 6 GB; confirm with the VRAM runs in `docs/facts_gr00t_n16.md` §9. All simulator and training work runs on a remote GPU. The laptop runs mocks and contract tests only.
> - The router prerequisites are built alongside the failure map (3.2.2, Section 8): scene spec (= Arena variation overrides), training spec (= checkpoint `processor_config.json`, `statistics.json` and the Arena modality config), instrumentation, clean vs incident modes, rollout cache, replay agreement.

## 5.2 Source of clean nominal rollouts: replacement text

> ### 5.2 Source of clean nominal rollouts (user-selectable)
> Implement one fully and keep the others as documented stubs.
>
> | Option | Prerequisite | Status in Arena | Best when |
> |---|---|---|---|
> | **A. The VLA's own successful rollouts** (rejection-sampled) | Nominal success ≥ about 80% | Needs the HDF5 demo recorder enabled during policy runs (Arena attaches `ArenaEnvRecorderManagerCfg` only when cameras are on); whether `policy_runner.py` writes it is UNVERIFIED | Default when no oracle exists |
> | **B. Scripted policy** (pose targets + IK) | Structured task with known goal states | Not in Arena. Building blocks: `droid_differential_ik`, `PoseTargetMotionGenerator`, cuRobo IK (Docker only) | DROID or Franka pick-and-place; doubles as the reference controller |
> | **C. Teleop demos + Isaac Lab Mimic** | Annotated source demos | **Exists for GR1 tasks**: annotated and generated HDF5 on HF (`nvidia/Arena-GR1-Manipulation-*`), `generate_dataset.py` | GR1 pilot: the policy's own training data came from here |
> | **D. Small RL oracle** | Reward, state-based env | Arena has RL workflows (Franka lift) | Tasks where A-C fail |
>
> Notes:
> - Options B-D produce simulator-exact actions. Option A produces filtered behaviour cloning; do not call it expert ground truth.
> - Log per-frame simulator state with the observations and actions (Arena's `InitialStateRecorderCfg` and `PostStepStatesRecorderCfg` terms). Rung B (5.3) and the behaviour analysis need it.
> - Convert with Arena's HDF5→LeRobot converter. For GR1, keep `action_name_sim: processed_actions` (joint space), because the policy runs on `gr1_joint`, not on `gr1_pink` where Mimic runs.
> - Keep a held-out split. For GR1, Mimic-generated data is the closest match to the checkpoint's training distribution.

## 5.3 Data-cost ladder: replacement text

> ### 5.3 Data-cost ladder for appearance shifts
> Try the cheapest rung first. The router's restoring factor picks the starting rung (3.2.10).
>
> | Rung | What | Implementation | Cannot do |
> |---|---|---|---|
> | **A. On-the-fly augmentation** | Colour jitter and random rotation inside GR00T's training pipeline (albumentations `ReplayCompose`: resize, random crop, optional `Rotate`, optional `ColorJitter`) | `--color-jitter-params ...`, `--random-rotation-angle`. These are the only built-in image augmentations in N1.6. When fine-tuning from a checkpoint, a `None` value keeps the checkpoint's own setting | Glare, specular materials, camera geometry, blur (no built-in blur) |
> | **B. Isaac randomization** | New frames under the shifted factor, with exact actions | **GR1:** regenerate with Isaac Lab Mimic while Arena lighting variations are enabled (HDR, intensity, colour, colour temperature). **Optional:** kinematic state-replay re-render of logged rollouts, which must be built (Arena only re-simulates) | Material and texture changes (no Arena variation); camera shift on GR1 (no camera variation registered) |
> | **C. Cosmos Transfer** | Generative re-rendering of clean rollouts | Unchanged (stretch) | Anything that alters geometry or contact timing |
>
> **How Rung B gets frames.**
> - Arena does not re-render recorded episodes. Its replay tools re-simulate (action replay, or replay of variation values and placement), and Arena's docs warn that PhysX replay is not reproducible across resets. So:
>   1. **Mimic regeneration (default for GR1).** Mimic is closed-loop on object poses and ignores pixels, so the factor can change without changing its success. It keeps successful trials only and records exact actions. Its trajectories are new, not the failing seeds: report this.
>   2. **State-replay re-render (optional).** Rebuild the env with the new build-time lighting, reset to the logged initial state, write the logged per-step state with physics off, render, and drop episodes whose poses diverge. This needs Isaac Lab 3.0 state-write APIs (UNVERIFIED). Build it only if (1) is not available for the task.
> - Pair frames with the recorded actions, apply the quality gates (identity, geometry, contact timing), and keep a sample for human spot-check. For camera changes, the original actions stay valid only for appearance shifts.

## 5.4 Training recipe: replacement text

> ### 5.4 Training recipe (time-boxed default)
> - **Default (cheapest that is still official):**
>   - `launch_finetune.py` with `--no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model`. In N1.6, "projector" means the action head's state encoder, action encoder and decoder, not the vision-language projector. The VL projector `mlp1` is tied to `--tune-visual`.
>   - `tune_vlln=True` (action-head LayerNorm) is on by default and not on the CLI.
>   - `--color-jitter-params` as in Rung A.
>   - **Before the first run, read `tune_top_llm_layers` in the base checkpoint's `config.json`.** The fine-tune path does not override it. If it is 4 (the code default), the "frozen LLM" recipe trains the top 4 LLM layers. Record the trainable-parameter count the launcher logs, and say which it was in the report.
> - **Arena's documented recipe** for the GR1 checkpoints is `--no-tune-llm --tune-visual --tune-projector --tune-diffusion-model` (1× 24 GB, batch 16, 30k steps; or 8× 48 GB, batch 24, 20k steps). Use it as the **vision-side ablation**, because it is what the pilot checkpoint was trained with. Its result is also the fairest "fixed recipe" baseline (Section 7).
> - **LoRA is not core.** GR00T N1.6 has no LoRA option. The vendored Eagle backbone code has `use_backbone_lora` / `use_llm_lora` (vision or LLM LoRA, `alpha = 2r`). They are off and not wired to the fine-tune CLI. There is no LoRA for the action head. Use them only if VRAM forces it. Label the rank untuned and do not sweep it.
> - **Hardware:** ≥24 GB per GPU (Arena's low-hardware figure). Memory levers: `--global-batch-size`, `--num-shards-per-epoch`, `--shard-size`, `--dataloader-num-workers`. `gradient_checkpointing` and the DeepSpeed stage are only reachable through `launch_train.py`.
> - Record the hyperparameters, GPU, VRAM, trainable-parameter count and the final `final_model_config.json`.
> - Expect run-to-run variance. The GR00T README reports up to 5-6 points between identical runs, so compare fine-tunes with error bars, not single runs.

## 6.1 Failure map: replacement text

> ### 6.1 Pick a failure that is worth fixing (failure map)
> Good target: nominal success ≥ about 80%, shifted success about 10-40%, caused by distribution shift, cheap to regenerate.
>
> 1. **Factors come from Arena variations**, set to fixed values with Hydra overrides (`<asset>.<variation>.enabled=true` plus sampler `low == high`, or a one-element `hdr_names`). Run `--list_variations` on the pilot environment first; it is the authoritative factor list.
>    - On GR1 `put_item_in_fridge_and_close_door` (dome light only), expect HDR, intensity, colour, colour temperature, and object disappear for distractors.
>    - On DROID `pick_and_place_maple_table`, also expect directional-light direction and camera extrinsics and intrinsics.
>    - There is no material variation.
> 2. **Grid:** e.g. 4 intensities × 3 HDRs × 2 colour temperatures on the pilot task. Lighting variations are build-time, so each cell is one environment build. Budget build time per cell.
> 3. Run 10-20 episodes per cell with the pretrained policy in clean mode.
> 4. Keep cells where nominal is high and the shifted condition drops below about 40%.
> 5. Track per-subtask progress from Arena's progress events (lift = `object_is_above_height`, place = `object_on_destination`). Add reach and grasp tracked events (3.2.2, `docs/PLAN_UPDATE.md` (d)).
> 6. Build each cell as **nominal plus named factor changes**: `SceneSpec` factors map one-to-one onto Arena variation keys. Record the nominal success rate before any bug is injected.
> 7. Qualify the reference controller on every cell. For GR1 that is the Mimic trial success rate. Lighting must not change it; a cell where it drops is dropped.
> 8. **Measure replay agreement**: same build seed, same episode indices, policy RNG seeded, clean mode, twice. Arena documents that PhysX is not reproducible across resets, so expect agreement < 1 and plan unpaired statistics until measured.
> 9. Measure the clean vs incident nominal gap (3.1) on the pilot task.

## 6.4 Bug injection suite: replacement table and notes

> Prefer silent bugs. Each bug has one expected verdict. "Where" names the `policy-wrapper/` file (copied from `isaaclab_arena_gr00t`, see Section 8 step 3) or the server-side config the bug lives in.
>
> | Bug | Where (pilot = GR1 joint space) | Expected verdict | Notes |
> |---|---|---|---|
> | RGB/BGR swap | `utils/image_conversion.py` (`bgr_conversion` flag of `resize_frames_with_padding`) or `policy/gr00t_core.py` | Code | Natural injection point |
> | Image flip / rotate 180° | `policy/gr00t_core.py` (`resize_rgb_for_policy`) | Code | |
> | Hand (gripper) sign or threshold | `policy/gr00t_core.py` (`build_gr00t_action_np`) on the hand joints | Code | GR1 has 6-DoF hands, not a binary gripper; "never opens" becomes "hand joints inverted" |
> | State/action index mapping | `embodiments/gr1/*_joint_space.yaml` order, `utils/joints_conversion.py` | Code | One swapped YAML line is a realistic silent bug |
> | Chunk / replan mismatch | `policy/config/*closedloop_config.yaml` (`action_chunk_length`, `action_horizon`) | Code | |
> | Denoising steps | **server side**: `num_inference_timesteps` in the checkpoint config | Code | Lives outside the Arena client; needs the server launch config in `policy-wrapper/` (Section 8 step 3) |
> | Normalisation-stats mismatch | **server side**: checkpoint `statistics.json` / processor | Code | Same |
> | Action latency in controller code | new delay buffer in `policy/gr00t_remote_closedloop_policy.py` / scheduler | Code | Arena has no latency path today; the bug adds one |
> | Stale actions from a shared server | **injected async mode** in the wrapper, enabled only by the incident-mode launcher | Infra | Arena's client is synchronous (`ActionChunkScheduler` fetches with current observations), so load alone cannot make actions stale. The fault needs a deliberately added async or prefetch path that only the server load and config turns on. State this in the paper |
> | Camera-to-wrist frame transform | **does not apply** to a joint-space policy | — | Replace with a camera-mount offset in the embodiment or camera config (vision-looking code bug) |
> | Joint gain decay | actuator overrides applied by vla-isaac from `config/eval.yaml` | Code | Gains live in Arena embodiment code, which is outside `policy-wrapper/` |
> | Wrong instruction for one object | `language_instruction` in the closed-loop yaml / env builder | Code | |
> | Preprocessing path only at one resolution | `resize_rgb_for_policy` resizes only when the shape differs | Code | Natural |
> | Mild normalisation offset | server-side stats | Code, multi-cause | |
> | Tensor shape mismatch | any | Rung 0 | |

## 6.5 Router evaluation set: added rows and notes

> Changes to the table:
> - **Vision** cases come only from factors Arena can express on the pilot task: lighting (HDR, intensity, colour, colour temperature) and distractor presence. Add camera offset only if the pilot is DROID or Franka. Drop "glossy material" unless a material variation is built and checked visual-only.
> - **Sim / asset:** `ObjectMassVariation` on the target object (physics only) is a ready-made sim/asset case. The reference controller (Mimic) should drop with it, and lighting should not change it.
> - **Infra:** the injected async mode above (one case is enough), plus whatever the measured clean vs incident gap turns out to be. If that gap is real at baseline, it is an unplanned infra-like confounder and goes in the no-fault calibration.

## 8. Build order: replacement text

> ## 8. Build order (time-boxed)
> Real deadline: replace. The router, Rungs 0-2, the incident loop and hagent already run on mocks (vla-router, hybrid-agent). What remains is everything that touches a simulator. Steps 0-3 are the likeliest to overrun.
>
> | # | Step | Estimate |
> |---|---|---|
> | 0 | **Compute gate.** Get a remote GPU (≥24 GB), install Arena and GR00T, and run the VRAM and nominal-success measurements (`docs/PILOT_AND_ORDER.md`, NEEDS A RUN) | 3-5 days |
> | 1 | **Pilot confirmation.** Checkpoint, task, nominal ≥80% in clean mode, clean vs incident gap, `--list_variations`, training spec extracted from checkpoint and modality config | 1 week |
> | 2 | **`policy-wrapper/` and `vla-isaac` IsaacBackend v0**: `RolloutBackend.run` over Arena's policy runner, episode JSONL → `EpisodeResult`, `code_key` stamping, scene spec → variation overrides | 1-2 weeks |
> | 3 | **Instrumentation**: wrapper logger (policy-input stats, step pairs, raw actions), HDF5 states → `EpisodeTrace` / `BehaviorTrace`, reach and grasp tracked events, `TraceSource` / `BehaviorSource` implementations | 1-2 weeks |
> | 4 | **Failure map + reference qualification (Mimic) + replay agreement** (6.1) | 1-2 weeks |
> | 5 | **Pilot gate** (6.2): Rung A, then B (Mimic regeneration), default recipe, held-out | 2 weeks |
> | 6 | **Router on real episodes**: calibration on no-fault incidents, then injected bugs (6.4) | 2 weeks |
> | 7 | **Loop end to end**: hagent on `policy-wrapper/` in Docker, Rung 1 on IsaacBackend, Rung 2 | 1-2 weeks |
> | 8 | Evaluation, baselines, write-up | 2 weeks |
>
> Cut rule: if step 0 or 1 slips by more than a week, shrink the bug suite to client-side bugs only (drop the server-side denoising and normalisation bugs and the injected infra mode), and run the router with G3 disabled (decision rows that need G3 stay undecided).

## 9. Risks: lines to add

> - **Compute:** the development GPU (RTX 3050 Laptop, 6 GB) cannot run the pilot. A remote GPU is a hard dependency from step 0.
> - **`tune_top_llm_layers` is inherited from the checkpoint** (code default 4). The "frozen backbone" default recipe may train 4 LLM layers. Read the checkpoint config before the first fine-tune.
> - **PhysX replay non-determinism is documented by Arena.** Action replay and replay agreement may be low. State-replay re-render is not available off the shelf.
> - **Arena re-simulates, it does not re-render.** Rung B on GR1 uses Mimic regeneration (new trajectories); the state-replay re-render is new code on unread Isaac Lab APIs.
> - **No material variation and no GR1 camera variation.** The vision factor set on the GR1 pilot is lighting and distractors only.
> - **Infra row realism.** The Arena GR00T client is synchronous, so the stale-server fault must be injected through an added async mode. Arena's own docs also show a nominal drop from one env to ten (0.88 → 0.605 on open-microwave) with no stated cause.
> - **Server ignores `--modality-config-path` when `--model-path` is given.** The modality config used at inference is the checkpoint's. The training spec must be read from the checkpoint, not from Arena's `*_data_config.py`.
> - **README/code mismatch on the backbone** ("Cosmos-Reason-2B variant" vs `nvidia/Eagle-Block2A-2B-v2`). Cite the code.
> - **GR1 composite task vs `PHASES`.** The close-door subtask has no slot in `BehaviorTrace.PHASES`. Extending it touches vla-router.
> - **Doc inconsistencies in Arena results.** The sequential-task evaluation page shows ranch bottle 0.9 and jug 0.1 in its metrics table, but its text says 0.8 and 0.0 (`A:docs/pages/example_workflows/sequential_static_manipulation/step_5_evaluation.rst:356-373`). Use only our own measurements.
