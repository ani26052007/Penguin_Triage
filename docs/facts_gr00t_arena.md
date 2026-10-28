# Facts: Isaac-GR00T and IsaacLab-Arena (as checked out in `external/`)

Method: read-only code and doc inspection. All paths are relative to `external/`. "NOT FOUND" means I searched and found nothing.
The external-chat claims were not pasted into this task. The "contradicts" and "NOT FOUND" lists at the end are therefore claims that are commonly assumed, checked against code. Please diff them against the actual chat.

## Checkout caveat (read first)
- `IsaacLab-Arena/submodules/Isaac-GR00T` is **not checked out**. `git submodule status` shows `-e29d8fc50b0e4745...`, and the directory is empty. Arena pins a GR00T commit that is not the `external/Isaac-GR00T` tree we have. Declared at `IsaacLab-Arena/.gitmodules` (submodule entries) and copied into the image at `IsaacLab-Arena/docker/Dockerfile.isaaclab_arena:44`.
- **Version skew.** `external/Isaac-GR00T` is **N1.7** (`Isaac-GR00T/README.md:50`). Arena's docs, finetune workflow and the DROID config describe **N1.6**:
  - `IsaacLab-Arena/docs/pages/quickstart/running_a_real_policy/gr00t.rst:4`, `:34` (`nvidia/GR00T-N1.6-DROID`)
  - `IsaacLab-Arena/osmo/finetune.yaml:61` (`--base-model-path=/workspace/pretrained_ckpts/GR00T-N1.6-3B`)
  - Arena does contain one N1.7 modality file: `IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py`.
- **Arena embodiment tag strings do not match N1.7 enum names.** Arena YAMLs use `GR1` and `OXE_DROID` (`IsaacLab-Arena/isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml:10`, `droid_manip_gr00t_closedloop_config.yaml:8`), and `NEW_EMBODIMENT` for G1 (`g1_locomanip_gr00t_closedloop_config.yaml:10`). The N1.7 `EmbodimentTag` enum has no `GR1` and no `OXE_DROID`. It has `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` (`Isaac-GR00T/gr00t/data/embodiment_tags.py:~70`, listed in the docstring at `:27-34`). The Arena strings are used only on the Arena side, as keys into its joint-remapping code (`IsaacLab-Arena/isaaclab_arena_gr00t/utils/joints_conversion.py:25-47`, which does `embodiment_tag.upper()` and compares to `"GR1"`, `"NEW_EMBODIMENT"`, `"OXE_DROID"`). Whether the client passes the tag to the server is not visible there. The server picks its tag via `--embodiment-tag` (`Isaac-GR00T/gr00t/eval/run_gr00t_server.py:61`).

---

# Isaac-GR00T

## 1. Model version, backbone, action head
- Version: **GR00T N1.7**, "General Availability" (`Isaac-GR00T/README.md:50-52`). Package `gr00t/model/gr00t_n1d7/`. The model config class is `Gr00tN1d7Config`, `model_type="Gr00tN1d7"` (`gr00t/configs/model/gr00t_n1d7.py:28,36`). `pyproject.toml:9` says `version = "0.1.0"`, which is the package version, not the model version.
- Backbone: `nvidia/Cosmos-Reason2-2B` (Qwen3-VL architecture) (`gr00t/configs/model/gr00t_n1d7.py:40`; `README.md:50,101,117`). Implemented by `Qwen3Backbone` (`gr00t/model/modules/qwen3_backbone.py:135`). It is a gated HF model that every checkpoint loads (`README.md:183`).
  - Vision encoder is `self.model.visual`, LLM is `self.model.language_model` (`qwen3_backbone.py:255-256,260`).
  - `select_layer=12`, `backbone_embedding_dim=2048` (`gr00t/configs/model/gr00t_n1d7.py:44,47`).
- Action head: `Gr00tN1d7ActionHead` (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:38`). It is a flow-matching DiT, `AlternateVLDiT` when `use_alternate_vl_dit=True` (default) (`gr00t_n1d7.py:50`; config `:88`). `diffusion_model_cfg`: 16 layers, 32 heads, head dim 48 (`gr00t/configs/model/gr00t_n1d7.py:91-103`; `README.md:119`, "32 to 16 diffusion layers").
- Full model class: `Gr00tN1d7(PreTrainedModel)` (`gr00t_n1d7.py:500`).
- Dims: `max_state_dim=132`, `max_action_dim=132`, `action_horizon=40`, `hidden_size=1024`, `input_embedding_dim=1536` (`gr00t/configs/model/gr00t_n1d7.py:74-78`). `max_num_embodiments=32` (`:123`).
- Other defaults: `model_dtype="bfloat16"` (`:37`), `use_relative_action=False` in the config (`:71`) but forced `True` by the finetune launcher (`gr00t/experiment/launch_finetune.py:102`).

## 2. Fine-tuning entry and tune flags
- Entry script: `gr00t/experiment/launch_finetune.py` (tyro CLI on `FinetuneConfig`, `launch_finetune.py:49`). The wrapper `examples/finetune.sh` runs it via `python` for 1 GPU or `torchrun` for more (`finetune.sh:143-162,217-224`). `CLAUDE.md` also lists `bash examples/finetune.sh ...` as the fine-tune entry.
- Wrapper hyperparameters (`examples/finetune.sh:143-162`): `--save_total_limit 5`, `--warmup_ratio 0.05`, `--weight_decay 1e-5`, `--learning_rate 1e-4`. Env defaults: `NUM_GPUS=1`, `SAVE_STEPS=1000`, `MAX_STEPS=10000`, `GLOBAL_BATCH_SIZE=32`, `DATALOADER_NUM_WORKERS=4`, `EPISODE_SAMPLING_RATE=0.1` (`:5-14`). Note that the wrapper's batch size (32) differs from the dataclass default (64).

Tune flags exposed on the CLI (`gr00t/configs/finetune_config.py`):
| flag | default | line |
|---|---|---|
| `tune_llm` | False | 49 |
| `tune_visual` | False | 52 |
| `tune_projector` | True | 55 |
| `tune_diffusion_model` | True | 58 |
| `state_dropout_prob` | 0.2 | 61 |

- Applied at `launch_finetune.py:78-82`.
- `tune_top_llm_layers`: **exists only in the model config** (default **0**, `gr00t/configs/model/gr00t_n1d7.py:43`). It is **not in `FinetuneConfig`** and is **not set in `launch_finetune.py`**, so the finetune CLI cannot set it. It could only be reached by editing the code or by a config load path. Implementation: `qwen3_backbone.py:259-262` unfreezes the top N LLM layers even when `tune_llm=False`. `README.md:119` confirms the default changed from 4 to 0.
- `tune_vlln` (action-head vision-language layer norm) default True (`gr00t/configs/model/gr00t_n1d7.py:115`). Not exposed on the CLI.
- Other constants forced by the launcher (`launch_finetune.py:98-102`): `load_bf16=False`, `reproject_vision=False`, `model_name="nvidia/Cosmos-Reason2-2B"`, `backbone_trainable_params_fp32=True`, `use_relative_action=True`.
- Training hyperparameters (`finetune_config.py:124-197`): `global_batch_size=64`, `gradient_accumulation_steps=1`, `learning_rate=1e-4`, `max_steps=10000`, `weight_decay=1e-5`, `warmup_ratio=0.05`, `save_steps=1000`, `shard_size=2**10`, `episode_sampling_rate=0.1`, `num_shards_per_epoch=1e5`. Also `ds_weights_alpha` (dataset mixing, `:172`) and `resume_from_checkpoint` (`:188`). Optimizer is `adamw_torch` (`launch_finetune.py:106`).
- **LoRA: not available in this version's code.** `peft==0.17.1` appears only as a dependency (`pyproject.toml:21`, `scripts/deployment/jetson/pyproject.toml:26`, `scripts/deployment/spark/pyproject.toml:23`). A grep for `lora|peft` in all `*.py`, `*.md` and `*.sh` found no code use and no config option. There are no LoRA config options (rank, alpha, target modules).
- VRAM effect of flags: the docs say the default tunes only the projector and diffusion head, with peak VRAM under about 35 GB/GPU. Enabling `--tune-llm` or `--tune-visual` increases VRAM significantly (80 GB+ recommended) (`getting_started/hardware_recommendation.md:53-54`).

## 3. Built-in image augmentation
- Builder: `build_image_transformations_albumentations` (`gr00t/model/gr00t_n1d7/image_augmentations.py:362`). Train pipeline (inside `ReplayCompose`, so the same augmentation is applied across views):
  - `SmallestMaxSize` → `FractionalRandomCrop` → `SmallestMaxSize` (`:422-425`)
  - optional `A.Rotate(limit=random_rotation_angle, p=1.0)` (`:428-429`)
  - optional `A.ColorJitter(brightness, contrast, saturation, hue, p=1.0)`, only when `color_jitter_params` is not None (`:431-442`)
- Eval pipeline is deterministic, with a center crop (`:480-490`).
- Mask-based extras, via `extra_augmentation_config`: `background_noise_transforms` (`BackgroundNoiseTransform`, mask value default `[0]`) and `masked_region_transforms` (`MaskedColorTransform`, `alpha_range` default `[0.3, 1.0]`, `p` default 0.5). These need mask data (`image_augmentations.py:451-474`; schema in `finetune_config.py:104-121`).
- Other options: `LetterBoxPad` (`image_augmentations.py:324`), `letter_box_transform=False` (`gr00t/configs/model/gr00t_n1d7.py:63`).
- Defaults:
  - Model config: `image_crop_size=(230,230)`, `image_target_size=(256,256)`, `random_rotation_angle=None`, `color_jitter_params=None`, `use_albumentations_transforms=True` (`gr00t/configs/model/gr00t_n1d7.py:54-62`).
  - `FinetuneConfig.random_rotation_angle=None`, `color_jitter_params=None` (`finetune_config.py:67,70`). Its docstring says "If None, applying the default color jitter augmentation from the pretrained model" (`:81`). The code does not say what that pretrained default is, so the actual value is **NOT FOUND** in code (it would be in the checkpoint's processor config).
  - `examples/finetune.sh:25` supplies `brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08` by default, and passes it on at `:177-180`. So jitter is on by default only when you use the wrapper script. `osmo/finetune.yaml` in Arena passes the same values.
  - Crop fraction: `shortest_image_edge` and `crop_fraction` must be set together (`launch_finetune.py:86-87`). The legacy fraction is `230/256` (`image_augmentations.py:402`).
- No built-in lighting, texture or blur augmentation beyond the above. Searching `image_augmentations.py` for other transforms found none.

## 4. Training data format
- **LeRobot v2** (with GR00T extensions). v3 must be converted with `scripts/lerobot_conversion/convert_v3_to_v2.py` (`getting_started/data_preparation.md:5,13,15`).
- Layout (`data_preparation.md:20-40`): `meta/{episodes.jsonl, modality.json, info.json, tasks.jsonl}`, `videos/chunk-000/observation.images.<name>/episode_XXXXXX.mp4`, `data/chunk-000/episode_XXXXXX.parquet`. GR00T also computes `meta/stats.json` and `meta/relative_stats.json` (`data_preparation.md`, "Extensions" section).
- Parquet columns (`data_preparation.md:~60-75`): `observation.state` (1D concatenated), `action` (1D concatenated), `timestamp`, `annotation.<source>.<type>[.<name>]` (index into `tasks.jsonl`), `task_index`, `episode_index`, `index`, `next.reward`, `next.done`. State must be in `observation.state`.
- `meta/modality.json` schema (`data_preparation.md`, "Schema" section):
  - `state.<key>: {start, end}` and `action.<key>: {start, end}`, zero-based `[start:end]`
  - `video.<new_key>: {original_key}`
  - `annotation.<key>: {}` (omit the `annotation` block if there are no annotations)
- The annotation key must agree in three places: parquet column, `modality.json` key (without the `annotation.` prefix), and `modality_keys` in the modality config (`data_preparation.md`, "Annotation Column Naming" table).
- The Python-side modality config (delta indices, keys, `ActionConfig`) is registered with `register_modality_config(...)` (`gr00t/configs/data/embodiment_configs.py:355-361`). A custom robot must be registered under `EmbodimentTag.NEW_EMBODIMENT` (`getting_started/finetune_new_embodiment.md:71-72`) and passed with `--modality-config-path` (`finetune_new_embodiment.md:121`).
- Converters in Isaac-GR00T: only `scripts/lerobot_conversion/convert_v3_to_v2.py`. **No Isaac Lab or HDF5 converter in Isaac-GR00T** (grep for `hdf5|h5py|isaaclab` in `*.py`/`*.md` hit only `tests/test_support/compass.py` and `examples/RoboLab/README.md`; not read in detail). The HDF5→LeRobot converter lives in Arena (see item 11).

## 5. Embodiment tags (`gr00t/data/embodiment_tags.py`)
- Pretrained (in the base model `nvidia/GR00T-N1.7-3B`; `PRETRAIN_TAGS` at `:192`):
  - `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`
  - `XDOF`
  - `XDOF_SUBTASK`
  - `REAL_G1`
  - `REAL_R1_PRO_SHARPA`
  - `REAL_R1_PRO_SHARPA_HUMAN`
  - `REAL_R1_PRO_SHARPA_MAXINSIGHTS`
  - `REAL_R1_PRO_SHARPA_MECKA`
- Pre-registered posttrain (**require a finetuned checkpoint**; `POSTTRAIN_TAGS` at `:206`): `UNITREE_G1` (`:102`), `UNITREE_G1_SONIC`, `SIMPLER_ENV_GOOGLE`, `SIMPLER_ENV_WIDOWX`, `LIBERO_PANDA` (`:122`).
- Finetune-only (`FINETUNE_ONLY_TAGS` at `:217`): `NEW_EMBODIMENT` (`:128`), `ROBOCASA_GR1_TABLETOP` (`:133`), `ROBOCASA_PANDA_OMRON` (`:139`).
- The policy rejects a tag that is not in the checkpoint and prints a hint (`gr00t/policy/gr00t_policy.py:120-163`, approx).
- Pre-registered modality configs exist for `oxe_droid_relative_eef_relative_joint`, `unitree_g1_sonic`, `unitree_g1_full_body_with_waist_height_nav_cmd`, `libero_sim`, `simpler_env_widowx`, `simpler_env_google`, `robocasa_panda_omron` and `robocasa_gr1_tabletop` (`gr00t/configs/data/embodiment_configs.py:28,67,115,193,211,229,247,288`). The pretrained real-robot tags carry their configs in the checkpoint's processor config (`embodiment_tags.py:27-28` docstring).
- Action chunk lengths in those configs: `range(40)` for the first two cases I read (`embodiment_configs.py:38,86`) and `range(50)` at `:133`. I did not check the rest.

## 6. Inference
- Server: `gr00t/eval/run_gr00t_server.py`. `ServerConfig` fields (`:53-87`): `model_path`, `embodiment_tag` (default `"new_embodiment"`), `device="cuda"`, `dataset_path`, `modality_config_path`, `execution_horizon` (only required for the replay policy), `host="0.0.0.0"`, `port=DEFAULT_MODEL_SERVER_PORT`, `strict=True`, `use_sim_policy_wrapper=False`.
- Transport: ZeroMQ **REP** socket, one message at a time (`gr00t/policy/server_client.py:167-169`). Client is a **REQ** socket with `timeout_ms=15000` default (`:286`, sockopts at `:301-304`). Message serialization is msgpack plus msgpack_numpy (`:22-24,33`). Endpoints: `ping`, `kill`, `get_action`, `reset`, `get_modality_config` (`:172-180`). Optional API token. The server loop is a single-threaded `recv → handle → send` (`:241-270`), so requests are served serially.
- Batching: `Gr00tPolicy.get_action` takes a batched observation of shape `(B, T, H, W, C)` for video (uint8) (`gr00t_policy.py:281-292`). Internally it unbatches into B single observations, processes each, then re-collates (`:177-198,400-412`). Model inference runs in bfloat16 (`:413`).
- Action chunk length: `action_horizon=40` in the model config (`gr00t/configs/model/gr00t_n1d7.py:76`). The per-embodiment horizon is defined by the `delta_indices` of the `action` modality (e.g. `list(range(40))` at `embodiment_configs.py:38`). The execution contract requires `action_delta == range(0, action_horizon)` (`gr00t/eval/_horizon_contract.py:142-155`).
- Replan steps: `n_action_steps` is the open-loop execution length, valid in `[1, action_horizon]` (`_horizon_contract.py:80-96,157-165`). The sim rollout script defaults it to **8** (`gr00t/eval/rollout_policy.py:546,649`). The server flag `--execution-horizon` is used for the replay policy (`run_gr00t_server.py:74`). The old flag `--action-horizon` was renamed (`README.md:121`; `_horizon_contract.py:33`).
- Denoising steps: `num_inference_timesteps=4` (`gr00t/configs/model/gr00t_n1d7.py:106`), used in the flow-matching loop (`gr00t_n1d7.py:355,397`). It is a model-config value. I found no inference CLI flag for it (only model-config and export-script reads).
- Measured rates in the docs: for example H100 11.7 Hz eager and 35.9 Hz with TensorRT (`getting_started/hardware_recommendation.md:15-25`). `README.md` states ~40 Hz on RTX PRO 6000 and ~28 Hz on Thor (TensorRT).

## 7. GPU/VRAM
- Inference: 1 GPU with 16 GB+ VRAM, CUDA 12.6+ (`README.md:134`; `hardware_recommendation.md:11`).
- Fine-tuning: 1 or more GPUs with 40 GB+ recommended (`README.md:136`; `hardware_recommendation.md:43`). Quick-start is 1× H100/L40/A100 at 40–80 GB with global batch 32 (`hardware_recommendation.md:47`). The default tune set (projector + diffusion head) stays under ~35 GB peak per GPU (`:53`). Enabling `--tune-llm` or `--tune-visual` needs 80 GB+ (`:54`). `--num-shards-per-epoch` controls host RAM (`:56`).
- ~3B parameters, bfloat16 (`hardware_recommendation.md:43`).

---

# IsaacLab-Arena

## 8. GR00T-compatible tasks/environments and embodiments
In-repo GR00T policy configs (`IsaacLab-Arena/isaaclab_arena_gr00t/policy/config/`):
- GR1 tabletop, "open the microwave" (`gr1_manip_gr00t_closedloop_config.yaml:6-10,15`): `action_horizon: 16`, `action_chunk_length: 16`, `task_mode_name: gr1_tabletop_manipulation`, modality config `embodiments/gr1/gr1_arms_only_data_config.py`, `pov_cam_name_sim: robot_pov_cam_rgb`, image 512×512×3.
- GR1 ranch bottle into fridge: `gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml`. `osmo/finetune.yaml:117-119` uses `embodiment_tag: GR1` and `gr1_ranch_bottle_into_fridge`.
- G1 loco-manipulation, `NEW_EMBODIMENT`, `action_horizon: 50`, `action_chunk_length: 50` (`g1_locomanip_gr00t_closedloop_config.yaml:9-10,19`).
- DROID, `OXE_DROID`, `action_horizon: 32`, `action_chunk_length: 32` (`droid_manip_gr00t_closedloop_config.yaml:7-8,18`).

Environments (`IsaacLab-Arena/isaaclab_arena_environments/`) referenced by GR00T flows: `gr1_open_microwave_environment.py`, `gr1_put_and_close_door_environment.py` (name `put_item_in_fridge_and_close_door`, `:51`), `gr1_turn_stand_mixer_knob_environment.py`, `gr1_table_multi_object_no_collision_environment.py`, `galileo_g1_locomanip_pick_and_place_environment.py`, and the DROID ones (`droid_table_multi_object_placement_environment.py`, `droid_deformable_pick_and_place_environment.py:35`). Experiment configs that use `Gr00tRemoteClosedloopPolicy`: `experiment_configs/droid_pnp_gr00t_experiment.yaml:18`, `droid_pnp_srl_gr00t_experiment.yaml`, and `eval_jobs_configs/droid_pnp_srl_gr00t_jobs_config.json`. Which tasks each GR00T config is documented to pair with is in the workflow docs (`docs/pages/example_workflows/{static_manipulation,sequential_static_manipulation,locomanipulation}`, all N1.6). I did not verify that every GR1 environment listed above has a trained checkpoint.

## 9. What `isaaclab_arena_gr00t` does (candidates for `policy-wrapper/`)
- `policy/gr00t_remote_closedloop_policy.py` is the remote policy (registered as `gr00t_remote_closedloop`, `:75-81`). It uses GR00T's own `PolicyClient` with `strict=False` (`:126-132`) and does a `ping` check at startup (`:133`). Config `Gr00tRemoteClosedloopPolicyCfg` has `num_envs`, `remote_host="localhost"`, `remote_port=5555`, `remote_api_token`, `scheduler` (`:51-72`).
- **Policy-input preprocessing** (obs): `get_action` → `_get_action_chunk` (`:151,173`) calls
  - `extract_obs_numpy_from_torch` (`policy/gr00t_core.py:162`)
  - `build_gr00t_policy_observations` (`gr00t_core.py:227`): resize with padding to `target_image_size` (`resize_rgb_for_policy`, `:201`), sim→policy joint remap (`utils/joints_conversion.py`), and construction of the nested `{"language","video","state"}` dict. The language entry is `[[task]]*num_envs` and video is `(N,1,H,W,C)`.
- **Action post-processing**: `build_gr00t_action_np` / `build_gr00t_action_tensor` (`gr00t_core.py:298,343`): policy→sim joint remap, and for G1 locomanip concatenation of `navigate_command`, `base_height_command` and a **zeroed torso RPY** (`:333-336`).
- **Chunk handling / schedulers**: `isaaclab_arena/policy/action_scheduling/action_chunk_scheduler.py` and `synced_batch_action_scheduler.py` (see item 12).
- The YAML-driven config (`policy/config/gr00t_closedloop_policy_config.py`) holds `action_horizon`, `action_chunk_length`, joint-config paths, camera name and image sizes. Embodiment-specific modality files are in `embodiments/{gr1,g1,droid}`.
- `utils/wait_for_gr00t_server.py` waits for server readiness. `lerobot/convert_hdf5_to_lerobot.py` is the data converter (item 11).
- The same adapter pattern for other servers (an `EmbodimentAdapter` plus `RemoteChunkReplayPolicy`) is in `isaaclab_arena/policy/remote_policy_base.py:17-60`.

## 10. Other packages
- `isaaclab_arena_curobo`: build-time **IK reachability validation** for object placement. `CuroboIKSolver` (`ik_solver.py:26`) is a sim-free cuRobo IK feasibility oracle. `ReachabilityValidator` (`ik_reachability_validator.py:67`) is registered as the `ik_reachable` check and rejects layouts where the robot cannot reach a top-down grasp at each movable object (module docstring, `:1-8`; `__init__.py` docstring). It is not a motion planner and not part of policy inference.
- `isaaclab_arena_cosmos`: `CosmosRemotePolicy` (`policy/cosmos_remote_policy.py`, name `cosmos_remote`): a remote closed-loop policy client using the chunk-replay mixin, with a DROID adapter (`droid_adapter.py`).
- `isaaclab_arena_dreamzero`: `DreamZeroRemotePolicy` (`policy/dreamzero_remote_policy.py`): a **websocket (msgpack)** client for a DreamZero server, with a reconnect budget (`RECONNECT_WAIT_S = 120`, `MAX_RECONNECT_ATTEMPTS`), a DROID adapter (`droid_adapter.py`; second-exterior-camera source `black|right|duplicate`), `image_utils.py` (resize with pad), and `docker/dreamzero_inference_server.yaml`.
- Also present and relevant: `isaaclab_arena_openpi` (Pi0 remote policy), `isaaclab_arena_vlm_agent` (VLM agent policy, with `motion_generator.py` and observation/action adapters), `isaaclab_arena_g1` (G1 whole-body controller, teleop, recorders).

## 11. Built-in tools
- **Subtask progress tracking**: `isaaclab_arena/progress_tracking/` (`progress_tracker.py`: `ProgressTracker` `:476`, `CompletionCriteriaRunner` `:140`, `ProgressTrackingRecorder` `:783`; `task_progress_cfg.py:13`). `TaskProgressCfg` has `success_criteria`, `subtasks_are_sequential`, `desired_subtask_success_state`, `tracked_events` (`task_progress_cfg.py:16-28`). Per-episode results are recorded via `recording/progress_terms.py:18` (`record_progress_results`) and `recording/common_terms.py:18`.
- **Perturbation / variation sweeps** (`isaaclab_arena/variations/`):
  - lighting: `LightIntensityVariation`, `LightColorVariation`, `LightColorTemperatureVariation`, `LightDirectionVariation`, and `HDRImageVariation` (background HDR)
  - camera: `CameraExtrinsicsVariation` (translation offset in the optical frame, `camera_extrinsics_variation.py`) and `CameraIntrinsicsVariation`
  - objects: `ObjectMassVariation`, `ObjectDisappearVariation`
  - samplers: `uniform_sampler.py`, `choice_sampler.py`, `bernoulli_sampler.py`
  - Build-time vs run-time: `BuildTimeVariationBase` / `RunTimeVariationBase` (`variation_base.py:169,182`). Hydra config in `variations_hydra.py`. Samples are recorded and replayable (`variation_recorder.py`, `recorded_variation_replay.py`, `variation_replay_scheduler.py`). An example config is `experiment_configs/droid_pnp_variations_experiment.yaml`; camera sensitivity example `droid_pnp_camera_sensitivity_openpi_experiment.yaml`.
  - **Material/texture variation: NOT FOUND** as a variation class (grep for material/texture in `variations/` found no file).
- **Sensitivity analysis**: `isaaclab_arena/analysis/sensitivity/` (`analyzer.py:16`: `SensitivityAnalyzer` fits a neural posterior with `sbi` MNPE/NPE over factors vs outcomes; `dataset.py`, `episode_results_reader.py`, `plotting.py`, `generate_report.py`). Docs: `docs/pages/concepts/concept_sensitivity_analysis.rst`.
- **Data recording**: `isaaclab_arena/recording/` (`EpisodeRecorderManager`, `ArenaRecorderManager`, episode results as JSONL). Teleop demos are recorded to HDF5 via the documented workflow (`docs/pages/example_workflows/static_manipulation/step_2_teleoperation.rst:68,172-173`).
- **Mimic**: Arena has Mimic envs (`--mimic` CLI flag, `isaaclab_arena/cli/isaaclab_arena_cli.py:56`; `GR1T2MimicEnv`, `embodiments/gr1t2/gr1t2.py:104,413`; a sequential-task mimic test `tests/test_sequential_task_mimic_data_generation.py`). The documented workflow annotates and generates demos (`step_3_data_generation.rst:45-58`, `step_1_environment_setup.rst:142-159`).
- **HDF5 → LeRobot**: `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py` (619 lines; uses `h5py`, `Gr00tDatasetConfig`, writes parquet and mp4 and `modality.json`-style metadata). Test: `isaaclab_arena_gr00t/tests/test_convert_hdf5_to_lerobot.py`. It targets N1.6-era datasets per the docs (`step_4_policy_training.rst:49`). I did not verify that its output is accepted by N1.7.

## 12. How parallel evaluation talks to the policy server
- One `PolicyClient` (ZMQ REQ, **blocking**, 15 s timeout by default) per policy instance. It sends one `get_action` request carrying all `num_envs` observations as a batch dimension (`gr00t_core.py:227-296`, `gr00t_remote_closedloop_policy.py:173-196`). The server (REP) processes one request at a time (`gr00t/policy/server_client.py:241-270`) and unbatches per-observation internally (`gr00t_policy.py:400-412`).
- `get_action` is **synchronous** inside the sim step: the environment waits for inference (`gr00t_remote_closedloop_policy.py:151-160`). No async prefetch or overlap was found, so no stale actions arise from latency (the sim is paused), but throughput is limited by the server round trip.
- Chunk scheduling (`ActionChunkScheduler`, `action_chunk_scheduler.py:16-100`): each env keeps its own index into a buffered chunk. A fetch happens whenever **any** env needs a new chunk (`:69`), but the returned tensor is only copied for the masked envs (`:71-74`). The server still computes the full batch, so inference for non-needing envs is wasted. Counters `_n_fetch_calls`, `_total_envs_needed` exist for this (`:48-52`). Each env consumes only `action_chunk_length` steps (`:81-98`). The rest of the `action_horizon` chunk is discarded.
- `SyncedBatchActionScheduler` (`synced_batch_action_scheduler.py:16-64`) waits until **all** envs need a chunk, then does one full-batch call. Waiting envs get `hold_action` (their current joint positions, `gr00t_remote_closedloop_policy.py:162`) for up to `action_chunk_length - 1` steps.
- On env reset the client calls `reset` on the server and resets the scheduler (`gr00t_remote_closedloop_policy.py:213-219`). Re-fetching after a mid-chunk reset (stale chunk) is handled by `env_requires_new_chunk[env_ids] = True` and zeroing the buffer (`action_chunk_scheduler.py:102-108`).
- Choice of scheduler: `ActionSchedulerType.CHUNK` (default) or `SYNCED_BATCH` (`gr00t_remote_closedloop_policy.py:35-48,70`).
- Other remote policies use a websocket protocol with reconnect (`remote_policy_base.py`, `websocket_client.py`, `dreamzero_remote_policy.py`). The GR00T path has no reconnect logic: a timeout leaves the REQ socket invalid (comment at `server_client.py:~341`, "REQ socket is now in an invalid state").

---

# Claims from the external chat that the code CONTRADICTS
(I did not have the chat text. These are common assumptions the code refutes.)
1. "LoRA is available / has these config options." In this checkout there is no LoRA code or config. `peft` is only a pinned dependency (`pyproject.toml:21`).
2. "Fine-tune the top-N LLM layers via a flag." `tune_top_llm_layers` is not on the finetune CLI. It defaults to 0, and the launcher never sets it (`launch_finetune.py:78-82`).
3. "Arena targets/pins the GR00T in `external/Isaac-GR00T`." Arena's own submodule is uninitialised, and its docs, finetune workflow and DROID config are for N1.6 (see the checkout caveat). Arena's `GR1`/`OXE_DROID` tag names do not exist in the N1.7 enum.
4. "Default fine-tune augmentation includes colour jitter." Only via `examples/finetune.sh` (`:25`). Calling `launch_finetune.py` directly leaves `color_jitter_params=None`, which `FinetuneConfig` documents as "default from the pretrained model" (`finetune_config.py:70,81`). The model config default is also `None` (`gr00t/configs/model/gr00t_n1d7.py:61`).
5. "Action horizon is 16" (N1.5/N1.6-era). N1.7 uses 40 (`gr00t/configs/model/gr00t_n1d7.py:76`; `README.md:120`). Arena configs use 16 (GR1), 32 (DROID) and 50 (G1), set in the modality config and not necessarily the N1.7 value.
6. "The state dropout default is 0.8 (or 0.2)." Both appear: the model config default is 0.8 (`gr00t/configs/model/gr00t_n1d7.py:118`) and the finetune CLI default 0.2 (`finetune_config.py:61`) overrides it (`launch_finetune.py:82`).
7. "Posttrain tags such as `UNITREE_G1` and `LIBERO_PANDA` work with the base model." They require a finetuned checkpoint (`embodiment_tags.py:206`; hint at `gr00t_policy.py:~143`).
8. "Parallel eval runs asynchronously against the server / has stale-action risk from latency." The client is synchronous and blocking (item 12). Stale chunks are an issue only across resets, and the scheduler zeroes them.

# Claims NOT FOUND
- Any LoRA parameter (rank, alpha, target modules).
- An Isaac Lab/HDF5 converter inside Isaac-GR00T (it is in Arena).
- A CLI flag for denoising steps (`num_inference_timesteps`) at inference.
- The numeric default colour jitter of the pretrained processor (it lives in checkpoint config, not in code).
- A material/texture variation class in Arena's `variations/`.
- Async inference, or action-chunk blending or temporal ensembling on the GR00T path in Arena (the schedulers only replay the first `action_chunk_length` actions).
- A documented VRAM figure for inference at batch > 1 or for many parallel envs.
- Documentation that Arena's HDF5→LeRobot output works with N1.7. Only a N1.7 G1 modality file exists (`g1_sim_wbc_data_gr00t_n_1_7_config.py`).
- Any N1.7-specific Arena workflow doc (all workflow docs say N1.6).
