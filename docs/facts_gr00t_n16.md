# Facts: the pinned Isaac-GR00T (N1.6) under IsaacLab-Arena

Written 2026-10-27. Read-only inspection, plus one `nvidia-smi` call. Nothing was installed, downloaded or run.

Path prefixes used in citations:

| Prefix | Path (relative to the repo root) |
|---|---|
| `G:` | `external/IsaacLab-Arena/submodules/Isaac-GR00T` (detached at `e29d8fc`) |
| `A:` | `external/IsaacLab-Arena` (HEAD `7632eb7d5e`) |
| `R:` | `vla-router` |
| `OLD:` | `docs/facts_gr00t_arena.md` (the N1.7 record; that clone is gone) |

Labels: **UNVERIFIED** means I found no file that settles it. **NEEDS A RUN** gives the command that would settle it. "NOT FOUND" means I searched and found nothing.

---

## 0. Checkout and machine

### Facts
- **GR00T pin.** `git submodule status` in `A:` prints `e29d8fc50b0e4745120ae3fb72447986fe638aa6 submodules/Isaac-GR00T (n1.5-release-11-ge29d8fc)`. The commit is dated 2026-02-03, "Update README.md (#531)". `git branch -r --contains HEAD` lists `origin/n1d6`, `origin/gr00t_1d6`, `origin/main` and `origin/release`. So the commit is in `origin/n1d6`, as stated.
- **Arena documents this exact pin.** `A:docs/pages/quickstart/running_a_real_policy/gr00t.rst:13-19` and `A:docs/pages/example_workflows/static_manipulation/step_5_evaluation.rst:41-45` link `tree/e29d8fc50b0e...` and say "pinned at commit `e29d8fc`".
- **IsaacLab submodule is not checked out.** The same `git submodule status` prints `-e3f7d9ff... submodules/IsaacLab`. The leading `-` means uninitialised. Arena says it runs on Isaac Sim 6.0.1 and Isaac Lab 3.0.0 (`A:docs/pages/quickstart/installation.rst:32`). So every Isaac Lab API cited below, such as `replay_demos.py`, `reset_to` and the recorder classes, is cited **from Arena's imports and docs only**. The Isaac Lab code itself was not read.
- **GPU** (`nvidia-smi`, 2026-10-27 00:30): one **NVIDIA GeForce RTX 3050 Laptop GPU, 6144 MiB**, driver 580.178.04, CUDA 13.0. 534 MiB was in use by Xorg, gnome-shell, Firefox and Nautilus.
- **Host:** 15 GiB RAM (`free -g`), 12 CPUs (`nproc`), 64 GB free on `/` (`df -h ~`). No Isaac Sim install was found (`~/isaacsim` does not exist). No Isaac Sim or Arena Docker image is present (`docker images`).

### What this means (opinion, kept separate)
- 6 GB is below every GR00T or Arena figure on disk: 24 GB for single-GPU fine-tuning (`A:docs/pages/example_workflows/static_manipulation/step_4_policy_training.rst:114`), "8GB+" just to build a TensorRT engine (`G:scripts/deployment/README.md:193`). I expect this laptop cannot run Isaac Sim with cameras and the GR00T server at the same time, and certainly cannot fine-tune. **NEEDS A RUN** to confirm inference alone, section 9.

---

## 1. Version

### Facts
- The **README says N1.6**: `G:README.md:3` (header alt text), `:22` ("We just released GR00T N1.6"), `:26`, `:46-55` ("What's New in GR00T N1.6").
- The **model code is N1.6**. The only model package is `G:gr00t/model/gr00t_n1d6/`. The only model config is `G:gr00t/configs/model/gr00t_n1d6.py`: class `Gr00tN1d6Config`, `model_type="Gr00tN1d6"` (`:13,17`), registered as `"GrootN1d6"` (`:163`). `G:gr00t/model/__init__.py:1` imports only `Gr00tN1d6Pipeline`. The launcher comment says "Launch finetuning for N1.6" (`G:gr00t/experiment/launch_finetune.py:1`).
- The package version is `version = "0.1.0"` (`G:pyproject.toml:9`). That is the package version, not the model version.
- The base checkpoint named everywhere is `nvidia/GR00T-N1.6-3B` (`G:README.md:126,146,213`).
- **Arena's docs match.** All Arena workflow docs say N1.6, for example `A:docs/pages/quickstart/running_a_real_policy/gr00t.rst:4`, `A:docs/pages/example_workflows/static_manipulation/step_4_policy_training.rst:5,49,108,168`, and `A:docs/pages/references/release_notes.rst:235` ("Upgraded GR00T to N1.6"). The GR00T Docker file is `A:isaaclab_arena_gr00t/docker/Dockerfile.gr00t_1_6:54-55`. One N1.7 artefact remains in Arena: `A:isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py`. Nothing in the GR1 or DROID paths uses it.
- **One README/code mismatch.** The README says the base VLM is "an internal NVIDIA Cosmos-Reason-2B VLM variant" (`G:README.md:53`). The code loads `nvidia/Eagle-Block2A-2B-v2` (section 2). I found nothing that says whether these are the same weights under two names: **UNVERIFIED**.

---

## 2. Backbone and action head

### Facts
- **Backbone class:** `EagleBackbone` (`G:gr00t/model/modules/eagle_backbone.py:8`). It only accepts `model_name == "nvidia/Eagle-Block2A-2B-v2"`, loaded from a vendored directory with `trust_remote_code` (`:40-49`). It asserts flash-attention and bf16 (`:41-44`).
- **What Eagle-Block2A-2B-v2 contains** (`G:gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/config.json`):
  - Vision encoder: SigLIP2, `model_type "siglip2_vision_model"`, 27 layers (`:73,76`). In code it is `self.vision_model = Siglip2VisionModel(...)` (`.../modeling_eagle3_vl.py:100`).
  - Language model: Qwen3, `_name_or_path "Qwen/Qwen3-1.7B"`, 28 layers (`config.json:25,40,42`). In code it is `self.language_model = Qwen3ForCausalLM(...)` (`modeling_eagle3_vl.py:116`).
  - Vision-to-LLM projector: `self.mlp1` (`modeling_eagle3_vl.py:123`).
- **LLM truncation:** layers above `select_layer` are popped (`eagle_backbone.py:52-53`). `select_layer=16` (`G:gr00t/configs/model/gr00t_n1d6.py:28`), so 16 of 28 Qwen3 layers are kept. `backbone_embedding_dim=2048` (`:25`).
- **Action head:** `Gr00tN1d6ActionHead` (`G:gr00t/model/gr00t_n1d6/gr00t_n1d6.py:19`). It is a flow-matching DiT: `AlternateVLDiT` when `use_alternate_vl_dit=True`, the default (`gr00t_n1d6.py:31-42`; config `:67`).
  - `diffusion_model_cfg`: **32 layers**, 32 heads, head dim 48, output dim 1024 (`gr00t_n1d6.py` config `:71-83`). The README agrees: "2x larger DiT (32 layers vs 16 layers in N1.5)" (`G:README.md:54`).
  - The head contains `state_encoder` and `action_decoder` (`CategorySpecificMLP`), `action_encoder` (`MultiEmbodimentActionEncoder`), `vlln` (LayerNorm) and `position_embedding` (`gr00t_n1d6.py:47-71`).
- **Full model:** `Gr00tN1d6(PreTrainedModel)` (`gr00t_n1d6.py:411`). It builds the backbone at `:441-452` and the head at `:455`.
- **Dims** (`gr00t_n1d6.py` config): `max_state_dim=29`, `max_action_dim=29`, `action_horizon=16`, `hidden_size=1024`, `input_embedding_dim=1536` (`:55-59`), `max_num_embodiments=32` (`:102`). The processor's own default is `max_action_horizon=40` (`G:gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py:129`). A checkpoint can override all of these in its `config.json`. The DROID checkpoint's real values are **UNVERIFIED**.
- **Flow matching at inference:** initial noise `torch.randn(...)` with no generator (`gr00t_n1d6.py:310-314`). Then `num_inference_timesteps` Euler steps (`:316-320`). The default is 4 (`config :86`).
- **dtype:** `model_dtype="bfloat16"` (config `:18`). The policy casts the model to bf16 (`G:gr00t/policy/gr00t_policy.py:84`).

---

## 3. Fine-tuning entry and tune flags

### Facts
- **Entry:** `G:gr00t/experiment/launch_finetune.py`, a tyro CLI on `FinetuneConfig` (`:33`). `G:gr00t/experiment/launch_train.py` exists for "more extensive" configuration (`G:README.md:228`). There is **no `examples/finetune.sh`** in this checkout (`ls examples/*.sh` finds nothing).
- **Tune flags on the CLI** (`G:gr00t/configs/finetune_config.py`):

| flag | default | line |
|---|---|---|
| `tune_llm` | False | 35 |
| `tune_visual` | False | 38 |
| `tune_projector` | True | 41 |
| `tune_diffusion_model` | True | 44 |
| `state_dropout_prob` | **0.0** | 47 |
| `random_rotation_angle` | None | 53 |
| `color_jitter_params` | None ("If None, applying the default color jitter augmentation from the pretrained model") | 56-68 |

  The launcher copies them onto the model config at `launch_finetune.py:57-63`.
- **What each flag freezes** (from code):
  - `tune_llm=False` freezes `language_model` (`eagle_backbone.py:69-70`).
  - `tune_visual=False` freezes **both** `vision_model` **and `mlp1`**, the vision-to-LLM projector (`eagle_backbone.py:71-73`).
  - `tune_projector` in this code means the **action-head** encoders and decoder, not the VL projector: `state_encoder`, `action_encoder`, `action_decoder`, `position_embedding`, `mask_token` (`gr00t_n1d6.py:98-105`).
  - `tune_diffusion_model` covers the DiT (`:106-107`).
  - `tune_vlln` covers the action head's LayerNorm on VL features (`:108-109`). It defaults to True (config `:95`) and is not on the CLI.
- **`tune_top_llm_layers`:**
  - Model-config default **4** (`G:gr00t/configs/model/gr00t_n1d6.py:24`). The README says the top 4 VLM layers are unfrozen during pretraining (`G:README.md:55`).
  - It is **not** in `FinetuneConfig` and is not set by the launcher.
  - When fine-tuning starts from a checkpoint, `AutoModel.from_pretrained(start_from_checkpoint, tune_llm=..., tune_visual=..., tune_projector=..., tune_diffusion_model=..., tune_vlln=..., state_dropout_prob=..., backbone_trainable_params_fp32=...)` passes no `tune_top_llm_layers` (`G:gr00t/model/gr00t_n1d6/setup.py:68-81`). The value therefore comes from the checkpoint's own `config.json`.
  - When it is >0, those top LLM layers are trainable **even with `tune_llm=False`** (`eagle_backbone.py:75-78`).
  - The value in `nvidia/GR00T-N1.6-3B/config.json` is **UNVERIFIED**, so whether the "default recipe" silently trains 4 LLM layers is unknown. **NEEDS A RUN:** `hf download nvidia/GR00T-N1.6-3B config.json --local-dir /tmp/n16cfg && grep -n tune_top_llm_layers /tmp/n16cfg/config.json`. The launcher also logs the trainable parameter count (`setup.py:105-110`) and writes `final_model_config.json` (`:101-103`), so the first fine-tune log also answers it.
- **Constants forced by the launcher** (`launch_finetune.py:65-70`): `load_bf16=False`, `reproject_vision=False`, `eagle_collator=True`, `model_name="nvidia/Eagle-Block2A-2B-v2"`, `backbone_trainable_params_fp32=True`, `use_relative_action=True`. The config default for the last one is False (`gr00t_n1d6.py:52`).
- **Training hyperparameters** (`finetune_config.py`): `global_batch_size=64` (`:71`), `dataloader_num_workers=2` (`:74`), `learning_rate=1e-4` (`:77`), `gradient_accumulation_steps=1` (`:80`), `save_steps=1000` (`:86`), `save_total_limit=5` (`:89`), `max_steps=10000` (`:102`), `weight_decay=1e-5` (`:105`), `warmup_ratio=0.05` (`:108`), `shard_size=1024` (`:111`), `episode_sampling_rate=0.1` (`:114`), `num_shards_per_epoch=1e5` (`:117`). The launcher sets `optim="adamw_torch"` (`launch_finetune.py:73`).
- **Not on the finetune CLI** (training config only, reachable through `launch_train.py`): `bf16=True`, `tf32=True` (`G:gr00t/configs/training/training_config.py:35,37`), `deepspeed_stage=2` (`:63`), `gradient_checkpointing=False` (`:64`).
- **Arena's own GR1 recipe tunes the vision encoder.** It passes `--no-tune-llm --tune-visual --tune-projector --tune-diffusion-model --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08`: 8 GPUs, batch 24, 20k steps (`A:docs/pages/example_workflows/static_manipulation/step_4_policy_training.rst:158-175`), or 1 GPU, batch 16, 30k steps (`:199-215`). The same recipe is used for the sequential task (`A:docs/pages/example_workflows/sequential_static_manipulation/step_4_policy_training.rst:175,214`).
- **Run-to-run variance:** the README reports differences "as large as 5-6% between runs" with the same configuration and seed (`G:README.md:241-243`).

## 4. LoRA

### Facts
- GR00T's own configs have **no LoRA option**. A grep for `lora|peft` over `*.py *.md *.sh *.toml` hits only:
  - the `peft==0.17.1` dependency (`G:pyproject.toml:21`);
  - the **vendored Eagle remote code**: `use_backbone_lora` and `use_llm_lora`, both 0 in the shipped config (`.../Eagle-Block2A-2B-v2/config.json:56-57`). When non-zero, they wrap `vision_model` or `language_model` in a PEFT LoRA with `lora_alpha = 2*r`, dropout 0.05 (`modeling_eagle3_vl.py:133-138,150-165`).
- Nothing in `FinetuneConfig`, `launch_finetune.py` or `setup.py` sets those two keys. Because `EagleBackbone` builds the Eagle model from the vendored config (`eagle_backbone.py:45-47`), turning LoRA on means editing that config or the code.
- No LoRA exists for the action head (DiT) anywhere.
- Whether those keys survive checkpoint loading, and whether the vision-side LoRA trains under `tune_visual=False` (which sets `requires_grad_(False)` on `vision_model` *after* construction, `eagle_backbone.py:71-72`), is **UNVERIFIED**.

## 5. Built-in augmentation

### Facts
- **Builder:** `build_image_transformations_albumentations` (`G:gr00t/model/gr00t_n1d6/image_augmentations.py:192`).
  - Train pipeline in a `ReplayCompose`, so every view gets the same draw (`:247`): `SmallestMaxSize` → `FractionalRandomCrop` → `SmallestMaxSize` (`:225-229`).
  - Then optional `A.Rotate`, only when `random_rotation_angle` is set and non-zero (`:231-232`).
  - Then optional `A.ColorJitter`, only when `color_jitter_params` is set (`:234-245`).
  - Eval pipeline: deterministic center crop (`:251-257`).
  - A torchvision variant also exists (`:321-360`).
- **No** blur, noise, HDR, mask or background augmentation exists in this file (the N1.7 file had mask-based extras; see the diff table).
- **Defaults:**
  - Model config: `shortest_image_edge=256`, `crop_fraction=0.95`, `image_crop_size=None`, `image_target_size=None`, `random_rotation_angle=None`, `color_jitter_params=None`, `use_albumentations_transforms=True` (`G:gr00t/configs/model/gr00t_n1d6.py:39-47`).
  - Processor class defaults: `shortest_image_edge=512`, `crop_fraction=0.95`, `use_albumentations=False` (`processing_gr00t_n1d6.py:119-120,130`).
- **When fine-tuning from a checkpoint, the checkpoint wins.** `Gr00tN1d6Processor.from_pretrained` loads `processor_config.json`, `statistics.json` and `embodiment_id.json` from the checkpoint (`processing_gr00t_n1d6.py:484-506`). It overrides only `random_rotation_angle`, `color_jitter_params` and `use_relative_action`, and only when the passed value is not None (`:513-522`). So a `None` jitter on the CLI keeps the checkpoint's jitter, and crop size and resize come from the checkpoint. The checkpoint's values are **UNVERIFIED**. **NEEDS A RUN:** `hf download nvidia/GR00T-N1.6-3B processor_config.json --local-dir /tmp/n16cfg && python -m json.tool /tmp/n16cfg/processor_config.json | grep -n -i "jitter\|crop\|edge\|rotation"`.
- The README and Arena fine-tune commands pass `--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08` explicitly (`G:README.md:224`; Arena refs in section 3).

## 6. Training data format

### Facts
- **GR00T-flavoured LeRobot v2.** LeRobot v3 must be converted with `G:scripts/lerobot_conversion/convert_v3_to_v2.py` (`G:getting_started/data_preparation.md:5-15`).
- **Layout** (`data_preparation.md:20-40`): `meta/{episodes.jsonl, modality.json, info.json, tasks.jsonl}`, `videos/chunk-000/observation.images.<name>/episode_XXXXXX.mp4`, `data/chunk-000/episode_XXXXXX.parquet`. `meta/stats.json` and `meta/relative_stats.json` are computed (`:146`).
- **Parquet:** `observation.state` and `action` are concatenated 1D float arrays (`:47-59`). Annotation columns are `annotation.<source>.<type>`, indices into `tasks.jsonl` (`:153-155`). `modality.json` slices state and action into named fields (`:101-106`).
- **Python modality config:** registered with `register_modality_config(config, embodiment_tag)`, which asserts that the tag is not already registered (`G:gr00t/configs/data/embodiment_configs.py:349-355`). It is passed with `--modality-config-path` (`finetune_config.py:28-32`; loaded at `launch_finetune.py:15-25,37-38`).
- **Arena's converter:** `A:isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py` with YAML configs in `A:isaaclab_arena_gr00t/lerobot/config/`. Example: `gr1_manip_config.yaml` maps `state_name_sim: robot_joint_pos`, `action_name_sim: processed_actions`, `pov_cam_name_sim: robot_pov_cam_rgb`, `fps: 50` (`A:docs/pages/example_workflows/static_manipulation/step_4_policy_training.rst:84-102`).
- **Arena's GR1 modality config:** `A:isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py`, registered under `EmbodimentTag.GR1` (`:88`).
  - video `ego_view`; state and action `left_arm, right_arm, left_hand, right_hand` with sin/cos state encoding; action `delta_indices` 0..15, all `ABSOLUTE` (`:14-80`).
  - language key `annotation.human.action.task_description`.

## 7. Embodiment tags

### Facts
- `EmbodimentTag` (`G:gr00t/data/embodiment_tags.py:14-61`):
  - pretrain: `ROBOCASA_PANDA_OMRON`, **`GR1`**;
  - pre-registered posttrain: `UNITREE_G1`, `LIBERO_PANDA`, `OXE_GOOGLE`, `OXE_WIDOWX`, **`OXE_DROID`** ("relative joint position actions"), `BEHAVIOR_R1_PRO`;
  - `NEW_EMBODIMENT`.
- **Projector slot per tag** (`processing_gr00t_n1d6.py:33-45`): gr1 20, oxe_droid 16, new_embodiment 10, and so on.
- **Pre-registered modality configs** (`embodiment_configs.py:11`): `unitree_g1` (`:13`, action range(30) `:31`), `libero_panda` (`:91`, range(16) `:109`), `oxe_widowx` (`:125`, range(8) `:144`), `oxe_google` (`:168`, range(8) `:187`), `behavior_r1_pro` (`:211`, range(32) `:247`) and `oxe_droid` (`:303`). For `oxe_droid`, video is `exterior_image_1_left, wrist_image_left`, state and action are `joint_position, gripper_position`, the action has range(32), joint_position is `RELATIVE` and gripper is `ABSOLUTE` (`:303-345`, reps at `:327,333`).
- `GR1` has **no** entry here. Its modality config comes from the checkpoint's processor config, or from Arena's file for fine-tuning.
- **Arena's tag strings exist in this enum.**
  - `GR1`: `A:isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml:10`.
  - `OXE_DROID`: `A:isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml:8`.
  - `NEW_EMBODIMENT` (G1): `A:.../g1_locomanip_gr00t_closedloop_config.yaml:10`.
- **Tag check:** `Gr00tPolicy.__init__` does `self.processor.get_modality_configs()[self.embodiment_tag.value]` (`G:gr00t/policy/gr00t_policy.py:93`). A tag missing from the checkpoint raises a plain `KeyError`, with no hint.

## 8. Inference

### Facts
- **Server:** `G:gr00t/eval/run_gr00t_server.py`. `ServerConfig` has `model_path`, `embodiment_tag` (default `NEW_EMBODIMENT`), `device="cuda"`, `dataset_path`, `modality_config_path`, `execution_horizon`, `host="0.0.0.0"`, `port=5555`, `strict=True` and `use_sim_policy_wrapper=False` (`:15-50`).
- **`--modality-config-path` is ignored when `--model-path` is given.** The model branch builds `Gr00tPolicy(embodiment_tag, model_path, device, strict)` (`:66-72`). The modality config path is read only in the `ReplayPolicy` branch (`:73-86`). Arena's docs pass it to the server anyway (`A:docs/pages/example_workflows/static_manipulation/step_5_evaluation.rst:63`). It has no effect there.
- **`ReplayPolicy`** serves dataset actions through the same server API (`run_gr00t_server.py:73-86`). It is a ready-made source of "known-good actions in the policy's own output format" (see PLAN_UPDATE §(c)).
- **Transport:** ZeroMQ `REP` socket (`G:gr00t/policy/server_client.py:63`), serving one request at a time in `run()` (`:110-140`). Client `REQ` (`:169`), `timeout_ms=15000` (`:155`). msgpack serialisation (`:5,17-22`). Endpoints: `ping`, `kill`, `get_action`, `reset`, `get_modality_config` (`:68-77`).
- **Batching:** `Gr00tPolicy._get_action` unbatches the observation and processes each one (`gr00t_policy.py:306-331`), then collates and casts to bf16 (`:339`).
- **`reset` is a no-op** that returns `{}` (`gr00t_policy.py:408-417`).
- **No seeding:** there is no `seed` anywhere in `gr00t_policy.py`. Action noise comes from the global torch RNG (`gr00t_n1d6.py:310`). Per-episode reproducibility of the policy's sampling therefore needs the caller to seed torch on the server. Nothing does this today.
- **Chunk length:**
  - Model config `action_horizon=16` (`gr00t_n1d6.py:57`). Per-embodiment length comes from the action `delta_indices` (section 7). Arena configs use 16 for GR1 and 32 for DROID (`A:.../gr1_manip_gr00t_closedloop_config.yaml:8`, `droid_manip_gr00t_closedloop_config.yaml:6`).
  - Execution length (replan steps) is client-side: Arena's `action_chunk_length` (same yaml files). The GR00T sim rollout script defaults `n_action_steps` to 8 (`G:gr00t/eval/rollout_policy.py:45,475`).
  - There is **no** `_horizon_contract.py` in this checkout (`ls G:gr00t/eval`).
- **Denoising steps:** `num_inference_timesteps=4` is a model-config value (`gr00t_n1d6.py` config `:86`). `--denoising-steps` / `denoising_steps` fields exist in `G:gr00t/eval/open_loop_eval.py:256` and `G:scripts/deployment/standalone_inference_script.py:591`, but grep finds **no use** of them. They are dead flags.
- **Open-loop evaluation exists:** `G:gr00t/eval/open_loop_eval.py`. It connects to a server (`host`, `port`, `:232-235`), runs dataset trajectories and writes GT-vs-predicted plots with unnormalised MSE (`G:README.md:257-270`). It is a candidate for vla-router's `PolicyOpenLoopProbe` stub (ISSUES #65).
- **Measured speed in the README:** 4 denoising steps, single view, `torch.compile`: RTX 4090 44 ms (22.8 Hz), H100 38 ms, RTX 5090 37 ms (`G:README.md:173-181`).

## 9. GPU and VRAM

### Facts
- No VRAM figure for **inference** exists in this checkout. `G:getting_started/hardware_recommendation.md` lists server bills of materials only (`:12-43`). The only number is "8GB+ recommended" for building a TensorRT engine (`G:scripts/deployment/README.md:193`).
- **Fine-tuning:** "We recommend using 1 H100 node or L40 node", "A6000 will also work" (`G:README.md:237-238`). Arena gives 8× ≥48 GB (best quality) or **1× 24 GB** (low hardware), about 2-3 h on one Ada6000 for 30k steps (`A:docs/pages/example_workflows/static_manipulation/step_4_policy_training.rst:113-114,131,179-188`). The `--num-shards-per-epoch`, `--shard-size` and `--dataloader-num-workers` flags reduce memory (`G:getting_started/policy.md:443-450`).
- Parameter count: "3B parameters" (`G:README.md:126`). How many are actually loaded after the LLM truncation at layer 16 is **UNVERIFIED**.
- Isaac Sim's own GPU requirement is only an external link (`A:docs/pages/quickstart/installation.rst:34-35`): **UNVERIFIED**.

### NEEDS A RUN (VRAM)
1. Inference VRAM of the pilot checkpoint, alone:
   ```
   cd external/IsaacLab-Arena/submodules/Isaac-GR00T && uv sync --python 3.10 && uv pip install -e .
   uv run python gr00t/eval/run_gr00t_server.py --model-path <CKPT_DIR> --embodiment-tag GR1 --device cuda --host 127.0.0.1 --port 5555 &
   sleep 120; nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   ```
2. Isaac Sim, GR1 env, cameras on, zero-action policy, with no server: same `nvidia-smi` query while this runs:
   `python isaaclab_arena/evaluation/policy_runner.py --viz none --policy_type zero_action --num_steps 200 --enable_cameras put_item_in_fridge_and_close_door --embodiment gr1_joint --object ranch_dressing_hope_robolab`
3. Both together, which is what Rung 1 needs. If (1) plus (2) is more than 6 GB, the pilot must run on a remote GPU.

---

## 10. Facts that DIFFER from the N1.7 file

The left column is **from the earlier file, not re-verified** (that clone no longer exists). The right column is verified in this checkout.

| Topic | N1.7 (from the earlier file, not re-verified) | N1.6 at `e29d8fc` (verified) |
|---|---|---|
| Version | N1.7 "General Availability" (`OLD:19`) | N1.6 (`G:README.md:22,26`); package `gr00t_n1d6` |
| Backbone | `nvidia/Cosmos-Reason2-2B` (Qwen3-VL), `Qwen3Backbone`, vision is `model.visual` (`OLD:20-21`) | `nvidia/Eagle-Block2A-2B-v2` = SigLIP2 (27 layers) + Qwen3-1.7B (28 layers), `EagleBackbone`, vision is `vision_model`, VL projector is `mlp1` (§2). The README says "Cosmos-Reason-2B variant" (`G:README.md:53`) |
| `select_layer` | 12 (`OLD:22`) | 16 (`gr00t_n1d6.py` config `:28`) |
| DiT depth | 16 layers (`OLD:23`) | 32 layers (config `:74`) |
| Dims | `max_state/action_dim=132`, `action_horizon=40` (`OLD:25`) | 29 / 29 / 16 (config `:55-57`); processor `max_action_horizon=40` (`processing:129`) |
| `tune_top_llm_layers` | default 0, not on CLI (`OLD:42`) | default **4**, not on CLI, inherited from the checkpoint config on fine-tune (§3) |
| `state_dropout_prob` | CLI 0.2, model 0.8 (`OLD:39,168`) | CLI 0.0 (`finetune_config.py:47`), model 0.0 (config `:98`) |
| `tune_visual` scope | vision encoder (`OLD:21`) | vision encoder **and** `mlp1` (`eagle_backbone.py:71-73`) |
| Fine-tune wrapper | `examples/finetune.sh` with jitter on by default (`OLD:29-30,60`) | no `finetune.sh`; jitter only if passed (README and Arena pass it) |
| Augmentation | crop 230→256 legacy, mask-based extras (`OLD:50-61`) | `shortest_image_edge`/`crop_fraction` (256 / 0.95 model, 512 / 0.95 processor), rotate, jitter; no mask extras (§5) |
| Optimizer forced by launcher | `adamw_torch` (`OLD:45`) | same (`launch_finetune.py:73`). Unchanged; listed because it was checked |
| LoRA | none anywhere (`OLD:46`) | none in GR00T configs, but the vendored Eagle code has `use_backbone_lora` / `use_llm_lora` (vision / LLM), not exposed (§4) |
| Tags | no `GR1`, no `OXE_DROID`; Arena tags mismatch (`OLD:12,76-90`) | `GR1` and `OXE_DROID` exist (`embodiment_tags.py:21,47`); Arena tags match (§7) |
| Bad-tag handling | hint printed (`OLD:88`) | plain `KeyError` (`gr00t_policy.py:93`) |
| Horizon contract | `gr00t/eval/_horizon_contract.py` (`OLD:96-97`) | file absent |
| Denoising flag | "no inference CLI flag" (`OLD:98`) | same, plus two dead `denoising_steps` fields (§8) |
| VRAM | inference 16 GB+, fine-tune 40 GB+, ~35 GB default tune set, 80 GB+ with LLM or visual tuning (`OLD:101-104`) | no inference figure; fine-tune 1× 24 GB low-hardware per Arena (§9) |
| Speed | H100 11.7 Hz eager, 35.9 Hz TRT (`OLD:99`) | RTX 4090 22.8 Hz, H100 26.3 Hz with `torch.compile` (`G:README.md:175-180`) |
| Arena pinned GR00T checkout | "not checked out" (`OLD:7`) | checked out at `e29d8fc` (§0) |
