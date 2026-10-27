# VERIFY items (settle on the rented GPU box)

Every `# VERIFY(<id>)` marker in `vla-isaac/` and `policy-wrapper/` has a row here. Nothing in this
list has been run. "Applied" is filled in by `vla-isaac/scripts/gpu/collect_measurements.py`.

Prefixes as in `docs/IMPL_PLAN.md` (`A:` Arena, `G:` GR00T). Commands run from `external/IsaacLab-Arena`
with Arena's venv active unless they say otherwise.

| id | What to check | Where it is used | Command that settles it | Added |
|---|---|---|---|---|
| V-viz | `--viz none` is accepted (Isaac Lab AppLauncher flag, not in Arena's CLI files) and runs headless | `BackendConfig.app_flags`, `arena_cmd` | `python isaaclab_arena/evaluation/experiment_runner.py --help \| grep -A3 -- --viz` | P1 |
| V-exp-yaml | the typed `environment:` keys of `put_item_in_fridge_and_close_door` in an experiment YAML (`type`, `embodiment`, `object`, `enable_cameras`) and the `--experiment_config` / `--experiment_output_directory` invocation | `config/pilot_gr1_fridge.yaml`, `arena_cmd.experiment_yaml` | run the YAML `IsaacBackend` writes (`runs/*/build0_n1/experiment.yaml`) by hand with `num_episodes: 1` | P1 |
| V-exp-remote-policy | the experiment-runner YAML accepts `policy.type` = a dotted class path of the remote GR00T policy, `policy_config_yaml_path`, `remote_host`, `remote_port` (the DROID example `A:isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml` uses them; the GR1 pilot path is untested) | `arena_cmd.PolicySpec` | same run as V-exp-yaml with the stock `isaaclab_arena_gr00t...Gr00tRemoteClosedloopPolicy` and the server from R2 | P1 (user) |
| V-runner-parity | nominal success via `policy_runner.py` vs the experiment runner, same build seed, same episodes, clean mode | backend switched to the experiment runner (IMPL_PLAN Q1) | `scripts/gpu/R4b_runner_parity.sh` (Phase 7): 50 episodes each, compare per-episode `success` on `(env_id, episode_in_env)` | P1 (user) |
| V-instruction | nominal success with the eval instruction ("sauce bottle") vs the training instruction from the dataset's `meta/tasks.jsonl` ("ranch dressing bottle" per Arena's converter YAML) | `policy-wrapper/config/eval.yaml` `client.language_instruction` | `scripts/gpu/R4c_instruction_ab.sh` (Phase 7): 20 nominal episodes each; if the training string scores as well, switch eval.yaml to it | P1 (user) |
| V-timestamped-dir | `policy_runner.py --output_base_dir D` adds a timestamped subdirectory (`timestamped_run_dir`) holding `episode_results_rank0.jsonl` | `IsaacBackend` with `runner: policy_runner` (reads `rglob`) | `ls -R` the R4 output | P1 |
| V-jsonl-success | `success` is a bool for this task (it is `None` without a `success` termination term, `A:isaaclab_arena/recording/common_terms.py:133-135`) | `results/episode_jsonl.read_records` | `head -1` of an R4 JSONL | P1 |
| V-var-record | the value format recorded under `variations` for a build-time draw (`[100.0]` or `100.0`) | fake runner fixtures, Phase 5 replay | `jq .variations` on one record of R7 | P1 |
| V-list-var | `--list_variations` output for the pilot env: asset names (`light`, `ranch_dressing_hope_robolab`), variation names, field paths, configclass `to_dict` extras | `config/factors_gr1_fridge.yaml`, `scene/list_variations.py` | `scripts/gpu/R3_list_variations.sh`, then `python -m vla_isaac.scene.list_variations --input ... --output /tmp/t.yaml --names config/factor_names_gr1_fridge.yaml && diff` | P1 |
| V-actuator | the GR1 robot cfg is at `embodiment.scene_config.robot`, its `actuators` dict holds groups with `stiffness` / `damping`, and editing them after `factory.build()` changes the gains | `arena_ext/actuator_env.py` | in Arena's python after SimulationApp: build the pilot env factory, `print(env.embodiment.scene_config.robot.actuators)`; then a 1-episode run with a 10x stiffness override and compare joint tracking | P1 |
| V-ext-env | a dynamic subclass of the pilot factory registers with `register_environment` under a new name, the experiment loader finds its cfg type, importing the factory before SimulationApp starts is safe, and `experiment_runner.main()` reads `sys.argv` | `arena_ext/actuator_env.py`, `arena_ext/launch_experiment.py` | `VLA_ISAAC_ACTUATOR_OVERRIDES='{}' VLA_ISAAC_BASE_ENV=isaaclab_arena_environments.gr1_put_and_close_door_environment:GR1PutAndCloseDoorEnvironment VLA_ISAAC_ACTUATED_ENV_NAME=vla_test python -m vla_isaac.arena_ext.launch_experiment --experiment_config <yaml with environment.type: vla_test> --experiment_output_directory /tmp/x --enable_cameras` | P1 |
| V-traj-size | what `record_trajectories: true` writes and how big it is. Its terms (`A:isaaclab_arena/terms/recorders.py:277-289`) hold states, actions, processed actions, episode id, gripper and EE poses but **no camera term**; camera frames are in `ArenaEnvRecorderManagerCfg`, which the builder passes only as `demo_recorder_config` (`arena_env_builder.py:474`). The 390 MB/episode estimate may therefore not apply | `TrajectoryConfig.camera_stride`, Phase 3 reader | 2-episode run with traces on: `du -sh`, `h5py` visit of the dataset | P1 |
| V-hdf5-keys | HDF5 group/key names written by Isaac Lab's recorder manager and terms | Phase 3 `traces/hdf5_reader.py` | `python -c "import h5py,sys;h5py.File(sys.argv[1]).visit(print)" <dataset>` | P0 |
| V-hdf5-export | when / where the recorder exports (each reset? successes only? export mode) | Phase 3 | list the run dir after the V-traj-size run | P0 |
| V-ee-frame | GR1 EE frame transformer and frame names | Phase 3 grasp record | HDF5 visit; `grep -rn get_ee_frame_transformer_names isaaclab_arena/embodiments` | P0 |
| V-obj-state | object pose path in HDF5 states and the pick-up object's scene name | Phase 3 | HDF5 visit | P0 |
| V-progress | predicate names in `progress.events` for this composite task | Phase 3 events | `jq .progress.events` on an R4 record | P0 |
| V-mimic-override | `generate_dataset.py` accepts variation overrides; trial-success line format | Phase 4 | `scripts/gpu/mimic_override_check.sh` (Phase 7) | P0 |
| V-hf-rev | HF revision hash of the pilot checkpoint | `policy-wrapper/config/eval.yaml` `TODO_PIN` | `python -c "from huggingface_hub import HfApi; print(HfApi().model_info('nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task').sha)"` | P0 |
| V-ckpt-cfg | `tune_top_llm_layers`, `num_inference_timesteps`, `action_horizon`, processor crop/resize/jitter | Phase 4 spec, Phase 6 launcher | R1 `grep` | P0 |
| V-client-tests | Arena's GR00T client tests import without Isaac Sim | Phase 2 parity | `scripts/gpu/check_client_tests.sh` | P0 |
| V-converter-noisaac | `convert_hdf5_to_lerobot.py` runs without Isaac | Phase 6 | Arena's `test_convert_hdf5_to_lerobot.py` in a plain venv | P0 |
| V-finetune-log | trainable-parameter log line format | Phase 6 launcher | first fine-tune log | P0 |
| V-rgb-order | `camera_obs` is RGB uint8 HWC | Phase 3 FrameStats | one instrument JSON from R4 | P0 |
| V-reset-calls | `policy.reset()` call pattern (no-arg at start, then per env) in multi-env runs | Phase 2 seeding, Phase 3 start order | instrument log of R5 | P0 |
| V-render-defaults | Arena default render settings (`env.render: {}`) | Phase 4 spec | record in R0 | P0 |

P0 = listed in `docs/IMPL_PLAN.md` §4; P1 = added in Phase 1 (user = asked for in the Phase 0 approval).
