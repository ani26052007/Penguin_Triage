# Implementation plan: simulation side (`vla-isaac/`, `policy-wrapper/`)

Written 2026-10-27, after Phase 0 of `docs/PROMPT_sim_side.md`. Status: **waiting for approval. No code has been written.**

Citation prefixes as in the docs: `G:` GR00T pin, `A:` Arena (HEAD `7632eb7d5e`), `R:` vla-router, `P:` `docs/plan.md`.

What I read in full: `docs/plan.md`, `docs/PLAN_UPDATE.md`, `docs/PILOT_AND_ORDER.md`, `docs/facts_gr00t_n16.md`, `DECISIONS.md`, `ISSUES.md`, `R:DECISIONS.md`, `R:README.md`; `R:router/schemas.py`, `R:ladder/backend.py`, `R:router/prereq/{trace,behavior_trace,rollout_cache}.py`, `R:router/stage1/cfg_contract.py`, `R:ladder/workspace.py`, `R:ladder/rung0.py`, `R:tests/workspace_cases.py`, `R:tests/test_backend_cache.py`, `R:tests/test_trace_code_key.py`; `A:isaaclab_arena_gr00t/{policy,utils,embodiments/gr1,tests}/*`, `A:isaaclab_arena/policy/action_scheduling/*`, `A:isaaclab_arena/evaluation/policy_runner{,_cli}.py`, `A:isaaclab_arena/recording/*`, `A:isaaclab_arena/variations/{variation_base,variations_hydra,variations_printing,recorded_variation_samples,recorded_variation_replay,variation_recorder,uniform_sampler,sampler_base,light_*,hdr_image}_variation.py` (others skimmed), `A:isaaclab_arena/terms/recorders.py`. I also read parts of the Arena CLI, env builder, experiment runner and the pilot environment, plus `G:gr00t/policy/{server_client,policy,gr00t_policy}.py`, `G:gr00t/eval/run_gr00t_server.py` and `G:gr00t/configs/finetune_config.py`.

Baseline: `cd vla-router && env -u PYTHONPATH ~/miniconda3/envs/vla/bin/python -m pytest -q` → **752 passed, 1 skipped**.

---

## 1. vla-router signatures I will implement or call (copied from the code)

### `R:ladder/backend.py`
```python
class RolloutBackend(ABC):
    @abstractmethod
    def run(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef,
            seeds: Sequence[int], mode: Mode) -> list[EpisodeResult]:
        """Return one EpisodeResult per seed, in the order of ``seeds``."""

class IsaacBackend(RolloutBackend):      # placeholder, lines 35-56
    def run(self, scene, code_ref, actor, seeds, mode):  # pragma: no cover
        raise NotImplementedError("IsaacBackend is not implemented; see TODO(verify) notes")

class MockBackend(RolloutBackend):
    def __init__(self, rules: Sequence[MockRule], default_rate: float,
                 replay_noise: float = 0.0, salt: str = "mock", workspaces=None) -> None
    def run(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef,
            seeds: Sequence[int], mode: Mode) -> list[EpisodeResult]
```
The real backend lives in `vla_isaac.backend.isaac_backend.IsaacBackend`. I will **not** replace the placeholder class in `R:ladder/backend.py`, because that would be a third vla-router edit (§3, D14).

### `R:router/schemas.py`
```python
class Mode(str, Enum):            CLEAN = "clean"; INCIDENT = "incident"
class ActorKind(str, Enum):       POLICY = "policy"; REFERENCE = "reference"
class ActionPipeline(str, Enum):  WRAPPER = "wrapper"; RAW = "raw"

@dataclass(frozen=True)
class Factor:      name: str; value: str
@dataclass(frozen=True)
class SceneSpec:
    nominal: str
    factors: tuple[Factor, ...] = ()
    def restore(self, name: str) -> "SceneSpec"
    def restore_all(self) -> "SceneSpec"
    def with_factor(self, name: str, value: str) -> "SceneSpec"
    def key(self) -> str
@dataclass(frozen=True)
class TrainingSpec:
    policy_name: str
    image_size: Optional[tuple[int, int]] = None            # (height, width)
    channel_order: Optional[str] = None
    norm_stats: Optional[Mapping[str, Mapping[str, Sequence[float]]]] = None
    camera_poses: Optional[Mapping[str, Sequence[float]]] = None
    action_horizon: Optional[int] = None
    replan_steps: Optional[int] = None
    denoising_steps: Optional[int] = None
    control_hz: Optional[float] = None
    render_settings: Optional[Mapping[str, Any]] = None
    state_layout: Optional[Mapping[str, Sequence[int]]] = None     # name -> [start, end)
    action_layout: Optional[Mapping[str, Sequence[int]]] = None
    instruction_templates: Optional[Mapping[str, str]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: Mapping) -> "TrainingSpec"     # unknown keys raise ValueError
@dataclass(frozen=True)
class CodeRef:     code_hash: str; eval_config: Mapping[str, Any] = field(default_factory=dict)
    def key(self) -> str      # f"{self.code_hash}+cfg:{short_hash(dict(self.eval_config))}"
@dataclass(frozen=True)
class ActorRef:    kind: ActorKind; checkpoint: str; pipeline: ActionPipeline = ActionPipeline.WRAPPER
    def key(self) -> str      # policy + RAW raises ValueError
@dataclass(frozen=True)
class EpisodeResult: seed: int; success: bool; logs: Mapping[str, Any] = field(default_factory=dict)
@dataclass(frozen=True)
class Incident: incident_id, scene, training_spec, code_ref, policy, failing_seeds,
                observed_mode=Mode.INCIDENT, observed_success_rate=None, reference_rate=None,
                log_refs: Mapping[str, str] = {}, nominal_scene=None
```

### `R:router/prereq/trace.py`
```python
@dataclass(frozen=True)
class FrameStats:  shape: tuple[int, int, int]; channel_means: tuple[float, ...]; min_value: float; max_value: float
                   row_profile=(); col_profile=(); saturated_fraction=0.0; missing_texture_fraction=0.0
@dataclass(frozen=True)
class FrameSample: step: int; camera: str; policy_input: FrameStats; render: Optional[FrameStats] = None
@dataclass(frozen=True)
class StepPair:    obs_step: int; applied_step: int; chunk_offset: int = 0     # extra_delay = applied - obs - offset
@dataclass(frozen=True)
class SimHealth:   nan_fraction=0.0; max_penetration=0.0; object_initial_pos=(); object_rest_pos=()
@dataclass(frozen=True)
class EpisodeTrace: seed: int; mode: Mode; frames=(); step_pairs=(); raw_actions=(); joint_cmd=(); joint_pos=();
                    gripper_cmd_width=(); gripper_width=(); sim: Optional[SimHealth] = None; code_key: Optional[str] = None
class TraceUnavailable(RuntimeError)
def verify_code_key(traces: Sequence, incident: Incident, what: str) -> None
class TraceSource(ABC):
    def load(self, incident: Incident, mode: Mode) -> list[EpisodeTrace]        # template: never override
    @abstractmethod
    def _load(self, incident: Incident, mode: Mode) -> list[EpisodeTrace]
LOG_REF_CLEAN = "clean"; LOG_REF_INCIDENT = "incident"                        # Incident.log_refs keys
```

### `R:router/prereq/behavior_trace.py`
```python
PHASES = ("reach", "grasp", "lift", "place")
@dataclass(frozen=True)
class GraspRecord: step: int; ee_pos: tuple[float, ...]; target_pos: tuple[float, ...]; distractor_pos=()
@dataclass(frozen=True)
class SubtaskProgress: name: str; completed: bool                 # name not in PHASES -> ValueError
@dataclass(frozen=True)
class BehaviorTrace: seed: int; mode: Mode; success: bool; grasp: Optional[GraspRecord] = None;
                     subtasks: tuple[SubtaskProgress, ...] = (); code_key: Optional[str] = None
class BehaviorSource(ABC):
    def load(self, incident: Incident, mode: Mode) -> list[BehaviorTrace]     # template
    @abstractmethod
    def _load(self, incident: Incident, mode: Mode) -> list[BehaviorTrace]
```
`R:router/vocab.py:50-51` also has `PHASES_ACTION_GAP = ("grasp", "lift", "place")` and `PHASES = ("reach",) + PHASES_ACTION_GAP`. `decide.py:402` reads `V.PHASES_ACTION_GAP`.

### `R:router/stage1/cfg_contract.py`
```python
FIELD_ORDER = ("action_horizon", "replan_steps", "denoising_steps", "control_hz", "norm_stats",
               "camera_poses", "image_size", "render_settings", "instruction_templates")
@dataclass(frozen=True)
class ConfigMapping:
    keys: Mapping[str, str]                                      # TrainingSpec field -> dotted eval-config path
    factor_covers: Mapping[str, str] = field(default_factory=dict)
    rtol: float = 1e-3
    atol: float = 1e-6
class ConfigContractCheck(Check):
    source = "contract_cfg"
    def __init__(self, mapping: ConfigMapping) -> None
    def run(self, incident: Incident) -> list[Evidence]
```

### `R:ladder/workspace.py`
```python
EVAL_CONFIG_PATH = "config/eval.yaml"
def file_config_loader(path: str = EVAL_CONFIG_PATH) -> ConfigLoader      # ConfigLoader = Callable[[Path], Mapping[str, Any]]
class WorkspaceError(RuntimeError)
class UnrebuildablePatch(WorkspaceError)
class WorkspaceManager:
    def __init__(self, store: str | Path, bases: Mapping[str, str | Path],
                 config_loader: ConfigLoader = load_eval_config)
    def base_ref(self, label: str, eval_config: Optional[Mapping] = None) -> CodeRef     # "<label>@<tree12>"
    def materialize(self, code_ref: CodeRef, dest: Optional[str | Path] = None) -> Path
    def view(self, code_ref: CodeRef) -> Path
    def extend(self, code_ref: CodeRef, patch: Diff) -> CodeRef
def code_ref_of(workdir: str | Path, config_loader: ConfigLoader = load_eval_config) -> CodeRef
```

### `R:router/prereq/rollout_cache.py` (target of change 1)
```python
@dataclass(frozen=True)
class CacheKey:  code: str; actor: str; scene: str; seed: int; mode: str
    @classmethod
    def make(cls, scene, code_ref, actor, seed, mode) -> "CacheKey"
class RolloutCache: def __init__(self, path: Optional[str] = None) -> None
class CachedBackend(RolloutBackend): def __init__(self, backend: RolloutBackend, cache: RolloutCache) -> None
```

### `R:ladder/rung0.py`, `R:ladder/stats.py`
```python
@dataclass(frozen=True)
class Rung0Config: lint_cmd=""; test_cmd=""; smoke_cmd=""; expected_files=(); forbidden_prefixes=(".git/",);
                   max_diff_lines=400; timeout_s=120; max_output_chars=3000
def run_rung0(env, cfg: Rung0Config, *, task: str = "", reviewer: Optional[Model] = None) -> Rung0Result
    # env needs execute(cmd, timeout_s) and git_diff()
def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Proportion
def agreement_rate(first: Mapping[int, bool], second: Mapping[int, bool]) -> float
```

### Error path (F3)
`R:ladder/loop.py:313`: `_Run.execute` catches `Exception` and ends the incident UNRESOLVED with `error_stage` set to router or rung1. A typed `ArenaRunError(RuntimeError)` raised from `IsaacBackend.run` is therefore enough. No router change is needed.

---

## 2. File list (adjusted to what I found)

```
vla-isaac/
  pyproject.toml                         # depends on vla-repair-ladder (editable path ../vla-router), pyyaml, numpy
  config/backend.example.yaml            # arena_root, gr00t_root, python exes, output_root, host/port, incident_num_envs, timeouts,
                                         # runner kind (D2), seed stride (D10), workspace store + bases
  config/pilot_gr1_fridge.yaml           # ALL task-specific values: task, embodiment, object, policy type/yaml, factor table,
                                         # hand-closure joints, reach distance, grasp threshold, phase map, Mimic args
  config/factors_gr1_fridge.yaml         # factor table; regenerated on the GPU box by list_variations.py
  vla_isaac/
    config.py                            # backend config loader (dataclasses, YAML)
    seeds.py                             # router seed <-> (build seed, episode index), grouping, order restore
    scene/factors.py                     # SceneSpec factor -> Hydra overrides (total, injective, restore(i))
    scene/list_variations.py             # parse `--list_variations` stdout into a factor table
    runner/arena_cmd.py                  # policy_runner.py argv AND experiment-runner YAML (D2)
    runner/subprocess_runner.py          # run with timeout, logs captured, ArenaRunError; FakeArenaRunner
    runner/actuator_env.py               # --external_environment_class_path wrapper that applies env.actuator_overrides (D7)
    results/episode_jsonl.py             # episode_results_*.jsonl -> EpisodeResult
    backend/server.py                    # serve.py process start/ping/stop/restart; ZMQ+msgpack `ping` (own code, D15)
    backend/isaac_backend.py             # IsaacBackend(RolloutBackend); vla_meta.json sidecar; backend_id
    traces/hdf5_reader.py                # adapter interface + h5py implementation
    traces/hand_closure.py               # configurable GR1 closure scalar
    traces/events.py                     # offline reach / grasp; lift / place from progress.events
    traces/sources.py                    # IsaacTraceSource, IsaacBehaviorSource (implement _load only)
    specs/extract_training_spec.py
    reference/mimic.py                   # RAW reference via generate_dataset.py; G3 -> NotImplementedError
    experiments/{grid,replay_agreement,mode_gap}.py
    replay/recorded_variations.py
    data/{rung_a,rung_b_mimic,mixer,heldout}.py
    data/config/gr1_ranch_bottle_convert.yaml     # copied from A:isaaclab_arena_gr00t/lerobot/config/gr1_manip_ranch_bottle_config.yaml (Apache header kept)
    train/finetune_launcher.py
  bugs/                                   # Phase 8, outside policy-wrapper and the agent mount
    ledger.yaml, inject.py, patches/*.patch
  scripts/gpu/00_check_gpu.sh, R0_setup.sh … R8_vram.sh, R7_dark_cell.sh, R7b_hdr_and_colour_temp.sh,
              mimic_override_check.sh, check_client_tests.sh, collect_measurements.py, run_order.md
  tests/
    contracts/backend_contract.py        # reusable RolloutBackend contract suite (D1), also run against MockBackend
    contracts/source_contract.py         # reusable TraceSource / BehaviorSource contract suite (D1)
    fixtures/                            # FIXTURE-labelled JSONL, HDF5 builders, checkpoint JSONs, Mimic stdout, list_variations stdout
    test_*.py                            # one per module

policy-wrapper/                          # new git repo, one snapshot commit
  NOTICE                                 # source paths + Arena commit 7632eb7d5e, Apache-2.0
  LICENSE                                # Apache-2.0 text (copied files keep their headers)
  pyproject.toml
  config/eval.yaml                       # pilot values (D8/D9/D4 pending your decision)
  policy_wrapper/
    __init__.py                          # empty (no eager imports)
    _arena.py                            # PolicyBase / PolicyCfg / register_policy: Arena's if importable, else stand-ins (D5)
    eval_config.py                       # reads config/eval.yaml `client:` and builds Gr00tClosedloopPolicyCfg
    policy/gr00t_core.py                 # copy; G1 constants import moved into the G1 branch
    policy/gr00t_remote_closedloop_policy.py   # copy; reads eval.yaml client section; seed in request options
    policy/config/gr00t_closedloop_policy_config.py, task_mode.py           # copies
    scheduling/action_scheduler.py, action_chunk_scheduler.py, synced_batch_action_scheduler.py   # copies of A:isaaclab_arena/policy/action_scheduling/*
    utils/image_conversion.py (cv2 import moved into the function), joints_conversion.py, io_utils.py, robot_joints.py   # copies
    embodiments/gr1/gr00t_26dof_joint_space.yaml, 36dof_joint_space.yaml, 54dof_joint_space.yaml,
                    gr1_arms_only_data_config.py, modality.json, info.json   # copies
    serve.py                             # Gr00tPolicy from eval.yaml `server:`
    instrument.py                        # stub in Phase 2, filled in Phase 3
  tests/
    conftest.py                          # fake `gr00t` modules (server_client, embodiment_configs, data.types) as Arena's test does
    test_gr00t_remote_closedloop_policy.py   # copied, re-pointed to GR1 26-DoF + policy_wrapper
    test_wire.py, test_serve.py, test_eval_config.py, test_instrument.py
```
No `isaaclab_arena_g1` copy: the GR1 path never uses it. `utils/robot_eef_pose.py` and `wait_for_gr00t_server.py` are not on the list in `docs/PILOT_AND_ORDER.md`, so I do not copy them.

---

## 3. Where the docs and the code disagree

**D1. There are no reusable contract suites in vla-router.** The prompt says to "reuse its backend, trace and behaviour contract test suites". What exists: `tests/test_backend_cache.py` (tests `MockBackend` and `CachedBackend` directly), `tests/test_trace_code_key.py` (a structural walk over `router` and `tests` only, plus `InMemory*` refusal tests) and `tests/test_stage1_foundations.py`. None of them is parametrised over a backend or source factory. `tests` is also not an installed package (`packages.find` includes only `ladder*` and `router*`).
*Plan:* write `vla-isaac/tests/contracts/{backend,source}_contract.py` as parametrised suites. Their properties come from the vla-router tests and docstrings: one result per seed in order, `CachedBackend` round-trip, a wrong seed set rejected, `load` not overridden, a wrong or missing `code_key` gives `TraceUnavailable`, mode filtering. I also run them against `MockBackend` and `InMemory*Source`, to show they are the same contract. Moving them into vla-router would be a third router edit.

**D2. `policy_runner.py` cannot write the per-step HDF5 or replay recorded variations.** `record_trajectories`, `recorder_dataset_export_dir_path` and `recorded_variation_samples_path` are `ArenaEnvBuilderCfg` fields (`A:isaaclab_arena/environments/arena_env_builder_cfg.py:31-38`), and `policy_runner`'s parser has no flag for them (`dataclass_from_cli` only reads attributes that exist, `A:isaaclab_arena/cli/dataclass_cli.py:64-70`). Without `record_trajectories`, only metrics recorder terms are attached (`arena_env_builder.py:428-433`). The experiment runner sets all three, plus `num_rebuilds` and a `variations:` dict, from a YAML `ArenaRunCfg` (`A:isaaclab_arena/evaluation/{arena_run.py:43-68, run_execution.py:90-172}`). It writes `<out>/<run>/episode_results_rebuild<i>.jsonl` and `dataset_<run>_rebuild<i>` into the same directory.
*Plan:* `arena_cmd.py` builds both forms. Phase 1 defaults to `policy_runner` as the prompt says. `IsaacBackend` switches to the experiment runner (generated YAML) when traces or recorded-variation replay are requested (Phases 3 and 5). The runner kind is a config key. **Decision for you (Q1).**

**D3. `--viz` is not an Arena flag.** Arena's CLI files never define it. Arena's own tests pass `--viz kit` (`A:isaaclab_arena/tests/test_policy_runner.py:45`), so it comes from Isaac Lab's `AppLauncher`, which is unread. `--viz none` gets a VERIFY marker. Settled facts: `--seed` defaults to 42 and `--num_envs` to 1 (`A:isaaclab_arena/cli/isaaclab_arena_cli.py:53-54`), and `--num_episodes` exists (`policy_runner_cli.py:351`).

**D4. The training instruction and the eval instruction differ in Arena's own files.** The closed-loop config says "Place the **sauce bottle** on the top shelf…" (`A:…/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml:6`). The HDF5→LeRobot converter config that made the training data says "Place the **ranch dressing bottle** on the top shelf…" (`A:…/lerobot/config/gr1_manip_ranch_bottle_config.yaml:14`). If the training spec takes `instruction_templates` from the dataset, the pilot's own nominal `eval.yaml` fires `contract_violation` (strength 1.0) on every incident. Which string the checkpoint saw is UNVERIFIED (the dataset's `meta/tasks.jsonl` settles it). **Decision for you (Q2).**
Also: the policy receives `env.get_language_instruction()` (`policy_runner.py:83`), not the YAML value. The YAML value is used only because `PutAndCloseDoorTask` has no `task_description` and `set_task_description(None)` falls back to it. To make `eval.yaml` the single source, `arena_cmd` always passes `--language_instruction <client.language_instruction>` (the flag "takes precedence over the task's own description", `policy_runner_cli.py:357-361`).

**D5. Arena's `PolicyBase` cannot be imported without Isaac Lab.** `isaaclab_arena/policy/__init__.py` eagerly imports `replay_action_policy` and others, which import `isaaclab`. Tested: `import isaaclab_arena.policy.policy_base` gives `ModuleNotFoundError: isaaclab`. `isaaclab_arena.assets.register` does import. A base class cannot be imported lazily, because it is needed when the class is defined.
*Plan:* `policy_wrapper/_arena.py` tries Arena's `PolicyBase` / `PolicyCfg` / `register_policy`. On `ImportError` it uses minimal local stand-ins (`PolicyBase(Generic[CfgT])` with `__init__(config)` and `is_remote=False`; `register_policy` as an identity decorator). This keeps Phase 2 tests GPU-free, so it is not a STOP. I will say so in `BUILD_STATUS.md`. The same applies to `isaaclab_arena.policy.action_scheduling`, which is why the schedulers are copied.

**D6. The variations catalogue docstring is stale.** The example prints `cracker_box.color.sampler.low`, but the field is `sampler_cfg` (`variation_base.py:41`, `light_intensity_variation.py:19`), and the real header is `Variations (Hydra-configurable)`, not the docstring's (`variations_printing.py:38-48` vs `:64`). The parser follows the code: `Asset: <a>`, `  <v> (<Class>, build-time|run-time)`, `    Enable: <a>.<v>.enabled=true  (default: …)`, `      <a>.<v>.<field> = <value>`. Overrides use `sampler_cfg.low/high`, as `docs/PILOT_AND_ORDER.md` R7 already does.

**D7. Actuator overrides have no Arena entry point.** Gains are embodiment code, not variations, so no Hydra override reaches them. A non-Arena path does exist: `--external_environment_class_path module:Class` (`isaaclab_arena_cli.py:140-150`, `isaaclab_arena_environments/cli.py:98-129`).
*Plan:* `vla_isaac/runner/actuator_env.py` subclasses the pilot env factory (`GR1PutAndCloseDoorEnvironment`, name `put_item_in_fridge_and_close_door`) and edits the robot's actuator cfg from a JSON passed via an env var. The Isaac Lab actuator field names are unread, so VERIFY. `IsaacBackend` uses it **only when `actor.kind is POLICY` and `env.actuator_overrides` is non-empty**. Mimic (RAW reference) is launched with the stock registration callback and never gets it (test in Phase 1 and Phase 4). Name collision with the registered env name: VERIFY.

**D8. The pilot `ConfigMapping` fires on the clean pilot config (image size).** `image_size→client.target_image_size` compares `TrainingSpec.image_size` (H, W) with `[512, 512, 3]`. That is a length mismatch, so it is a violation at strength 1.0. `obs_contract` needs `image_size` to be (H, W) (`R:router/stage1/obs_contract.py:251-252`).
*Options:* (a) add `client.target_image_hw: [512, 512]`, mapped, and have the client assert it equals `target_image_size[:2]`; (b) drop `image_size` from the mapping (the obs contract still checks the logged policy-input shape). I recommend (b): no duplicate key that can drift. **Decision for you (Q3).**

**D9. The pilot `ConfigMapping` fires on the clean pilot config (norm stats).** `norm_stats→server.norm_stats_override` compares the checkpoint statistics (a mapping, which `action_contract` needs, `R:router/stage1/action_contract.py:70`) with `null`. That is a violation on nominal.
*Options:* (a) drop `norm_stats` from the mapping. The normalisation bugs are then caught by the action and observation contracts only, which weakens the 6.4 "Code" expectation for the norm bugs. (b) Compare an identity string such as `server.norm_stats_id: checkpoint` instead. This does not work as is: `ConfigMapping` accepts only `FIELD_ORDER` fields, `norm_stats` must stay the real statistics for `action_contract`, and a second field would need `TrainingSpec.extra`, which `cfg_contract` does not read. So (b) needs a third router change. (c) Have `serve.py` write the statistics it actually loaded into the episode outputs, and compare them in vla-isaac outside the router. I recommend (a) plus (c) as evidence for the report. The Phase 2 test then expects one violation per edited **mapped** key. **Decision for you (Q3).**

**D10. Incident-mode episode identity is not recorded.** The k-th-episode stability holds for the k-th episode **to start** (`variation_base.py:129-130`). The JSONL has `env_id` and `episode_in_env` but no global start index (`common_terms.py:131-148`). With `num_envs=1`, k = `episode_in_env`. With N > 1 the start order is not in the record.
*Plan:* clean mode is paired (k = `episode_in_env`). Incident mode assigns seeds to records in a deterministic documented order (`(episode_in_env, env_id)`) and marks `logs["paired"] = False`. Phase 3's wrapper instrumentation logs the start order (the wrapper sees every `reset(env_ids)`), which makes incident-mode pairing exact afterwards. Router seed encoding: `seed = build_seed * stride + k`, with `stride` from config (default 10000). The router pads with small consecutive seeds (`R:router/replay.py:60-63`), so a 30-seed replay is one build. Seed k needs episodes 0..k in that build, and extra episodes are discarded.

**D11. `PHASES` exists in two places.** It is `R:router/prereq/behavior_trace.py:36` and `R:router/vocab.py:50-51`, and `decide.py:402` uses the vocab copy. Change 2 must touch both (see §5).

**D12. `docs/PLAN_UPDATE.md` (d) suggests Arena `TrackedEventCfg` for reach and grasp. The prompt says to compute them offline.** I follow the prompt.

**D13. Version wording.** `A:AGENTS.md` says Isaac Sim 6.2 / Isaac Lab 3.0. `docs/facts_gr00t_n16.md` and `A:docs/…/installation.rst:32` say Isaac Sim 6.0.1. `R0_setup.sh` records `isaacsim` and `isaaclab` versions into `measurements/` instead of assuming either.

**D14. The vla-router placeholder `IsaacBackend` and Rung 1 `kind: isaac`.** `R:ladder/rung1.py:338` raises for `kind: isaac`. Wiring Rung 1's CLI to the real backend would be a third router edit. I leave it alone and note it in `BUILD_STATUS.md` as follow-up (it needs your approval).

**D15. GR00T is under the NVIDIA License, not Apache-2.0** (`G:LICENSE`). I copy no GR00T code. `vla_isaac/backend/server.py` speaks the documented wire format itself: msgpack `{"endpoint": "ping"}`, a dict reply, and an `"error"` key on failure (`G:gr00t/policy/server_client.py:68-142, 186-211`). `serve.py` imports `Gr00tPolicy` and `PolicyServer` at run time only.

**D16. Smaller items.**
- `Gr00tClosedloopPolicyCfg.denoising_steps` and `.seed` exist client-side but nothing in the client reads them (grep). Denoising is server-side only, which matches facts §8.
- The pilot env's default embodiment is `gr1_pink` (`gr1_put_and_close_door_environment.py:38`). The policy needs `gr1_joint`, so it is always passed explicitly.
- `Gr00tPolicy` has `model.action_head.num_inference_timesteps` (`G:gr00t_n1d6.py:45,455`) and `processor.set_statistics(stats, override=True)` (`G:processing_gr00t_n1d6.py:214-220`), which are the hooks `serve.py` uses. The server's `get_action` endpoint forwards `options` (`G:policy.py:65-85`, `server_client.py:217-222`).
- **Per-request seed choice:** the client sends `options={"seed": s}` with `s = hash(config seed, env reset count, request index)`. `serve.py` seeds `torch` from it and falls back to a counter that resets on `reset` when it is absent. Reasons: it survives server restarts, and the seed is logged on the client side next to the step pairs. Arena's client sends no options today. Adding them is part of the policy-wrapper copy, not a bug.
- **Repo state:** the top-level repo has staged changes that are not mine (`.gitmodules`, `external/IsaacLab-Arena`) and moved `PLAN_*.md` files. My phase commits stage only my own paths and leave those alone. `policy-wrapper/` becomes a nested repo, and I add it to the top-level `.gitignore` (**Q5**).

---

## 4. Arena / Isaac Lab / GR00T APIs I depend on but could not read (each becomes a `VERIFY` item)

All of Isaac Lab is unread, because `submodules/IsaacLab` is not checked out.

| id | What | Used by | Command that settles it (GPU box) |
|---|---|---|---|
| V-viz | `--viz none` accepted (AppLauncher) | arena_cmd | `python isaaclab_arena/evaluation/policy_runner.py --help \| grep -A3 -- --viz` |
| V-headless | headless default without `--viz` | arena_cmd | same `--help`; R3 run without a display |
| V-hdf5-keys | HDF5 group/key names written by Isaac Lab `RecorderManager` + `ActionStateRecorderManagerCfg`, `InitialStateRecorderCfg`, `PostStepStatesRecorderCfg`, `PostStepProcessedActionsRecorderCfg`: `data/demo_*/{actions,processed_actions,initial_state,states/articulation/robot/joint_position,…}` | hdf5_reader | `python -c "import h5py;f=h5py.File(P);f.visit(print)"` on one experiment-runner dataset |
| V-hdf5-export | when and where the recorder exports (per reset? only successes? `dataset_export_mode`) | backend, hdf5_reader | run 2 episodes with `record_trajectories: true`; list output dir |
| V-hdf5-policy-runner | whether `policy_runner` writes any HDF5 at all (metrics recorders only) | D2 | R4 output dir listing |
| V-ee-frame | GR1 `get_ee_frame_transformer_names()` and frame names (left/right hand) | hdf5_reader, grasp | `grep` in `A:isaaclab_arena/embodiments/gr1*`, then the HDF5 visit |
| V-obj-state | object pose path in HDF5 states (`states/rigid_object/<name>/root_pose`) and the pick-up object's scene name | hdf5_reader | HDF5 visit |
| V-actuator | Isaac Lab `ArticulationCfg.actuators[<group>]` field names (`stiffness`, `damping`) for GR1, and whether a factory subclass may edit them before build | actuator_env | `python -c` printing `embodiment.scene_config.robot.actuators` in the container |
| V-ext-env-name | an external env class with the same `name` as the registered one is accepted | actuator_env | run R3 with `--external_environment_class_path vla_isaac.runner.actuator_env:PilotEnv` |
| V-mimic-override | `generate_dataset.py` accepts Hydra variation overrides after its args; trial-success line format in stdout | reference/mimic | `scripts/gpu/mimic_override_check.sh` |
| V-mimic-out | Mimic output format ("N/M trials succeeded") | mimic parser | same |
| V-list-var | actual `--list_variations` output for the pilot env (light asset name `light`, variation names) | factors | R3 |
| V-progress | predicate names in `progress.events` for this composite (`object_is_above_height`, `object_on_destination`, door) | events | one R4 JSONL |
| V-jsonl-success | `success` is `None` when no `success` termination term (`common_terms.py:133-135`) | episode_jsonl | one R4 JSONL |
| V-exp-yaml | typed `environment:` keys for `put_item_in_fridge_and_close_door` in experiment YAML (`type`, `object`, `embodiment`, `enable_cameras`) and the `--experiment_config` invocation | arena_cmd | `python isaaclab_arena/evaluation/experiment_runner.py --experiment_config <gen>.yaml` with 1 episode |
| V-timestamped-dir | `timestamped_run_dir` naming under `--output_base_dir` | results | R4 listing |
| V-hf-rev | HF revision hash of the pilot checkpoint | eval.yaml `TODO_PIN` | `hf api` / `huggingface-cli` repo info |
| V-ckpt-cfg | `tune_top_llm_layers`, `num_inference_timesteps`, `action_horizon`, processor crop/resize/jitter in the pilot checkpoint | specs, launcher | R1 `grep` |
| V-train-instr | instruction string in the training dataset's `meta/tasks.jsonl` (D4) | specs | `hf download … --repo-type dataset` of the LeRobot set, `cat meta/tasks.jsonl` |
| V-client-tests | Arena client tests import without Isaac Sim | Phase 2 parity | `scripts/gpu/check_client_tests.sh` |
| V-converter-noisaac | `convert_hdf5_to_lerobot.py` runs without Isaac (imports) | Phase 6 tests | run Arena's test in a plain venv on the GPU box |
| V-finetune-log | trainable-parameter log line format (`G:setup.py:105-110`) | launcher | first fine-tune log |
| V-rgb-order | camera obs is RGB uint8 HWC in `camera_obs` | instrument, FrameStats | one instrument JSON from R4 |
| V-reset-calls | `policy.reset()` call pattern (initial no-arg, then per-env) in multi-env runs | seeds, instrument | instrument log of R5 |
| V-render-defaults | Arena default render settings (for `env.render: {}`) | specs | unknown; record in R0 |

---

## 5. vla-router changes (the two allowed ones)

**Change 1 (ISSUES #19): the cache key includes the backend.**
- `CacheKey` gains `backend: str = ""`. `to_dict` omits it when empty, and `from_dict` defaults it to `""`, so old JSONL caches load unchanged.
- `CachedBackend` reads `getattr(backend, "backend_id", "")`. `RolloutBackend` gets a class attribute `backend_id: str = ""`, and `MockBackend` keeps `""`, so existing caches and tests are unaffected.
- `IsaacBackend.backend_id = "isaac:" + short_hash(arena commit, GR00T commit, runner kind)`.
- New test: two backends with different ids sharing one cache do not see each other's episodes, and an old-format line still loads.
- Note in `R:DECISIONS.md`.

**Change 2: the door subtask.**
- **Recommended:** a single generic optional phase `"post_place"` is appended to both `PHASES` tuples (`behavior_trace.py`, `vocab.py`) and to `PHASES_ACTION_GAP`.
- It is optional by construction: `failure_phase()` only considers phases present in a record. Tasks without it are unchanged, and every existing `PHASES` test still holds, because the tests derive their sets from `V.PHASES`.
- The mapping "composite subtask → phase" is vla-isaac YAML (`phase_map: {close_door: post_place}`), so a DROID pilot needs no router change.
- A door failure with the right implied target reads as an action/skill gap (row 8), which is the plan's definition.
- **Alternative (zero router lines):** fold the door into `place` (place completed = on shelf AND door closed). It is simpler, but the report can no longer tell "dropped" from "door".
- **Decision for you (Q4).**

---

## 6. Phase order and what each test proves

Phases 1-8 as in the prompt, each ending with `pytest` in `vla-isaac`, `policy-wrapper` (from Phase 2) and `vla-router`, a `BUILD_STATUS.md` section, and one commit per touched repo. Specific test shapes:
- Phase 1. Factor mapping:
  - every table entry maps;
  - two factors never share an override path;
  - an unknown factor raises `UnknownFactor`;
  - `restore(i)` removes exactly factor i's override strings.

  Actuator overrides: the `PolicyEnv` path appears in the argv for a POLICY actor and is absent for REFERENCE.
- Phase 2. `cfg_contract` gets one violation per edited mapped key. Also Rung 0 with `test_cmd="python -m pytest -q"` on a materialised copy.
- Phase 3. Synthetic HDF5 is written by the test with h5py, using the key layout from config (so V-hdf5-keys only edits YAML).
- Phase 5. Grid and agreement run on `MockBackend` with `replay_noise` set, which gives a known agreement rate.

---

## 7. Questions before I start

- **Q1 (D2).** OK to use Arena's experiment runner (generated YAML) whenever traces or recorded-variation replay are needed, and `policy_runner` otherwise? Or the experiment runner for everything (one code path)? I recommend **the experiment runner for everything**. It is the only path that records trajectories, it supports `num_rebuilds`, and `policy_runner`'s HDF5 behaviour is unverified. `arena_cmd` still builds the `policy_runner` argv for the R-scripts.
- **Q2 (D4).** Keep `client.language_instruction` as the closed-loop config's "sauce bottle" string, and let Stage 1 flag it if the dataset says "ranch dressing bottle"? Or set it to the training string? I recommend **keeping Arena's eval string** (matching "exactly the pilot values") and settling it with V-train-instr on day one.
- **Q3 (D8, D9).** Drop `image_size` and `norm_stats` from the pilot `ConfigMapping` (recommended), or add the extra keys?
- **Q4 (§5).** `post_place` optional phase (recommended), or fold the door into `place`?
- **Q5.** Python environment for tests: the `vla` conda env (Python 3.12, hagent installed) lacks numpy, torch, h5py, pyzmq, msgpack and opencv. OK to `pip install numpy h5py pyzmq msgpack opencv-python-headless gymnasium` and CPU-only `torch` into it? (These are packages, not models or datasets.) Base miniconda has a CUDA torch and no hagent. Also: put `policy-wrapper/` in the top-level `.gitignore` (recommended), or register it some other way?

---

## 8. Accepted decisions (handoff; supersedes §3-§7 where they differ)

State on 2026-10-27: Phases 1 and 2, router changes 1, 3 and 4 are done (see `docs/BUILD_STATUS.md`, "Current state"). Next is **Phase 3**. Router change 2 is part of it.

### User answers to Q1-Q5
- **Q1 (runner).** Use Arena's experiment runner for every run.
  - `record_trajectories` is a per-call option on `IsaacBackend.run`, **off by default**. It is turned on only when traces are requested.
  - The camera stride is in the backend config (`trajectories.camera_stride`).
  - VERIFY items: **V-runner-parity** (nominal success via `policy_runner` vs the experiment runner on the same seeds; GPU script `R4b_runner_parity.sh`) and **V-exp-remote-policy** (the experiment YAML accepts the remote GR00T policy type, its config path and remote host/port).
  - The user's 390 MB/episode estimate is replaced by **V-traj-size**: the trajectory terms may hold no camera frames (accepted).
- **Q2 (instruction).** Keep "sauce bottle" in `eval.yaml`. Add a day-one GPU script `R4c_instruction_ab.sh`: 20 nominal episodes with each string (eval vs the training string from `meta/tasks.jsonl`), result written to `measurements/`. If the training string scores as well, switch `eval.yaml` to it (V-instruction).
- **Q3 (image_size, norm_stats).** Keep both in the mapping. There is no router change: vla-isaac provides a custom `config_loader` for `WorkspaceManager`, which reads `config/eval.yaml` and adds:
  - `client.target_image_hw` = `target_image_size[:2]`;
  - `server.effective_norm_stats` = the checkpoint's `statistics.json` (resolved from `server.checkpoint`) when `norm_stats_override` is null, otherwise the override file. It has the same structure `extract_training_spec` produces for `TrainingSpec.norm_stats` (`vla_isaac/specs/norm_stats.py`).

  The mapping is `image_size → client.target_image_hw` and `norm_stats → server.effective_norm_stats`. Tests: nominal gives 0 violations; each edited mapped key gives exactly 1; a `norm_stats_override` pointing at modified stats gives 1 `norm_stats` violation. All done in Phase 2.
- **Q4 (door subtask).** Add an optional `post_place` phase (router change 2, Phase 3): appended to both `PHASES` tuples (`router/prereq/behavior_trace.py`, `router/vocab.py`) and to `PHASES_ACTION_GAP`, with the composite-subtask → phase map in vla-isaac YAML. **Required test:** a pick-only `BehaviorTrace` (no `post_place` record) gets the same verdict as before the change.
- **Q5 (environment).** Do not install into the `vla` env (hagent runs there). Use `conda create --name vla-sim --clone vla` with CPU-only torch; the versions are pinned in vla-isaac's `[test]` extra. Done. Always run tests as `env -u PYTHONPATH ~/miniconda3/envs/vla-sim/bin/python -m pytest -q`, because the shell's ROS `PYTHONPATH` breaks pytest. `policy-wrapper/` is in the top-level `.gitignore`.

### Other user instructions from the Phase 0 approval
- Episodes produced beyond the requested seeds (seed k needs episodes 0..k of its build) go into the rollout cache, never discarded. Done in Phase 1.
- Router change 3 (D14) is approved. Done; see below.
- Leave the staged `.gitmodules` / `external/IsaacLab-Arena` changes and the moved `PLAN_*.md` files alone, and never stage them. Commit only your own paths: `git commit --only -- <paths>`.

### Router change 4: G3 unavailable (user's six conditions; done in `e0c88d6`)
1. `ReplayUnavailable(NotImplementedError)` lives in `ladder/backend.py`. `IsaacBackend` raises it for REFERENCE + WRAPPER **before any side effect**: no materialise, no server, no subprocess. This is tested.
2. Audit every call site that requests a reference actor with WRAPPER, and list them in `vla-router/DECISIONS.md`. Catch `ReplayUnavailable` **only at G3**. If any other site uses WRAPPER for the reference, STOP and ask. The audit found none: only G3 at `branch_g.py:37`; G2 and C1 are RAW.
3. G3 skipped is its own state, `unavailable`, never no-effect, restored or failed. `decide()` must not read it as pipeline evidence either way. Implemented as `decide(..., unavailable={"G3"})`, `RouterResult.unavailable` and `ReplayRunner.unavailable`; only `vocab.MAY_BE_UNAVAILABLE = {G3}` may be unavailable.
4. A verdict where G3 is unavailable AND row 4 fired by elimination carries a flag visible in the report: `Verdict.g3_unavailable=True`, `confidence=low (elimination only, G3 unavailable)`, also in `verdict_text` and `PassRecord`.
5. `MockBackend(wrapper_reference_unavailable=True)`. Tests:
   - Branch G with a located Stage 1 contract hit gives the same verdict;
   - Branch G with no contract hit gives code by elimination, low confidence, flag set;
   - any other `NotImplementedError` still ends UNRESOLVED;
   - Branch C is unaffected.
6. A `DECISIONS.md` entry in the existing style, referencing decision 3 (G3 deferred).

### Router change 3 (done in `1ae2385`)
- Rung 1 `kind: isaac` takes `{"factory": "pkg.module:attr[.attr]", "args": {...}}`. vla-router never imports vla-isaac.
- vla-isaac's entry is `vla_isaac.backend.isaac_backend:IsaacBackend.from_config` with `args: {"config": <backend.yaml>}`.
- Tested with a dummy import path. Recorded in `DECISIONS.md`.

### Deviations made in Phases 1-2 (all reported to the user; the first and last were explicitly accepted)
- **`environment_class` in the task config** (`module:Class` of the env factory). The actuated launcher needs it, and Python must not name the pilot. Accepted.
- **The eval config must name the scene's environment.** The task config's `environment_checks` (`type → env.task`, `embodiment → env.embodiment`, `object → env.object`) are checked against the code's `eval.yaml`. A mismatch raises `ConfigError`.
- **Camera stride.** Arena records every step, so the stride is applied by the trace reader and written into every `vla_meta.json`.
- **Cache miss counting (router change 4 side fix).** `CachedBackend` counts hits/misses after the backend returns, so a refused run adds no misses.
- **Location loss.** With G3 unavailable, `code_partial_offset` loses the location `observation_side_or_policy_loading`, which needs G3 restored.
- **Policy registry name** is `policy_wrapper_gr00t_remote_closedloop`, not Arena's `gr00t_remote_closedloop`, which would collide with Arena's own GR00T policy if both were imported.
- **`scheduler`** moved from `Gr00tRemoteClosedloopPolicyCfg` into `eval.yaml` `client:` (with a `scheduler` field on `Gr00tClosedloopPolicyCfg`), so `client:` is the client's single source of truth. `bgr_conversion` is a new config field and is honoured by `resize_rgb_for_policy`.
- **Arena stand-ins.** `policy_wrapper/_arena.py` uses Arena's `PolicyBase` / `PolicyCfg` / `register_policy` when importable and minimal stand-ins otherwise (D5; not a STOP). The `isaaclab_arena_g1` constants are imported only in the G1 branch.
- **`cv2`** is still imported at module level in `utils/image_conversion.py` (IMPL_PLAN said to move it). headless opencv is in the test env.
- **Per-request seed (D16).** The client sends `options={"seed": request_seed(cfg.seed, resets, request index)}` (`policy_wrapper/seeding.py`). `serve.py` seeds torch with it, or with its own counters (restarted by `reset`) when absent, and returns the seed in the reply's info.
- **`serve.py` checkpoint.** It resolves the checkpoint from `--checkpoint-dir`, a local path relative to the repo root, or the local HF cache with `local_files_only` (V-ckpt-cache), and never downloads. `server.checkpoint_revision: TODO_PIN` is an added key (V-hf-rev). None of the pilot doc's comments that name bug homes are copied into `eval.yaml`.
- **`checkpoints:` map** in the backend config (checkpoint id → local dir). The Q3 loader uses it to find `statistics.json`, and `IsaacBackend` passes `--checkpoint-dir` and `--device` to `serve.py`. A checkpoint that cannot be found is a `WorkspaceError`, so a laptop without the checkpoint cannot name policy-wrapper code with this loader. This is intended.
- **New task-config sections.** `eval_config:` (key paths, derived-key names, norm-stats group order) and `config_mapping:` (the pilot `ConfigMapping`).
- **Test fixtures.** The cfg_contract tests build the `TrainingSpec` by hand from the pilot values plus a FIXTURE `statistics.json`; Phase 4 extracts the real spec. policy-wrapper's tests use `tests/fake_gr00t`, stand-ins written for the tests; no GR00T code is copied (D15).
- **V-traj-size** replaces the 390 MB estimate. Accepted.

### Decisions from the Phase 3 review (2026-10-28; before Phase 4)
- **Changed router test accepted.** `vla-router/tests/test_router.py:110` now derives the failure-phase signals from `V.PHASES` instead of listing four literally (router change 2). Accepted as is.
- **Hand limits stay VERIFY, the grasping hand is chosen per episode.** The open / closed finger values remain placeholders under V-hand-limits. `width_hand` is no longer a config choice: the grasping hand is the hand whose closure first crosses the grasp threshold while it is within the reach distance of the target object; ties go to the nearer hand. The chosen hand is recorded with the episode's behaviour trace (small fix a).
- **Router change 5 approved.** `SimHealth.max_penetration` (and any other `SimHealth` field vla-isaac cannot measure) becomes `Optional[float]`, `None` = not measured. The sanity check skips `None` and says "not measured" in the evidence detail; a measured value behaves exactly as before; existing traces with numbers are unchanged. Tests and a `vla-router/DECISIONS.md` entry (small fix b).
- **Stale observations reused by the client are out of scope.** There is no 6.4 bug for them. The client logs a content hash of every image it sends to the server (small fix c), so a reused frame stays detectable later. **Known limitation:** the staleness checks see only action-side delay (step pairs); an observation reused by the client is not evidence for any check today.
- **Phase 4 points.**
  1. `extract_training_spec` and the Q3 config loader use one shared norm-stats function, so `TrainingSpec.norm_stats` and `server.effective_norm_stats` cannot differ in structure. Test: the nominal pilot gives zero `cfg_contract` violations end to end (spec extracted from fixture checkpoint files, `eval.yaml` through the loader).
  2. The checkpoint's modality / processor config wins over Arena's data config; every field that differs between them is listed in the spec's `extra` with both values.
  3. `instruction_templates` comes from the dataset's `meta/tasks.jsonl` when a local path is configured; otherwise it is marked missing, never filled silently from the eval string (V-train-instr).
  4. Mimic reference: RAW only; never the actuated launcher; no downloads (the annotated source dataset path comes from config, VERIFY when absent); one `EpisodeResult` per requested seed in trial order with `logs["paired"] = False` (Mimic trials are not paired with policy seeds; stated in `vla-router/DECISIONS.md`); `reference_unqualified` uses a Wilson lower bound from `ladder/stats.py` against a configurable threshold, not the raw rate.
  5. Same rules as before: one commit per phase per repo; never stage `.gitmodules`, `external/IsaacLab-Arena`, `hybrid-agent` or the moved `PLAN_*.md`; update `BUILD_STATUS.md` ("Current state") and `VERIFY_ON_GPU.md`; stop on the STOP conditions.

### Decisions from the Phase 4 review (2026-10-28; before Phase 5)
- **Mimic qualification uses 50 trials by default**, set in config (`reference.qualification.trials`), not 20. With 20 trials only 20/20 clears a 0.8 threshold at 95 % (the Wilson lower bound of 19/20 is 0.764).
- **Mimic seed (V-mimic-seed) becomes a day-one GPU check**: two runs with different seeds, compare the trial outcomes. The Mimic runner and the qualification code handle both outcomes, with tests:
  - **the seed can be varied per run**: each run gets its own datagen seed, and repeated runs give independent trials, so an interval over the pooled trials is valid;
  - **the seed cannot be varied**: the reference is deterministic. Qualification is a single pass/fail with `logs["deterministic"] = True`, and no interval is reported from repeated runs (they would repeat the same trials).
- **Router change 6 is approved.** `cfg_contract` emits a visible note for every field it skipped because the spec lacked it (`not compared: <field> (missing in spec)`), like change 5 did for `max_penetration`. Nominal verdicts stay unchanged. Tests and a `vla-router/DECISIONS.md` entry; its own commit.
- **Phase 5 additions.**
  1. The grid runner records the reference's success rate per cell next to the policy's (the qualification table of plan 6.1 step 7), using the RAW Mimic reference, and marks a cell `reference_unqualified` with the same rule as Phase 4. Those cells are flagged in the output table.
  2. `replay_agreement` reports, per cell: the agreement rate with its interval, both-success, both-failure and discordant counts, and the overall success rate. A policy that always fails must not look reproducible without that being visible.
  3. `mode_gap` reports the clean and incident nominal rates with Wilson intervals, and whether they differ, using the shared stats module (no reimplementation). Incident-mode pairing is marked unpaired (D10).
  4. `replay/recorded_variations`: stripping one key refuses a file whose remaining keys do not all belong to enabled variations (Arena asserts otherwise), and says which key.
  5. Everything runs on `MockBackend` with known rates and agreement. Tests cover resume after a crash mid-grid, and sharding across two fake hosts giving the same table as one host.
