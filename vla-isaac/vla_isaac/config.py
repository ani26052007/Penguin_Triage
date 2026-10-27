"""Backend and task configuration (YAML).

Two files, both plain data:

* the **backend config** says where things are on this machine (Arena root, GR00T root, the
  Python executables, the output root, the policy server's host and port, timeouts, the
  incident-mode env count, the seed stride, the workspace store and bases);
* the **task config** says which Arena environment a scene id means and how the code under
  test's eval config is read (the pilot is ``config/pilot_gr1_fridge.yaml``). Nothing
  task-specific lives in Python, so switching the pilot to DROID is a new YAML file.

Relative paths are resolved against the directory of the YAML file that names them. Unknown or
missing keys raise ``ConfigError``: a typo must not silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


class ConfigError(ValueError):
    """A backend or task config that cannot be used as written."""


def _strict(d: Mapping, cls, where: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}; known: {sorted(known)}")


def _path(base: Path, value: Any) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def read_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no config file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} does not hold a mapping")
    return data


# --------------------------------------------------------------------------- #
# backend config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ServerConfig:
    """Where the GR00T policy server listens and how long to wait for it."""
    host: str = "127.0.0.1"
    port: int = 5555
    start_timeout_s: float = 600.0          # model load takes minutes
    ping_timeout_ms: int = 2000
    ping_interval_s: float = 2.0
    stop_timeout_s: float = 30.0


@dataclass(frozen=True)
class Timeouts:
    run_s: float = 7200.0                   # one Arena run (one build, all its episodes)
    list_variations_s: float = 900.0


@dataclass(frozen=True)
class TrajectoryConfig:
    """Per-step HDF5 trajectories. Off by default: they are recorded only for calls that ask for
    traces (``IsaacBackend.run(..., record_trajectories=True)``).

    camera_stride: every how many steps a camera frame is kept when traces are read. Arena itself
                   records every step (``A:isaaclab_arena/terms/recorders.py:29-41``), so the
                   stride is applied by the reader and recorded in every run's ``vla_meta.json``.
                   Whether these trajectories hold camera frames at all, and their size, is
                   VERIFY(V-traj-size)."""
    camera_stride: int = 10


@dataclass(frozen=True)
class BackendConfig:
    arena_root: Path
    gr00t_root: Path
    arena_python: str                       # Isaac Sim / Arena interpreter (runs the experiment runner)
    server_python: str                      # GR00T interpreter (runs policy-wrapper's serve.py)
    output_root: Path
    task_config: Path
    workspace_store: Path
    workspace_bases: Mapping[str, Path]
    incident_num_envs: int = 10
    seed_stride: int = 10000                # router seed = build_seed * seed_stride + episode_index
    runner: str = "experiment_runner"       # experiment_runner | policy_runner
    app_flags: tuple[str, ...] = ("--viz", "none")   # VERIFY(V-viz): AppLauncher flag, not in Arena's CLI files
    policy_device: str = "cuda:0"
    checkpoints: Mapping[str, Path] = field(default_factory=dict)   # eval-config checkpoint id -> local dir
    server: ServerConfig = field(default_factory=ServerConfig)
    timeouts: Timeouts = field(default_factory=Timeouts)
    trajectories: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    RUNNERS = ("experiment_runner", "policy_runner")

    def __post_init__(self) -> None:
        if self.runner not in self.RUNNERS:
            raise ConfigError(f"runner must be one of {self.RUNNERS}, got {self.runner!r}")
        if self.incident_num_envs < 2:
            raise ConfigError("incident_num_envs must be >= 2 (clean mode is 1 env)")
        if self.seed_stride < 1:
            raise ConfigError("seed_stride must be >= 1")
        if self.trajectories.camera_stride < 1:
            raise ConfigError("trajectories.camera_stride must be >= 1")


def backend_config_from_dict(d: Mapping, base: Path) -> BackendConfig:
    _strict(d, BackendConfig, "backend config")
    missing = [f.name for f in fields(BackendConfig)
               if f.name not in d and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ConfigError(f"backend config: missing keys {missing}")
    kw = dict(d)
    for key in ("arena_root", "gr00t_root", "output_root", "task_config", "workspace_store"):
        kw[key] = _path(base, kw[key])
    if not isinstance(kw["workspace_bases"], Mapping) or not kw["workspace_bases"]:
        raise ConfigError("backend config: workspace_bases must map labels to checkout directories")
    kw["workspace_bases"] = {str(k): _path(base, v) for k, v in kw["workspace_bases"].items()}
    if "checkpoints" in kw:
        if not isinstance(kw["checkpoints"], Mapping):
            raise ConfigError("backend config: checkpoints must map checkpoint ids to local directories")
        kw["checkpoints"] = {str(k): _path(base, v) for k, v in kw["checkpoints"].items()}
    for key, cls in (("server", ServerConfig), ("timeouts", Timeouts), ("trajectories", TrajectoryConfig)):
        if key in kw:
            _strict(kw[key], cls, f"backend config [{key}]")
            kw[key] = cls(**kw[key])
    if "app_flags" in kw:
        kw["app_flags"] = tuple(str(x) for x in kw["app_flags"])
    return BackendConfig(**kw)


def load_backend_config(path: str | Path) -> BackendConfig:
    path = Path(path).resolve()
    return backend_config_from_dict(read_yaml(path), path.parent)


# --------------------------------------------------------------------------- #
# task config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PolicySection:
    type: str                               # dotted class path handed to Arena (policy.type)
    config_path: str                        # the code's eval config, relative to the working copy


@dataclass(frozen=True)
class ServeSection:
    module: str                             # ``python -m <module>`` inside the working copy
    config_path: str


@dataclass(frozen=True)
class EvalKeys:
    """Dotted paths into the code's eval config (``CodeRef.eval_config``)."""
    language_instruction: str
    actuator_overrides: str


@dataclass(frozen=True)
class EvalConfigSection:
    """How vla-isaac's config loader reads the code's eval config (``vla_isaac.workspace``).

    The loader returns the eval config plus two derived keys, so the router's config contract
    compares like with like (IMPL_PLAN D8, D9; user decision Q3):
      ``derived_image_hw``    = ``image_size_key[:2]``  (the training spec's image size is (H, W));
      ``derived_norm_stats``  = the statistics the server uses: the ``norm_stats_override_key`` file
                                when set, else the checkpoint's ``statistics.json``, flattened per
                                modality in ``norm_stats_groups`` order (``vla_isaac.specs.norm_stats``)."""
    image_size_key: str
    checkpoint_key: str
    norm_stats_override_key: str
    embodiment_key: str
    derived_image_hw: str
    derived_norm_stats: str
    norm_stats_groups: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class TaskConfig:
    nominal_scene: str                      # ``SceneSpec.nominal`` this task config serves
    environment: Mapping[str, Any]          # Arena experiment-YAML ``environment:`` block; ``type`` required
    environment_class: str                  # "module:Class" of that environment's Arena factory
    actuated_environment_type: str          # registered by vla_isaac.arena_ext.actuator_env (policy runs only)
    environment_checks: Mapping[str, str]   # environment key -> eval-config path that must hold the same value
    policy: PolicySection
    serve: ServeSection
    eval_keys: EvalKeys
    eval_config: EvalConfigSection
    config_mapping: Mapping[str, str]       # router ConfigMapping.keys: TrainingSpec field -> eval-config path
    factor_table: Path
    source: Optional[Path] = None

    def __post_init__(self) -> None:
        if "type" not in self.environment:
            raise ConfigError("task config: environment.type is required")
        if ":" not in self.environment_class:
            raise ConfigError("task config: environment_class must be 'module:Class'")
        unknown = set(self.environment_checks) - set(self.environment)
        if unknown:
            raise ConfigError(f"task config: environment_checks name keys not in environment: {sorted(unknown)}")


def task_config_from_dict(d: Mapping, base: Path, source: Optional[Path] = None) -> TaskConfig:
    _strict(d, TaskConfig, "task config")
    need = ("nominal_scene", "environment", "environment_class", "actuated_environment_type", "environment_checks",
            "policy", "serve", "eval_keys", "eval_config", "config_mapping", "factor_table")
    missing = [k for k in need if k not in d]
    if missing:
        raise ConfigError(f"task config: missing keys {missing}")
    for key, cls in (("policy", PolicySection), ("serve", ServeSection), ("eval_keys", EvalKeys),
                     ("eval_config", EvalConfigSection)):
        _strict(d[key], cls, f"task config [{key}]")
    ec = dict(d["eval_config"])
    if not isinstance(ec.get("norm_stats_groups"), Mapping) or not ec["norm_stats_groups"]:
        raise ConfigError("task config [eval_config]: norm_stats_groups must map modalities to group names")
    ec["norm_stats_groups"] = {str(k): tuple(str(g) for g in v) for k, v in ec["norm_stats_groups"].items()}
    if not isinstance(d["config_mapping"], Mapping):
        raise ConfigError("task config: config_mapping must map TrainingSpec fields to eval-config paths")
    return TaskConfig(nominal_scene=str(d["nominal_scene"]), environment=dict(d["environment"]),
                      environment_class=str(d["environment_class"]),
                      actuated_environment_type=str(d["actuated_environment_type"]),
                      environment_checks=dict(d["environment_checks"]),
                      policy=PolicySection(**d["policy"]), serve=ServeSection(**d["serve"]),
                      eval_keys=EvalKeys(**d["eval_keys"]), eval_config=EvalConfigSection(**ec),
                      config_mapping={str(k): str(v) for k, v in d["config_mapping"].items()},
                      factor_table=_path(base, d["factor_table"]),
                      source=source)


def load_task_config(path: str | Path) -> TaskConfig:
    path = Path(path).resolve()
    return task_config_from_dict(read_yaml(path), path.parent, source=path)


def get_dotted(cfg: Mapping, dotted: str) -> tuple[bool, Any]:
    """(found, value) for ``a.b.c`` in nested mappings (same rule as the router's config contract)."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node
