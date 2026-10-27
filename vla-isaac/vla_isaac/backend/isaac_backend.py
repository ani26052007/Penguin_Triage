"""``IsaacBackend``: vla-router's ``RolloutBackend`` over Isaac Lab Arena and the GR00T server.

``run(scene, code_ref, actor, seeds, mode)``, for a policy actor:

  1. materialise ``code_ref`` with the ``WorkspaceManager`` (the working copy is the code under
     test: policy-wrapper with its ``config/eval.yaml``);
  2. start that copy's ``serve.py`` and wait for ``ping``;
  3. per build seed, run Arena's experiment runner on a generated one-run YAML with
     ``PYTHONPATH`` = the working copy (so ``policy.type`` resolves to the copy's client);
  4. parse ``episode_results_rebuild0.jsonl`` into ``EpisodeResult``s;
  5. stamp ``code_key`` = ``code_ref.key()`` into every result's logs and into a ``vla_meta.json``
     sidecar next to the outputs (trace sources refuse outputs without it, ISSUES #52).

Traces (``record_trajectories=True``): Arena records its HDF5 trajectories and the policy wrapper's
instrument writes its per-episode files (``VLA_INSTRUMENT_*`` in the run's environment) into the
build's ``instrument/`` directory. ``logs["paths"]["run_root"]`` is the incident log reference for the
mode (``vla_isaac.traces.sources``).

Modes: clean = 1 env, incident = ``BackendConfig.incident_num_envs`` envs sharing the server.
Seeds: ``vla_isaac.seeds`` (build seed, episode index); results come back in the caller's order.
Output: each call writes under ``runs/<hash>`` of (code, actor, scene, mode, backend, traces), wiped
before each build. ``run_tag`` adds itself to that key, so two calls with different tags (the two runs
of ``vla_isaac.experiments.replay_agreement``) keep separate directories and both runs' raw files.
Seed k of a build needs episodes 0..k; the extra episodes go into ``cache`` (the ``RolloutCache``
the router's ``CachedBackend`` wraps), keyed with this backend's ``backend_id``.

Actuator overrides (``env.actuator_overrides`` in the eval config) are code under test and apply
to policy actors only. Reference actors never get them: RAW reference runs go to the reference
runner (``vla_isaac.reference.mimic.MimicReference`` by default when the task config has a
``reference:`` section), which receives the scene's variations and nothing from the code's eval
config, and a WRAPPER reference (G3) raises ``ReplayUnavailable`` (decision 3, router change 4) before
anything is built, started or run.

Errors: a crashed or timed-out Arena run raises ``ArenaRunError``; bad records raise
``EpisodeResultsError``; the incident loop turns either into UNRESOLVED (``R:ladder/loop.py:313``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from ladder.backend import ReplayUnavailable, RolloutBackend, wrapper_reference
from ladder.workspace import WorkspaceManager
from router.prereq.rollout_cache import CacheKey, RolloutCache
from router.schemas import (ActorKind, ActorRef, CodeRef, EpisodeResult, Mode, SceneSpec,
                            short_hash)

from vla_isaac.arena_ext.actuator_env import ENV_BASE, ENV_NAME, ENV_OVERRIDES
from vla_isaac.backend.server import PolicyServerProcess
from vla_isaac.config import (BackendConfig, ConfigError, TaskConfig, get_dotted, load_backend_config,
                              load_task_config)
from vla_isaac.results.episode_jsonl import (EpisodeResultsError, find_result_files, index_records, read_records,
                                             to_episode_result)
from vla_isaac.runner.arena_cmd import (PolicySpec, RunSpec, dump_experiment_yaml, experiment_argv,
                                        policy_runner_argv)
from vla_isaac.runner.subprocess_runner import ArenaRunner, RunInvocation, SubprocessRunner
from vla_isaac.scene.factors import FactorTable
from vla_isaac.seeds import SeedPlan, join_seed, plan_seeds
from vla_isaac.traces import instrument_reader as instrument

META_FILE = "vla_meta.json"
VLA_ISAAC_ROOT = Path(__file__).resolve().parents[2]


class ReferenceRunner(Protocol):
    """Runs the RAW reference controller (``vla_isaac.reference.mimic.MimicReference``)."""

    def run(self, scene: SceneSpec, actor: ActorRef, seeds: Sequence[int], *,
            variations: Mapping[str, Any]) -> list[EpisodeResult]: ...


ServerFactory = Callable[..., PolicyServerProcess]


@dataclass(frozen=True)
class PlannedRun:
    """Everything one Arena run needs; a pure function of the backend's inputs."""
    spec: RunSpec
    invocation: RunInvocation
    run_dir: Path                        # where the episode JSONL and the HDF5 land
    actuated: bool


def _git_head(path: Path) -> str:
    try:
        p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "nogit"
    return p.stdout.strip() if p.returncode == 0 else "nogit"


def _file_bytes(path: Optional[Path]) -> str:
    return Path(path).read_text(encoding="utf-8") if path and Path(path).is_file() else ""


def backend_id_for(cfg: BackendConfig, task: TaskConfig) -> str:
    """Names what produces an episode: Arena and GR00T commits, runner, task config and factor table."""
    return "isaac:" + short_hash({"arena": _git_head(cfg.arena_root), "gr00t": _git_head(cfg.gr00t_root),
                                  "runner": cfg.runner, "task": _file_bytes(task.source),
                                  "factors": _file_bytes(task.factor_table)}, 12)


class IsaacBackend(RolloutBackend):
    def __init__(self, cfg: BackendConfig, task: TaskConfig, workspaces: WorkspaceManager, *,
                 factors: Optional[FactorTable] = None, runner: Optional[ArenaRunner] = None,
                 server_factory: Optional[ServerFactory] = None, cache: Optional[RolloutCache] = None,
                 reference: Optional[ReferenceRunner] = None, backend_id: Optional[str] = None) -> None:
        self.cfg, self.task, self.workspaces = cfg, task, workspaces
        self.factors = factors if factors is not None else FactorTable.load(task.factor_table)
        self.runner = runner or SubprocessRunner()
        self.server_factory = server_factory or PolicyServerProcess
        self.cache = cache
        if reference is None and task.reference is not None:
            from vla_isaac.reference.mimic import MimicReference

            reference = MimicReference(cfg, task)
        self.reference = reference
        self.backend_id = backend_id or backend_id_for(cfg, task)

    @classmethod
    def from_config(cls, config: str | Path, **kw: Any) -> "IsaacBackend":
        """The backend for a backend-config YAML: its task config, and policy-wrapper bases with the
        eval-config loader of ``vla_isaac.workspace``. Rung 1's ``kind: isaac`` names this
        (``"vla_isaac.backend.isaac_backend:IsaacBackend.from_config"``, args ``{"config": path}``)."""
        from vla_isaac.workspace import make_workspaces

        cfg = load_backend_config(config)
        task = load_task_config(cfg.task_config)
        return cls(cfg, task, make_workspaces(cfg, task), **kw)

    # -- the RolloutBackend call ------------------------------------------------
    def run(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef, seeds: Sequence[int], mode: Mode,
            *, record_trajectories: bool = False, run_tag: Optional[str] = None) -> list[EpisodeResult]:
        if wrapper_reference(actor):
            raise ReplayUnavailable("the reference controller through the wrapper (G3) is deferred "
                                    "(decision 3): no joint-space reference exists")
        mode = Mode(mode)
        plan = plan_seeds([int(s) for s in seeds], self.cfg.seed_stride)
        self._check_scene(scene)
        if actor.kind is ActorKind.REFERENCE:
            return self._run_reference(scene, actor, plan)

        workdir = self._materialize(code_ref)
        produced: dict[int, EpisodeResult] = {}
        server = self._server(workdir, code_ref)
        server.start()
        try:
            for build_seed in plan.builds:
                if not server.ping():
                    server.restart()
                planned = self.plan_run(scene, code_ref, actor, mode, build_seed, plan.episodes_needed(build_seed),
                                        workdir=workdir, record_trajectories=record_trajectories, run_tag=run_tag)
                produced.update(self._execute(planned, scene, code_ref, actor, mode, build_seed,
                                              server_log=server.log_path))
        finally:
            server.stop()
        self._cache_extras(produced, plan, scene, code_ref, actor, mode)
        return plan.in_caller_order(produced)

    # -- planning (pure) ----------------------------------------------------------
    def plan_run(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef, mode: Mode, build_seed: int,
                 num_episodes: int, *, workdir: Path, record_trajectories: bool = False,
                 run_tag: Optional[str] = None) -> PlannedRun:
        """The Arena run for one build. Actuator overrides are planned only for a policy actor."""
        mode = Mode(mode)
        cfg, task = self.cfg, self.task
        ev = code_ref.eval_config
        instruction = self._eval(ev, task.eval_keys.language_instruction)
        self._check_environment(ev)
        overrides = self.actuator_overrides(actor, ev)
        environment = dict(task.environment)
        if overrides:
            environment["type"] = task.actuated_environment_type
        policy = PolicySpec(type=task.policy.type, policy_config_yaml_path=str(Path(workdir) / task.policy.config_path),
                            policy_device=cfg.policy_device, remote_host=cfg.server.host, remote_port=cfg.server.port)
        spec = RunSpec(run_name=f"vla_b{build_seed}", environment=environment, seed=build_seed,
                       num_envs=1 if mode is Mode.CLEAN else cfg.incident_num_envs, num_episodes=num_episodes,
                       language_instruction=str(instruction), policy=policy,
                       variations=self.factors.assignments(scene), record_trajectories=record_trajectories)
        out = self._run_root(scene, code_ref, actor, mode, record_trajectories, run_tag) / f"build{build_seed}_n{num_episodes}"
        yaml_path = out / "experiment.yaml"
        if cfg.runner == "experiment_runner":
            argv = experiment_argv(cfg.arena_python, yaml_path, out / "arena", environment=environment,
                                   app_flags=cfg.app_flags, actuated=bool(overrides))
            run_dir = out / "arena" / spec.run_name
        else:
            if overrides:
                raise ConfigError("actuator overrides need the experiment runner (runner: experiment_runner)")
            argv = policy_runner_argv(cfg.arena_python, spec, out / "arena", app_flags=cfg.app_flags)
            run_dir = out / "arena"                 # VERIFY(V-timestamped-dir): a timestamped subdir is added
        env = {"PYTHONPATH": os.pathsep.join([str(workdir), str(VLA_ISAAC_ROOT)])}
        if record_trajectories:
            env.update(self.instrument_env(out / "instrument"))
        if overrides:
            env.update({ENV_OVERRIDES: json.dumps(overrides, sort_keys=True), ENV_BASE: task.environment_class,
                        ENV_NAME: task.actuated_environment_type})
        inv = RunInvocation(argv=tuple(argv), cwd=cfg.arena_root, env=env, timeout_s=cfg.timeouts.run_s,
                            log_path=out / "arena.log", output_dir=out / "arena", spec=spec)
        return PlannedRun(spec, inv, run_dir, bool(overrides))

    def instrument_env(self, directory: Path) -> dict[str, str]:
        """The policy wrapper's instrumentation settings for a run that records traces."""
        t = self.cfg.trajectories
        keys = self.task.traces.instrument_obs_keys if self.task.traces is not None else ()
        return {instrument.ENV_DIR: str(directory), instrument.ENV_FRAME_EVERY: str(t.camera_stride),
                instrument.ENV_PROFILE_BUCKETS: str(t.profile_buckets), instrument.ENV_OBS_KEYS: ",".join(keys)}

    def actuator_overrides(self, actor: ActorRef, eval_config: Mapping[str, Any]) -> dict:
        """The code's actuator overrides, for a policy actor; ``{}`` for any reference actor."""
        if actor.kind is not ActorKind.POLICY:
            return {}
        found, value = get_dotted(eval_config, self.task.eval_keys.actuator_overrides)
        if not found or value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigError(f"eval config {self.task.eval_keys.actuator_overrides} must be a mapping, got {value!r}")
        return json.loads(json.dumps(value))

    # -- steps --------------------------------------------------------------------
    def _execute(self, planned: PlannedRun, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef, mode: Mode,
                 build_seed: int, *, server_log: Path) -> dict[int, EpisodeResult]:
        inv, spec = planned.invocation, planned.spec
        out = inv.log_path.parent
        if out.exists():
            shutil.rmtree(out)                      # a fresh directory per run: no stale results
        out.mkdir(parents=True)
        yaml_path = dump_experiment_yaml(spec, out / "experiment.yaml")
        self.runner.run(inv)
        files = find_result_files(planned.run_dir) if self.cfg.runner == "experiment_runner" else \
            sorted(planned.run_dir.rglob("episode_results_rank*.jsonl"))
        if not files:
            raise EpisodeResultsError(f"no episode_results_rank*.jsonl under {planned.run_dir}")
        records = [r for f in files for r in read_records(f)]
        by_k = index_records(records, num_envs=spec.num_envs, num_episodes=spec.num_episodes, build_seed=build_seed)
        meta = {"code_key": code_ref.key(), "code_hash": code_ref.code_hash, "backend_id": self.backend_id,
                "scene": scene.key(), "actor": actor.key(), "mode": mode.value, "build_seed": build_seed,
                "num_envs": spec.num_envs, "num_episodes": spec.num_episodes,
                "record_trajectories": spec.record_trajectories,
                "camera_stride": self.cfg.trajectories.camera_stride,
                "profile_buckets": self.cfg.trajectories.profile_buckets, "seed_stride": self.cfg.seed_stride,
                "instrument_dir": inv.env.get(instrument.ENV_DIR), "actuated": planned.actuated,
                "argv": list(inv.argv), "experiment_yaml": str(yaml_path), "run_dir": str(planned.run_dir),
                "results": [str(f) for f in files]}
        meta_path = out / META_FILE
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        paired = spec.num_envs == 1
        extra = {"backend": self.backend_id, "code_key": code_ref.key(), "mode": mode.value,
                 "record_trajectories": spec.record_trajectories,
                 "paths": {"vla_meta": str(meta_path), "run_dir": str(planned.run_dir), "log": str(inv.log_path),
                           "server_log": str(server_log), "experiment_yaml": str(yaml_path),
                           "run_root": str(out.parent)}}
        return {join_seed(build_seed, k, self.cfg.seed_stride):
                to_episode_result(rec, seed=join_seed(build_seed, k, self.cfg.seed_stride), episode_index=k,
                                  paired=paired, extra_logs=extra)
                for k, rec in by_k.items()}

    def _cache_extras(self, produced: Mapping[int, EpisodeResult], plan: SeedPlan, scene: SceneSpec,
                      code_ref: CodeRef, actor: ActorRef, mode: Mode) -> None:
        if self.cache is None:
            return
        asked = set(plan.seeds)
        for seed, ep in sorted(produced.items()):
            if seed not in asked:
                self.cache.put(CacheKey.make(scene, code_ref, actor, seed, mode, self.backend_id), ep)

    def _run_reference(self, scene: SceneSpec, actor: ActorRef, plan: SeedPlan) -> list[EpisodeResult]:
        if self.reference is None:
            raise NotImplementedError(f"task config {self.task.source} has no reference section, so RAW reference "
                                      f"runs have no runner (vla_isaac.reference.mimic)")
        # The reference runs the scene's variations and nothing from the code's eval config.
        return self.reference.run(scene, actor, plan.seeds, variations=self.factors.assignments(scene))

    def _server(self, workdir: Path, code_ref: CodeRef) -> PolicyServerProcess:
        s, cfg = self.task.serve, self.cfg
        argv = [cfg.server_python, "-m", s.module, "--config", s.config_path,
                "--host", cfg.server.host, "--port", str(cfg.server.port), "--device", cfg.policy_device]
        found, checkpoint = get_dotted(code_ref.eval_config, self.task.eval_config.checkpoint_key)
        if found and str(checkpoint) in cfg.checkpoints:
            argv += ["--checkpoint-dir", str(cfg.checkpoints[str(checkpoint)])]
        log = cfg.output_root / "servers" / f"{short_hash(code_ref.key(), 12)}.log"
        return self.server_factory(argv, cwd=workdir, env={"PYTHONPATH": str(workdir)}, cfg=cfg.server, log_path=log)

    def _materialize(self, code_ref: CodeRef) -> Path:
        dest = self.cfg.output_root / "work" / short_hash(code_ref.key(), 12)
        if dest.exists():
            shutil.rmtree(dest)
        return self.workspaces.materialize(code_ref, dest)

    def _run_root(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef, mode: Mode, traj: bool,
                  run_tag: Optional[str] = None) -> Path:
        key = {"code": code_ref.key(), "actor": actor.key(), "scene": scene.key(), "mode": Mode(mode).value,
               "backend": self.backend_id, "traj": traj}
        if run_tag is None:
            return self.cfg.output_root / "runs" / short_hash(key, 16)
        if not run_tag or not all(c.isalnum() or c in "-_." for c in run_tag):
            raise ConfigError(f"run_tag {run_tag!r} must be a non-empty name of letters, digits, '-', '_' or '.'")
        return self.cfg.output_root / "runs" / f"{short_hash({**key, 'tag': run_tag}, 16)}_{run_tag}"

    # -- checks -------------------------------------------------------------------
    def _check_scene(self, scene: SceneSpec) -> None:
        if scene.nominal != self.task.nominal_scene:
            raise ConfigError(f"scene {scene.nominal!r} is not served by task config {self.task.source} "
                              f"(nominal_scene {self.task.nominal_scene!r})")
        self.factors.assignments(scene)            # unknown factors raise before anything runs

    def _check_environment(self, eval_config: Mapping[str, Any]) -> None:
        for key, path in self.task.environment_checks.items():
            got = self._eval(eval_config, path)
            if got != self.task.environment[key]:
                raise ConfigError(f"eval config {path} = {got!r}, but the scene's environment has "
                                  f"{key} = {self.task.environment[key]!r}")

    @staticmethod
    def _eval(eval_config: Mapping[str, Any], path: str) -> Any:
        found, value = get_dotted(eval_config, path)
        if not found:
            raise ConfigError(f"eval config has no {path!r}")
        return value
