"""Arena command lines and experiment YAML, built as pure functions (same inputs -> same command).

Every backend run uses Arena's experiment runner with a generated one-run YAML (IMPL_PLAN Q1): it
is the only entry point that exposes ``environment_builder.record_trajectories`` and
``recorded_variation_samples_path`` (``A:isaaclab_arena/environments/arena_env_builder_cfg.py:
31-38``; ``A:isaaclab_arena/evaluation/arena_run.py:43-68``). The ``policy_runner.py`` argv is
also built here, for the GPU runbook and the runner-parity check (V-runner-parity).

Experiment YAML layout, from ``A:isaaclab_arena_environments/experiment_configs/
droid_pnp_gr00t_experiment.yaml`` and ``droid_pnp_variations_experiment.yaml``::

    runs:
      <run_name>:
        environment: {type: ..., enable_cameras: true, embodiment: ..., object: ...}
        environment_builder: {seed, num_envs, language_instruction, record_trajectories}
        policy: {type, policy_config_yaml_path, policy_device, remote_host, remote_port}
        rollout_limit: {num_episodes: N}
        variations: {"light.intensity.enabled": true, "light.intensity.sampler_cfg.low": [100.0], ...}

VERIFY(V-exp-yaml): the typed ``environment:`` keys of ``put_item_in_fridge_and_close_door`` and that
a dotted class path is accepted as ``policy.type`` with these remote fields (V-exp-remote-policy).
Results land in ``<experiment_output_directory>/<run_name>/episode_results_rebuild0.jsonl``
(``A:isaaclab_arena/evaluation/run_execution.py:61,117``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from vla_isaac.scene.factors import hydra_override

EXPERIMENT_RUNNER = "isaaclab_arena/evaluation/experiment_runner.py"
POLICY_RUNNER = "isaaclab_arena/evaluation/policy_runner.py"
ACTUATED_LAUNCHER_MODULE = "vla_isaac.arena_ext.launch_experiment"


@dataclass(frozen=True)
class PolicySpec:
    type: str
    policy_config_yaml_path: str
    policy_device: str
    remote_host: str
    remote_port: int

    def to_dict(self) -> dict:
        return {"type": self.type, "policy_config_yaml_path": self.policy_config_yaml_path,
                "policy_device": self.policy_device, "remote_host": self.remote_host,
                "remote_port": self.remote_port}


@dataclass(frozen=True)
class RunSpec:
    """One Arena run: one build (``seed``) and its first ``num_episodes`` episodes."""
    run_name: str
    environment: Mapping[str, Any]
    seed: int
    num_envs: int
    num_episodes: int
    language_instruction: Optional[str]
    policy: PolicySpec
    variations: Mapping[str, Any] = field(default_factory=dict)
    record_trajectories: bool = False
    recorded_variation_samples_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.num_envs < 1 or self.num_episodes < 1:
            raise ValueError("num_envs and num_episodes must be >= 1")
        if "type" not in self.environment:
            raise ValueError("environment.type is required")


def experiment_yaml(spec: RunSpec) -> dict:
    builder: dict[str, Any] = {"seed": spec.seed, "num_envs": spec.num_envs,
                               "record_trajectories": spec.record_trajectories}
    if spec.language_instruction is not None:
        builder["language_instruction"] = spec.language_instruction
    if spec.recorded_variation_samples_path is not None:
        builder["recorded_variation_samples_path"] = spec.recorded_variation_samples_path
    run: dict[str, Any] = {
        "environment": dict(spec.environment),
        "environment_builder": builder,
        "policy": spec.policy.to_dict(),
        "rollout_limit": {"num_episodes": spec.num_episodes},
    }
    if spec.variations:
        run["variations"] = dict(spec.variations)
    return {"runs": {spec.run_name: run}}


def dump_experiment_yaml(spec: RunSpec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(experiment_yaml(spec), sort_keys=True), encoding="utf-8")
    return path


def _needs_cameras(environment: Mapping[str, Any]) -> bool:
    return bool(environment.get("enable_cameras", False))


def experiment_argv(python: str, yaml_path: Path, output_dir: Path, *, environment: Mapping[str, Any],
                    app_flags: tuple[str, ...] = (), actuated: bool = False) -> list[str]:
    """argv run with cwd = Arena root. ``actuated`` uses vla-isaac's launcher, which registers the
    actuator-override environment before handing over to Arena's experiment runner."""
    entry = ["-m", ACTUATED_LAUNCHER_MODULE] if actuated else [EXPERIMENT_RUNNER]
    argv = [python, *entry, "--experiment_config", str(yaml_path), "--experiment_output_directory", str(output_dir)]
    if _needs_cameras(environment):
        # The experiment runner asserts the CLI flag when an env asks for cameras
        # (A:isaaclab_arena/evaluation/experiment_runner.py:53-60).
        argv.append("--enable_cameras")
    argv.extend(app_flags)                    # VERIFY(V-viz): e.g. --viz none comes from Isaac Lab's AppLauncher
    return argv


def policy_runner_argv(python: str, spec: RunSpec, output_base_dir: Path, *,
                       app_flags: tuple[str, ...] = ()) -> list[str]:
    """``policy_runner.py`` argv: policy flags, the env sub-command, then the variation overrides
    (``docs/PILOT_AND_ORDER.md`` R4/R7). The policy flags are the policy config's field names
    (``A:isaaclab_arena/evaluation/policy_runner_cli.py:299-307``)."""
    if spec.record_trajectories or spec.recorded_variation_samples_path:
        raise ValueError("policy_runner.py has no flag for trajectories or recorded variations (IMPL_PLAN D2)")
    p = spec.policy
    argv = [python, POLICY_RUNNER, *app_flags,
            "--seed", str(spec.seed),
            "--policy_type", p.type,
            "--policy_config_yaml_path", p.policy_config_yaml_path,
            "--policy_device", p.policy_device,
            "--remote_host", p.remote_host,
            "--remote_port", str(p.remote_port),
            "--num_episodes", str(spec.num_episodes),
            "--num_envs", str(spec.num_envs),
            "--output_base_dir", str(output_base_dir)]
    if _needs_cameras(spec.environment):
        argv.append("--enable_cameras")
    if spec.language_instruction is not None:
        argv += ["--language_instruction", spec.language_instruction]
    env = dict(spec.environment)
    argv.append(str(env.pop("type")))
    env.pop("enable_cameras", None)
    for key in sorted(env):
        argv += [f"--{key}", str(env[key])]
    argv += [hydra_override(k, v) for k, v in spec.variations.items()]
    return argv
