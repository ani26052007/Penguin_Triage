from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vla_isaac.runner.arena_cmd import PolicySpec, RunSpec, experiment_argv, experiment_yaml, policy_runner_argv

POL = PolicySpec("policy_wrapper.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy",
                 "/w/config/eval.yaml", "cuda:0", "127.0.0.1", 5555)
ENV = {"type": "put_item_in_fridge_and_close_door", "embodiment": "gr1_joint", "object": "ranch_dressing_hope_robolab",
       "enable_cameras": True}
VAR = {"light.intensity.enabled": True, "light.intensity.sampler_cfg.low": [100.0],
       "light.intensity.sampler_cfg.high": [100.0]}


def spec(**kw):
    base = dict(run_name="vla_b0", environment=ENV, seed=0, num_envs=1, num_episodes=20,
                language_instruction="Place the sauce bottle.", policy=POL, variations=VAR)
    base.update(kw)
    return RunSpec(**base)


def test_policy_runner_argv_matches_the_runbook_shape():
    argv = policy_runner_argv("python", spec(), Path("outputs/x"), app_flags=("--viz", "none"))
    assert argv == ["python", "isaaclab_arena/evaluation/policy_runner.py", "--viz", "none", "--seed", "0",
                    "--policy_type", POL.type, "--policy_config_yaml_path", "/w/config/eval.yaml",
                    "--policy_device", "cuda:0", "--remote_host", "127.0.0.1", "--remote_port", "5555",
                    "--num_episodes", "20", "--num_envs", "1", "--output_base_dir", "outputs/x", "--enable_cameras",
                    "--language_instruction", "Place the sauce bottle.",
                    "put_item_in_fridge_and_close_door", "--embodiment", "gr1_joint",
                    "--object", "ranch_dressing_hope_robolab",
                    "light.intensity.enabled=true", "light.intensity.sampler_cfg.low=[100.0]",
                    "light.intensity.sampler_cfg.high=[100.0]"]


def test_policy_runner_cannot_record_trajectories():
    with pytest.raises(ValueError, match="D2"):
        policy_runner_argv("python", spec(record_trajectories=True), Path("o"))


def test_experiment_yaml_and_argv_are_deterministic():
    a, b = experiment_yaml(spec()), experiment_yaml(spec())
    assert a == b and yaml.safe_dump(a, sort_keys=True) == yaml.safe_dump(b, sort_keys=True)
    argv = experiment_argv("py", Path("/o/experiment.yaml"), Path("/o/arena"), environment=ENV, app_flags=("--viz", "none"))
    assert argv == ["py", "isaaclab_arena/evaluation/experiment_runner.py", "--experiment_config", "/o/experiment.yaml",
                    "--experiment_output_directory", "/o/arena", "--enable_cameras", "--viz", "none"]
    no_cam = experiment_argv("py", Path("y"), Path("o"), environment={"type": "t"})
    assert "--enable_cameras" not in no_cam


def test_recorded_variation_replay_path_goes_into_the_builder_block():
    y = experiment_yaml(spec(recorded_variation_samples_path="rec.jsonl"))
    assert y["runs"]["vla_b0"]["environment_builder"]["recorded_variation_samples_path"] == "rec.jsonl"
    assert "variations" not in experiment_yaml(spec(variations={}))["runs"]["vla_b0"]
