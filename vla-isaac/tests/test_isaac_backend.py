"""IsaacBackend on fakes: seeds, commands, code key, actuator overrides, errors, the extra-episode cache."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from ladder.backend import ReplayUnavailable
from ladder.loop import IncidentLoop, LoopConfig, LoopStatus, StubDataLadder
from ladder.rung1 import Rung1Config, staged_plan_for
from router.prereq.rollout_cache import CacheKey, CachedBackend, RolloutCache
from router.replay import ReplayConfig
from router.router import RouterConfig, RouterInputs
from router.schemas import EpisodeResult, Incident, Mode, SceneSpec, TrainingSpec

from vla_isaac.arena_ext.actuator_env import ENV_BASE, ENV_NAME, ENV_OVERRIDES
from vla_isaac.backend.isaac_backend import META_FILE
from vla_isaac.config import ConfigError
from vla_isaac.results.episode_jsonl import EpisodeResultsError
from vla_isaac.runner.subprocess_runner import ArenaRunError, FakeArenaRunner
from vla_isaac.scene.factors import UnknownFactor

from tests.helpers import DARK, NOMINAL, POLICY, REF_RAW, REF_WRAP, make_backend, with_eval

GAINS = {"arms": {"stiffness": 40.0, "damping": 4.0}}


def test_seed_order_is_preserved_across_builds(tmp_path):
    rig = make_backend(tmp_path)
    seeds = [20003, 5, 10001, 2, 10000]
    out = rig.backend.run(NOMINAL, rig.code(), POLICY, seeds, Mode.CLEAN)
    assert [e.seed for e in out] == seeds
    assert [(e.logs["build_seed"], e.logs["episode_index"]) for e in out] == [(2, 3), (0, 5), (1, 1), (0, 2), (1, 0)]
    runs = [(c.spec.seed, c.spec.num_episodes) for c in rig.runner.calls]
    assert runs == [(0, 6), (1, 2), (2, 4)]               # one run per build, episodes 0..max(k)


def test_one_server_per_call_started_and_stopped_from_the_working_copy(tmp_path):
    rig = make_backend(tmp_path)
    rig.backend.run(NOMINAL, rig.code(), POLICY, [0, 10000], Mode.CLEAN)
    (s,) = rig.servers
    assert s.events == ["start", "stop"]
    assert s.argv[1:3] == ["-m", "policy_wrapper.serve"] and (Path(s.cwd) / "config" / "eval.yaml").is_file()
    assert s.env["PYTHONPATH"] == str(s.cwd)
    assert s.argv[3:] == ["--config", "config/eval.yaml", "--host", "127.0.0.1", "--port", "5555", "--device", "cuda:0"]


def test_the_server_gets_the_local_checkpoint_dir_from_the_backend_config(tmp_path):
    ckpt = "org/repo/sub"
    rig = make_backend(tmp_path, eval_config=with_eval(server={"checkpoint": ckpt}),
                       checkpoints={ckpt: tmp_path / "ckpt"})
    rig.backend.run(NOMINAL, rig.code(), POLICY, [0], Mode.CLEAN)
    assert rig.servers[0].argv[-2:] == ["--checkpoint-dir", str(tmp_path / "ckpt")]
    other = make_backend(tmp_path / "o", eval_config=with_eval(server={"checkpoint": "elsewhere"}),
                         checkpoints={ckpt: tmp_path / "ckpt"})
    other.backend.run(NOMINAL, other.code(), POLICY, [0], Mode.CLEAN)
    assert "--checkpoint-dir" not in other.servers[0].argv


def test_a_server_that_stops_answering_is_restarted_between_builds(tmp_path):
    rig = make_backend(tmp_path)

    class Flaky(FakeArenaRunner):
        def run(self, inv):
            super().run(inv)
            rig.servers[0].alive = False
    rig.backend.runner = Flaky()
    rig.backend.run(NOMINAL, rig.code(), POLICY, [0, 10000], Mode.CLEAN)
    assert rig.servers[0].events == ["start", "restart", "stop"]


def test_clean_and_incident_modes(tmp_path):
    rig = make_backend(tmp_path)
    clean = rig.backend.run(NOMINAL, rig.code(), POLICY, [0, 1, 2], Mode.CLEAN)
    inc = rig.backend.run(NOMINAL, rig.code(), POLICY, [0, 1, 2, 3, 4, 5], Mode.INCIDENT)
    assert [c.spec.num_envs for c in rig.runner.calls] == [1, 4]
    assert all(e.logs["paired"] for e in clean) and not any(e.logs["paired"] for e in inc)
    assert {e.logs["env_id"] for e in inc} == {0, 1, 2, 3}


def test_code_key_is_stamped_in_logs_and_sidecar(tmp_path):
    rig = make_backend(tmp_path)
    code = rig.code()
    (ep,) = rig.backend.run(DARK, code, POLICY, [3], Mode.CLEAN)
    assert ep.logs["code_key"] == code.key() and ep.logs["backend"] == "isaac:test"
    meta = json.loads(Path(ep.logs["paths"]["vla_meta"]).read_text())
    assert meta["code_key"] == code.key() and meta["scene"] == DARK.key() and meta["mode"] == "clean"
    assert meta["record_trajectories"] is False and meta["camera_stride"] == 10
    assert Path(ep.logs["paths"]["vla_meta"]).name == META_FILE
    assert ep.logs["variations"] == {"light.intensity": [100.0]}       # kept in logs
    assert "progress" in ep.logs and "placement" in ep.logs


def test_logs_keep_the_raw_files_and_progress(tmp_path):
    rig = make_backend(tmp_path)
    (ep,) = rig.backend.run(NOMINAL, rig.code(), POLICY, [0], Mode.CLEAN)
    p = ep.logs["paths"]
    assert Path(p["episode_results"]).is_file() and Path(p["experiment_yaml"]).is_file()
    assert ep.logs["progress"]["events"][0]["predicate_name"] == "object_is_above_height"


def test_same_inputs_give_the_same_command(tmp_path):
    a = make_backend(tmp_path / "a")
    work = tmp_path / "w"
    p1 = a.backend.plan_run(DARK, a.code(), POLICY, Mode.CLEAN, 3, 12, workdir=work)
    p2 = a.backend.plan_run(DARK, a.code(), POLICY, Mode.CLEAN, 3, 12, workdir=work)
    assert p1 == p2
    a.backend.run(DARK, a.code(), POLICY, [5], Mode.CLEAN)
    a.backend.run(DARK, a.code(), POLICY, [5], Mode.CLEAN)
    c1, c2 = a.runner.calls
    assert c1.argv == c2.argv and c1.env == c2.env
    assert (c1.output_dir.parent / "experiment.yaml").read_text() == (c2.output_dir.parent / "experiment.yaml").read_text()


def test_experiment_yaml_carries_scene_seed_envs_and_instruction(tmp_path):
    rig = make_backend(tmp_path)
    rig.backend.run(DARK, rig.code(), POLICY, [10002], Mode.INCIDENT)
    inv = rig.runner.calls[0]
    doc = yaml.safe_load((inv.output_dir.parent / "experiment.yaml").read_text())
    run = doc["runs"]["vla_b1"]
    assert run["environment_builder"] == {"seed": 1, "num_envs": 4, "record_trajectories": False,
                                          "language_instruction": with_eval()["client"]["language_instruction"]}
    assert run["rollout_limit"] == {"num_episodes": 3}
    assert run["variations"] == {"light.intensity.enabled": True, "light.intensity.sampler_cfg.low": [100.0],
                                 "light.intensity.sampler_cfg.high": [100.0]}
    assert run["environment"]["type"] == "put_item_in_fridge_and_close_door"
    assert run["policy"]["policy_config_yaml_path"].endswith("config/eval.yaml")
    assert inv.argv[1] == "isaaclab_arena/evaluation/experiment_runner.py" and "--enable_cameras" in inv.argv
    workdir = Path(inv.env["PYTHONPATH"].split(":")[0])               # the code under test comes first
    assert (workdir / "config" / "eval.yaml").is_file() and run["policy"]["policy_config_yaml_path"].startswith(str(workdir))
    assert ENV_OVERRIDES not in inv.env


def test_record_trajectories_is_per_call_and_off_by_default(tmp_path):
    rig = make_backend(tmp_path)
    rig.backend.run(NOMINAL, rig.code(), POLICY, [0], Mode.CLEAN)
    (ep,) = rig.backend.run(NOMINAL, rig.code(), POLICY, [0], Mode.CLEAN, record_trajectories=True)
    assert [c.spec.record_trajectories for c in rig.runner.calls] == [False, True]
    assert ep.logs["record_trajectories"] is True


# ---- actuator overrides: code under test, policy only ----------------------------------- #

def test_actuator_overrides_are_applied_for_the_policy(tmp_path):
    rig = make_backend(tmp_path, eval_config=with_eval(env={"actuator_overrides": GAINS}))
    planned = rig.backend.plan_run(NOMINAL, rig.code(), POLICY, Mode.CLEAN, 0, 1, workdir=tmp_path / "w")
    assert planned.actuated and json.loads(planned.invocation.env[ENV_OVERRIDES]) == GAINS
    assert planned.invocation.env[ENV_NAME] == planned.spec.environment["type"] == \
        "vla_put_item_in_fridge_and_close_door_actuated"
    assert planned.invocation.env[ENV_BASE].endswith(":GR1PutAndCloseDoorEnvironment")
    assert planned.invocation.argv[1:3] == ("-m", "vla_isaac.arena_ext.launch_experiment")


@pytest.mark.parametrize("actor", [REF_RAW, REF_WRAP])
def test_actuator_overrides_are_absent_for_a_reference_actor(tmp_path, actor):
    rig = make_backend(tmp_path, eval_config=with_eval(env={"actuator_overrides": GAINS}))
    code = rig.code()
    assert rig.backend.actuator_overrides(actor, code.eval_config) == {}
    planned = rig.backend.plan_run(NOMINAL, code, actor, Mode.CLEAN, 0, 1, workdir=tmp_path / "w")
    assert not planned.actuated
    assert not {ENV_OVERRIDES, ENV_BASE, ENV_NAME} & set(planned.invocation.env)
    assert planned.spec.environment["type"] == "put_item_in_fridge_and_close_door"
    assert "launch_experiment" not in " ".join(planned.invocation.argv)


def test_reference_through_the_wrapper_raises_replay_unavailable_before_any_side_effect(tmp_path, monkeypatch):
    rig = make_backend(tmp_path, cache=RolloutCache())
    code = rig.code()
    built = []
    monkeypatch.setattr(rig.workspaces, "materialize", lambda *a, **k: built.append(a))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no subprocess for G3"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("no subprocess for G3"))
    with pytest.raises(ReplayUnavailable, match="G3"):
        rig.backend.run(NOMINAL, code, REF_WRAP, [0, 10001], Mode.CLEAN)
    with pytest.raises(ReplayUnavailable):           # not even a malformed request gets further
        rig.backend.run(SceneSpec("not_the_pilot"), code, REF_WRAP, [-1], Mode.INCIDENT)
    assert not built and not rig.runner.calls and not rig.servers and not list(rig.cache.keys())
    assert not (tmp_path / "out").exists()
    cached = CachedBackend(rig.backend, RolloutCache())
    with pytest.raises(ReplayUnavailable):
        cached.run(NOMINAL, code, REF_WRAP, [0], Mode.CLEAN)
    assert cached.misses == 0


def test_raw_reference_goes_to_the_reference_runner_with_scene_variations_only(tmp_path):
    seen = {}

    class Ref:
        def run(self, scene, actor, seeds, *, variations):
            seen.update(scene=scene, variations=variations)
            return [EpisodeResult(s, True) for s in seeds]
    rig = make_backend(tmp_path, eval_config=with_eval(env={"actuator_overrides": GAINS}), reference=Ref())
    out = rig.backend.run(DARK, rig.code(), REF_RAW, [4, 1], Mode.CLEAN)
    assert [e.seed for e in out] == [4, 1] and not rig.runner.calls and not rig.servers
    assert seen["variations"]["light.intensity.sampler_cfg.low"] == [100.0]
    assert not any("actuator" in k for k in seen["variations"])
    with pytest.raises(NotImplementedError, match="Phase 4"):
        make_backend(tmp_path / "b").backend.run(NOMINAL, rig.code(), REF_RAW, [0], Mode.CLEAN)


# ---- extra episodes go into the cache --------------------------------------------------- #

def test_episodes_beyond_the_requested_seeds_go_into_the_cache(tmp_path):
    cache = RolloutCache(str(tmp_path / "rollouts.jsonl"))
    rig = make_backend(tmp_path, cache=cache)
    cb = CachedBackend(rig.backend, cache)
    code = rig.code()
    out = cb.run(DARK, code, POLICY, [4], Mode.CLEAN)                 # runs episodes 0..4 of build 0
    assert [e.seed for e in out] == [4] and len(cache) == 5
    assert all(CacheKey.make(DARK, code, POLICY, s, Mode.CLEAN, "isaac:test") in cache for s in range(5))
    again = cb.run(DARK, code, POLICY, [0, 1, 2, 3, 4], Mode.CLEAN)  # all served from the cache
    assert len(rig.runner.calls) == 1 and cb.hits == 5 and [e.seed for e in again] == [0, 1, 2, 3, 4]


# ---- errors ----------------------------------------------------------------------------- #

def test_a_crashed_run_raises_and_stops_the_server(tmp_path):
    rig = make_backend(tmp_path, runner=FakeArenaRunner(crash_returncode=139))
    with pytest.raises(ArenaRunError) as ei:
        rig.backend.run(NOMINAL, rig.code(), POLICY, [0], Mode.CLEAN)
    assert ei.value.returncode == 139 and rig.servers[0].events == ["start", "stop"]


def test_a_run_with_missing_episodes_raises(tmp_path):
    rig = make_backend(tmp_path, runner=FakeArenaRunner(short_by=1))
    with pytest.raises(EpisodeResultsError, match="missing episode indices"):
        rig.backend.run(NOMINAL, rig.code(), POLICY, [3], Mode.CLEAN)


def test_an_unknown_factor_or_scene_raises_before_anything_runs(tmp_path):
    from router.schemas import Factor, SceneSpec
    rig = make_backend(tmp_path)
    with pytest.raises(UnknownFactor):
        rig.backend.run(SceneSpec("gr1_fridge_ranch", (Factor("material(mug)", "glossy"),)), rig.code(), POLICY,
                        [0], Mode.CLEAN)
    with pytest.raises(ConfigError, match="not served"):
        rig.backend.run(SceneSpec("droid_maple"), rig.code(), POLICY, [0], Mode.CLEAN)
    assert not rig.runner.calls and not rig.servers


def test_an_eval_config_for_another_environment_is_refused(tmp_path):
    rig = make_backend(tmp_path, eval_config=with_eval(env={"object": "jug"}))
    with pytest.raises(ConfigError, match="object"):
        rig.backend.run(NOMINAL, rig.code(), POLICY, [0], Mode.CLEAN)


def test_a_crashed_run_ends_the_incident_unresolved_in_the_loop(tmp_path):
    """F3: any exception from the backend ends the incident UNRESOLVED with the error recorded."""
    rig = make_backend(tmp_path, runner=FakeArenaRunner(crash_returncode=1))
    code = rig.code()
    incident = Incident("inc-1", DARK, TrainingSpec("gr00t"), code, POLICY, failing_seeds=(0, 1, 2),
                        observed_mode=Mode.CLEAN)
    cfg = LoopConfig(RouterConfig(ReplayConfig(0.2, (0.7, 0.95)), "mimic-gr1_pink"),
                     Rung1Config(staged_plan_for(50), 0.2, (0.7, 0.95), 1.0, 0.9))
    res = IncidentLoop(rig.backend, cfg, repairer=None, inputs=RouterInputs(), data_ladder=StubDataLadder()).run(incident)
    assert res.status is LoopStatus.UNRESOLVED
    assert (res.error_stage, res.error_type) == ("router", "ArenaRunError")


# ---- Rung 1 kind: isaac (router change 3) ------------------------------------------------ #

def test_rung1_kind_isaac_builds_this_backend_from_its_import_path(tmp_path):
    from ladder.rung1 import backend_from_factory
    from tests.helpers import PILOT_TASK, make_checkout

    checkout = make_checkout(tmp_path / "checkout")
    cfg = {"arena_root": str(tmp_path), "gr00t_root": str(tmp_path), "arena_python": "py", "server_python": "py",
           "output_root": str(tmp_path / "out"), "task_config": str(PILOT_TASK),
           "workspace_store": str(tmp_path / "store"), "workspace_bases": {"pw": str(checkout)}}
    path = tmp_path / "backend.yaml"
    path.write_text(yaml.safe_dump(cfg))
    be = backend_from_factory({"kind": "isaac", "factory": "vla_isaac.backend.isaac_backend:IsaacBackend.from_config",
                               "args": {"config": str(path), "backend_id": "isaac:t"}})
    assert type(be).__name__ == "IsaacBackend" and be.backend_id == "isaac:t"
    assert be.task.nominal_scene == "gr1_fridge_ranch" and set(be.workspaces.bases) == {"pw"}
