from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vla_isaac.config import ConfigError, load_backend_config, load_task_config

CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_the_example_backend_config_loads_and_resolves_paths():
    cfg = load_backend_config(CONFIG / "backend.example.yaml")
    assert cfg.arena_root == (CONFIG / "../../external/IsaacLab-Arena").resolve()
    assert cfg.runner == "experiment_runner" and cfg.incident_num_envs == 10 and cfg.seed_stride == 10000
    assert cfg.server.port == 5555 and cfg.trajectories.camera_stride == 10 and cfg.app_flags == ("--viz", "none")
    assert cfg.task_config == CONFIG / "pilot_gr1_fridge.yaml"


def test_the_pilot_task_config_loads():
    t = load_task_config(CONFIG / "pilot_gr1_fridge.yaml")
    assert t.environment["type"] == "put_item_in_fridge_and_close_door" and t.factor_table.is_file()


def _with(tmp_path, **changes):
    d = yaml.safe_load((CONFIG / "backend.example.yaml").read_text())
    d.update(changes)
    for k in [k for k, v in changes.items() if v is None]:
        del d[k]
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


@pytest.mark.parametrize("changes,match", [
    ({"runnr": "x"}, "unknown keys"),
    ({"arena_root": None}, "missing keys"),
    ({"runner": "isaac_lab"}, "runner"),
    ({"incident_num_envs": 1}, "incident_num_envs"),
    ({"server": {"prot": 1}}, "unknown keys"),
    ({"workspace_bases": {}}, "workspace_bases"),
])
def test_bad_backend_configs_raise(tmp_path, changes, match):
    with pytest.raises(ConfigError, match=match):
        load_backend_config(_with(tmp_path, **changes))


def test_bad_task_config_raises(tmp_path):
    d = yaml.safe_load((CONFIG / "pilot_gr1_fridge.yaml").read_text())
    d["environment_checks"]["camera"] = "env.camera"
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="environment_checks"):
        load_task_config(p)
