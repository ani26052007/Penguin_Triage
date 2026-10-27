"""Router change 3: Rung 1 ``kind: isaac`` builds its backend from an import path in the config
(``"package.module:attr[.attr]"`` called with ``args``); vla-router never imports the simulator side."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ladder.backend import MockBackend
from ladder.rung1 import backend_from_factory, load_object, main
from tests.test_rung1 import write_config

DUMMY = '''
from ladder.backend import MockBackend

calls = []


def make(default_rate, salt="dummy"):
    calls.append((default_rate, salt))
    return MockBackend([], default_rate, salt=salt)


class Factory:
    @classmethod
    def from_config(cls, rate):
        return MockBackend([], rate)


def not_a_backend():
    return object()
'''


@pytest.fixture
def dummy(tmp_path, monkeypatch):
    (tmp_path / "dummy_sim_backend.py").write_text(DUMMY)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "dummy_sim_backend"
    sys.modules.pop("dummy_sim_backend", None)


def isaac_config(tmp_path: Path, backend: dict) -> Path:
    cfg = json.loads(write_config(tmp_path / "base.json", 0.97).read_text())
    cfg["backend"] = {"kind": "isaac", **backend}
    p = tmp_path / "isaac.json"
    p.write_text(json.dumps(cfg))
    return p


def test_a_dummy_import_path_builds_the_backend_and_rung1_runs_on_it(tmp_path, dummy, capsys):
    p = isaac_config(tmp_path, {"factory": f"{dummy}:make", "args": {"default_rate": 0.97, "salt": "s"}})
    assert main(["--seeds", "30", "--config", str(p)]) == 0
    assert capsys.readouterr().out.startswith("RUNG1 PASS")
    assert sys.modules[dummy].calls == [(0.97, "s")]


def test_the_path_may_name_a_class_attribute_and_args_are_optional(dummy):
    b = backend_from_factory({"kind": "isaac", "factory": f"{dummy}:Factory.from_config", "args": {"rate": 0.5}})
    assert isinstance(b, MockBackend) and b.default_rate == 0.5
    assert load_object("ladder.backend:MockBackend") is MockBackend


@pytest.mark.parametrize("backend,match", [
    ({}, "needs \"factory\""),
    ({"factory": "dummy_sim_backend.make"}, "package.module:attribute"),
    ({"factory": "no_such_module_xyz:make"}, "ModuleNotFoundError"),
    ({"factory": "dummy_sim_backend:missing"}, "AttributeError"),
    ({"factory": "dummy_sim_backend:not_a_backend"}, "not a RolloutBackend"),
    ({"factory": "dummy_sim_backend:calls"}, "not callable"),
    ({"factory": "dummy_sim_backend:make", "args": [0.9]}, "mapping"),
])
def test_bad_factories_are_clean_config_errors(tmp_path, dummy, capsys, backend, match):
    with pytest.raises(ValueError, match=match):
        backend_from_factory({"kind": "isaac", **backend})
    assert main(["--config", str(isaac_config(tmp_path, backend))]) == 2
    assert "run_rung1: error: ValueError" in capsys.readouterr().err


def test_vla_router_does_not_import_the_simulator_package():
    root = Path(__file__).resolve().parents[1]
    hits = [str(p.relative_to(root)) for d in ("ladder", "router") for p in (root / d).rglob("*.py")
            if "vla_isaac" in p.read_text(encoding="utf-8").replace("vla_isaac.backend.isaac_backend:IsaacBackend.from_config", "")]
    assert hits == []
