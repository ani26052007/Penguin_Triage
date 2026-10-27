"""Phase 6: Rung B, Mimic regeneration under the failing factor through the RAW reference runner, and
the command for Arena's HDF5 -> LeRobot converter.

Mimic never runs here (``FakeMimicRunner`` from ``tests/test_mimic.py``). Arena's converter runs only when it
imports in this environment (VERIFY(V-converter-noisaac)); otherwise that test is skipped with the reason."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from vla_isaac.data.config import PipelineConfig
from vla_isaac.data.rung_b_mimic import (DETERMINISTIC_WARNING, RungBError, convert, converter_config, lerobot_dir,
                                         plan_conversion, regenerate, rung_b_report)
from vla_isaac.reference.mimic import MimicSeed
from vla_isaac.runner.subprocess_runner import ArenaRunError, ArenaRunner, RunInvocation
from vla_isaac.scene.factors import FactorTable

from tests.helpers import backend_config
from tests.test_mimic import PER_RUN, TASK, make_reference

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT.parent / "external" / "IsaacLab-Arena"
PIPE = PipelineConfig.load(ROOT / "config" / "finetune_gr1_fridge.yaml")
FACTORS = FactorTable.load(TASK.factor_table)
DARK_C = {"lighting.intensity": "100.0"}
NOMINAL = "gr1_fridge_ranch"


# ---- regeneration ------------------------------------------------------------------------------------

def test_fixed_seed_regeneration_is_reported_as_deterministic_with_the_diversity_warning(tmp_path):
    ref, runner, source = make_reference(tmp_path, outcomes=[True] * 50 + [False] * 5)
    r = regenerate(ref, FACTORS, NOMINAL, DARK_C, PIPE.rung_b, episodes=50)
    d = r.to_dict()
    assert r.deterministic and d["trajectory_sequence"] == "single fixed sequence" and r.run_seed is None
    assert r.warnings == (DETERMINISTIC_WARNING.format(n=50),)
    assert "one fixed trajectory sequence" in r.warnings[0] and "diversity comes only from the scene variations" in r.warnings[0]
    assert (r.successes, r.attempts_total, r.episodes) == (50, 55, 50)
    argv = list(runner.calls[0].argv)
    assert argv[argv.index("--generation_num_trials") + 1] == "50"            # trials 0..49 of the one sequence
    assert any(a.startswith("light.intensity") for a in argv)                 # the scene's lighting override
    assert "--external_environment_class_path" not in argv and not runner.calls[0].env   # RAW: no actuator overrides
    assert r.hdf5.name == "generated.hdf5" and r.hdf5.parent == runner.calls[0].output_dir
    again = regenerate(ref, FACTORS, NOMINAL, DARK_C, PIPE.rung_b, episodes=50)
    assert list(runner.calls[1].argv) == argv and again.hdf5 == r.hdf5          # the same run, the same sequence
    rep = rung_b_report([r])
    assert rep["deterministic"] is True and rep["warnings"] == list(r.warnings)


def test_a_per_run_seed_is_not_deterministic_and_has_no_warning(tmp_path):
    ref, runner, _ = make_reference(tmp_path, outcomes=[True] * 60, seed=PER_RUN)
    r = regenerate(ref, FACTORS, NOMINAL, DARK_C, PIPE.rung_b, episodes=60)
    argv = list(runner.calls[0].argv)
    assert not r.deterministic and r.warnings == () and r.run_seed is not None
    assert argv[argv.index("--seed") + 1] == str(r.run_seed)
    assert r.to_dict()["trajectory_sequence"] == "seeded per run"


def test_the_default_episode_count_comes_from_config(tmp_path):
    ref, runner, _ = make_reference(tmp_path, outcomes=[True] * PIPE.rung_b.episodes)
    assert regenerate(ref, FACTORS, NOMINAL, DARK_C, PIPE.rung_b).episodes == PIPE.rung_b.episodes == 100


@pytest.mark.parametrize("episodes", [49, 101])
def test_the_episode_count_must_be_in_the_configured_range(tmp_path, episodes):
    ref, _, _ = make_reference(tmp_path, outcomes=[True] * 120)
    with pytest.raises(RungBError, match="50-100"):
        regenerate(ref, FACTORS, NOMINAL, DARK_C, PIPE.rung_b, episodes=episodes)


def test_the_nominal_condition_and_zero_successes_are_refused(tmp_path):
    ref, _, _ = make_reference(tmp_path, outcomes=[False] * 50)
    with pytest.raises(RungBError, match="nominal"):
        regenerate(ref, FACTORS, NOMINAL, {}, PIPE.rung_b, episodes=50)
    with pytest.raises(RungBError, match="no successful demo"):
        regenerate(ref, FACTORS, NOMINAL, DARK_C, PIPE.rung_b, episodes=50)


# ---- conversion: YAML and command ---------------------------------------------------------------------

def test_the_converter_template_is_an_unchanged_copy_of_arenas():
    arena = ARENA / "isaaclab_arena_gr00t" / "lerobot" / "config" / "gr1_manip_ranch_bottle_config.yaml"
    assert PIPE.rung_b.converter.template.read_bytes() == arena.read_bytes()
    assert "SPDX-License-Identifier: Apache-2.0" in PIPE.rung_b.converter.template.read_text()


def test_the_written_yaml_points_at_the_generated_hdf5_and_keeps_processed_actions(tmp_path):
    hdf5 = tmp_path / "gen" / "generated.hdf5"
    data = converter_config(PIPE.rung_b.converter.template, hdf5, ARENA, "processed_actions")
    assert data["action_name_sim"] == "processed_actions"
    assert data["data_root"] == str(tmp_path / "gen") and data["hdf5_name"] == "generated.hdf5"
    for key in ("modality_template_path", "info_template_path", "policy_joints_config_path",
                "action_joints_config_path", "state_joints_config_path"):
        assert Path(data[key]).is_absolute() and Path(data[key]).is_file(), key      # resolved against the Arena root
    original = yaml.safe_load(PIPE.rung_b.converter.template.read_text())
    changed = {k for k in original if original[k] != data[k]}
    assert changed == {"data_root", "hdf5_name", "modality_template_path", "info_template_path",
                       "policy_joints_config_path", "action_joints_config_path", "state_joints_config_path"}


def test_a_template_with_other_actions_is_refused(tmp_path):
    tpl = tmp_path / "t.yaml"
    tpl.write_text(PIPE.rung_b.converter.template.read_text().replace('action_name_sim: "processed_actions"',
                                                                        'action_name_sim: "actions"'))
    with pytest.raises(RungBError, match="processed_actions"):
        converter_config(tpl, tmp_path / "g.hdf5", ARENA, "processed_actions")


def test_the_converter_command_is_arenas(tmp_path):
    cfg = backend_config(tmp_path, tmp_path)
    hdf5 = tmp_path / "ref" / "generated.hdf5"
    inv = plan_conversion(cfg, PIPE.rung_b, hdf5, tmp_path / "work")
    assert list(inv.argv) == ["arena-python", str(cfg.arena_root / "isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py"),
                              "--yaml_file", str(tmp_path / "work" / "converter.yaml")]
    assert inv.cwd == cfg.arena_root and inv.env == {}
    assert inv.output_dir == lerobot_dir(hdf5) == tmp_path / "ref" / "generated" / "lerobot"
    written = (tmp_path / "work" / "converter.yaml").read_text()
    assert written.startswith("# Written by vla_isaac.data.rung_b_mimic") and "Apache-2.0" in written.splitlines()[1]
    assert yaml.safe_load(written)["hdf5_name"] == "generated.hdf5"
    assert plan_conversion(cfg, PIPE.rung_b, hdf5, tmp_path / "work").argv == inv.argv       # same inputs, same command
    inv.output_dir.mkdir(parents=True)
    with pytest.raises(RungBError, match="ask on stdin"):
        plan_conversion(cfg, PIPE.rung_b, hdf5, tmp_path / "work")


class FakeConverter(ArenaRunner):
    def __init__(self, write=True):
        self.write, self.calls = write, []

    def run(self, inv: RunInvocation) -> None:
        self.calls.append(inv)
        if self.write:
            (inv.output_dir / "meta").mkdir(parents=True)
            for n in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
                (inv.output_dir / "meta" / n).write_text("{}")


def test_convert_returns_a_lerobot_source_with_the_condition(tmp_path):
    cfg = backend_config(tmp_path, tmp_path)
    inv = plan_conversion(cfg, PIPE.rung_b, tmp_path / "ref" / "generated.hdf5", tmp_path / "work")
    src = convert(FakeConverter(), inv, "rb_dark", DARK_C)
    assert src.path == inv.output_dir and src.condition == DARK_C and src.id == "rb_dark"
    inv2 = plan_conversion(cfg, PIPE.rung_b, tmp_path / "ref2" / "generated.hdf5", tmp_path / "work2")
    with pytest.raises(ArenaRunError, match="converter left no"):
        convert(FakeConverter(write=False), inv2, "rb", DARK_C)


# ---- the converter itself, only if it imports here (V-converter-noisaac) -------------------------------

IMPORT_CHECK = """
import sys
sys.path.insert(0, sys.argv[1])
import isaaclab_arena_gr00t.lerobot.convert_hdf5_to_lerobot
"""


def converter_import_error() -> str | None:
    p = subprocess.run([sys.executable, "-I", "-B", "-c", IMPORT_CHECK, str(ARENA)], capture_output=True, text=True)
    return None if p.returncode == 0 else (p.stderr.strip().splitlines() or ["unknown error"])[-1]


def test_arenas_converter_on_its_own_fixture_hdf5(tmp_path):
    err = converter_import_error()
    if err is not None:
        pytest.skip(f"V-converter-noisaac: Arena's converter does not import in this environment ({err}); it needs "
                    f"no Isaac module, but its third-party imports (tqdm, imageio, pandas) are not installed here. "
                    f"YAML and command construction are tested above")
    test_data = ARENA / "isaaclab_arena_gr00t" / "tests" / "test_data"
    hdf5 = tmp_path / "test_g1_hdf5_to_lerobot.hdf5"
    shutil.copyfile(test_data / "test_g1_hdf5_to_lerobot.hdf5", hdf5)
    data = converter_config(test_data / "test_g1_locomanip_lerobot" / "test_g1_locomanip_config.yaml", hdf5, ARENA,
                            "processed_actions")
    cfg_path = tmp_path / "converter.yaml"
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))
    script = ARENA / "isaaclab_arena_gr00t" / "lerobot" / "convert_hdf5_to_lerobot.py"
    run = subprocess.run([sys.executable, "-B", str(script), "--yaml_file", str(cfg_path)], cwd=ARENA,
                         env={"PYTHONPATH": str(ARENA), "PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    out = lerobot_dir(hdf5)
    for n in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
        assert (out / "meta" / n).is_file()
    assert len(list((out / "data").rglob("*.parquet"))) == 1
