"""policy-wrapper as a WorkspaceManager base: the materialise / config-loader round trip, the pilot
ConfigMapping on the base and on edited eval configs (one violation per edited mapped key), the
derived keys (IMPL_PLAN Q3), and Rung 0 on the repo.

Runs on the real policy-wrapper checkout (``POLICY_WRAPPER_ROOT``, default ``../policy-wrapper``);
the checkpoint statistics are a FIXTURE."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from ladder.rung0 import Rung0Config, run_rung0
from ladder.workspace import WorkspaceError, WorkspaceManager, code_ref_of, file_config_loader
from router import vocab as V
from router.schemas import ActorKind, ActorRef, Incident, Mode, SceneSpec, TrainingSpec
from router.stage1.cfg_contract import ConfigContractCheck

from vla_isaac.config import load_task_config
from vla_isaac.specs.norm_stats import load_flat_statistics
from vla_isaac.workspace import config_mapping, eval_config_loader, make_workspaces

from tests.helpers import PILOT_TASK, backend_config

PW = Path(os.environ.get("POLICY_WRAPPER_ROOT", Path(__file__).resolve().parents[2] / "policy-wrapper"))
pytestmark = pytest.mark.skipif(not (PW / ".git").exists(), reason=f"no policy-wrapper checkout at {PW}")

TASK = load_task_config(PILOT_TASK)
CKPT_ID = "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge"
CKPT = Path(__file__).resolve().parent / "fixtures" / "checkpoint_gr1"
STATS = load_flat_statistics(CKPT / "statistics.json", "gr1", TASK.eval_config.norm_stats_groups)
INSTRUCTION = "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."
SPEC = TrainingSpec("gr00t_n1d6_gr1_ranch_bottle_into_fridge", image_size=(512, 512), norm_stats=STATS,
                    action_horizon=16, replan_steps=16, denoising_steps=4, control_hz=50, render_settings={},
                    instruction_templates={"default": INSTRUCTION})        # pilot values; Phase 4 extracts them


@pytest.fixture
def ws(tmp_path) -> WorkspaceManager:
    return WorkspaceManager(tmp_path / "store", {"pw": PW}, config_loader=eval_config_loader(TASK, {CKPT_ID: CKPT}))


def edited(ws, tmp_path, edit) -> "CodeRef":
    work = ws.materialize(ws.base_ref("pw"), tmp_path / "edit")
    path = work / "config" / "eval.yaml"
    cfg = yaml.safe_load(path.read_text())
    edit(cfg, work)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return ws.adopt(work)


def contract(code_ref) -> list:
    inc = Incident("pw", SceneSpec(TASK.nominal_scene), SPEC, code_ref, ActorRef(ActorKind.POLICY, "ckpt"), (0,),
                   Mode.CLEAN)
    return ConfigContractCheck(config_mapping(TASK)).run(inc)


def test_materialize_and_the_file_loader_round_trip_the_eval_config(ws, tmp_path):
    base = ws.base_ref("pw")
    assert base.code_hash.startswith("pw@") and "+" not in base.code_hash
    work = ws.materialize(base, tmp_path / "w")
    raw = file_config_loader("config/eval.yaml")(work)
    assert raw == yaml.safe_load((PW / "config" / "eval.yaml").read_text())
    assert code_ref_of(work, eval_config_loader(TASK, {CKPT_ID: CKPT})) == base
    tracked = sorted(p.relative_to(work) for p in work.rglob("*") if p.is_file() and ".git" not in p.parts)
    assert tracked == sorted(p.relative_to(PW) for p in PW.rglob("*") if p.is_file() and ".git" not in p.parts
                             and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)


def test_the_loader_adds_the_derived_image_hw_and_effective_norm_stats(ws):
    cfg = ws.base_ref("pw").eval_config
    assert cfg["client"]["target_image_hw"] == [512, 512] and cfg["client"]["target_image_size"] == [512, 512, 3]
    assert cfg["server"]["effective_norm_stats"] == STATS
    assert len(STATS["action"]["min"]) == 26 and set(STATS) == {"state", "action"}
    assert STATS["action"]["min"][:7] == [-1.0] * 7 and STATS["action"]["min"][7:14] == [-1.1] * 7


def test_make_workspaces_registers_policy_wrapper_with_that_loader(tmp_path):
    cfg = backend_config(tmp_path, PW, workspace_bases={"pw": PW}, checkpoints={CKPT_ID: CKPT})
    ws = make_workspaces(cfg, TASK)
    assert ws.base_ref("pw").eval_config["server"]["effective_norm_stats"] == STATS


def test_a_checkpoint_the_loader_cannot_find_is_an_error(tmp_path):
    ws = WorkspaceManager(tmp_path / "store", {"pw": PW}, config_loader=eval_config_loader(TASK, {}))
    with pytest.raises(WorkspaceError, match="checkpoints map"):
        ws.base_ref("pw")


def test_the_nominal_eval_config_gives_no_violation(ws):
    assert contract(ws.base_ref("pw")) == []


def _set(path: str, value):
    def edit(cfg, work):
        section, key = path.split(".")
        cfg[section][key] = value
    return edit


EDITS = {                                        # TrainingSpec field -> edit of the mapped eval.yaml key
    "action_horizon": _set("client.action_horizon", 32),
    "replan_steps": _set("client.action_chunk_length", 8),
    "denoising_steps": _set("server.denoising_steps", 8),
    "control_hz": _set("env.control_hz", 30),
    "image_size": _set("client.target_image_size", [480, 640, 3]),
    "instruction_templates": _set("client.language_instruction", "Put the bottle in the fridge."),
    "render_settings": _set("env.render", {"preset": "fast"}),
}


def test_every_mapped_field_has_an_edit():
    assert set(EDITS) | {"norm_stats"} == set(TASK.config_mapping)


@pytest.mark.parametrize("field", sorted(EDITS))
def test_each_edited_mapped_key_gives_exactly_one_violation(ws, tmp_path, field):
    (ev,) = contract(edited(ws, tmp_path, EDITS[field]))
    assert ev.id == f"s1.cfg.{field}"
    assert ev.signal == (V.SIG_RENDER_SETTINGS_DIFFER if field == "render_settings" else V.SIG_CONTRACT_VIOLATION)
    assert ev.location == f"eval_config.{TASK.config_mapping[field]}"


def test_a_norm_stats_override_with_modified_statistics_gives_one_norm_stats_violation(ws, tmp_path):
    def override(cfg, work):
        stats = json.loads((CKPT / "statistics.json").read_text())
        stats["gr1"]["action"]["left_arm"]["max"] = [5.0] * 7
        (work / "config" / "stats_override.json").write_text(json.dumps(stats))
        cfg["server"]["norm_stats_override"] = "config/stats_override.json"
    code = edited(ws, tmp_path, override)
    assert code.eval_config["server"]["effective_norm_stats"]["action"]["max"][:7] == [5.0] * 7
    (ev,) = contract(code)
    assert ev.id == "s1.cfg.norm_stats" and ev.signal == V.SIG_CONTRACT_VIOLATION
    assert ev.location == "eval_config.server.effective_norm_stats"


def test_an_override_with_the_same_statistics_gives_no_violation(ws, tmp_path):
    def override(cfg, work):
        (work / "config" / "same.json").write_text((CKPT / "statistics.json").read_text())
        cfg["server"]["norm_stats_override"] = "config/same.json"
    assert contract(edited(ws, tmp_path, override)) == []


class _Env:
    """``run_rung0`` env over hagent's LocalEnv, with the patch as ``git_diff``."""

    def __init__(self, work: Path, diff: str):
        from hagent.env import LocalEnv
        self.env, self.diff = LocalEnv(work, timeout_s=600), diff

    def execute(self, cmd, timeout_s=None):
        return self.env.execute(cmd, timeout_s)

    def git_diff(self):
        return self.diff


RUNG0 = Rung0Config(test_cmd=f"{sys.executable} -m pytest -q -p no:cacheprovider", timeout_s=600)


@pytest.mark.parametrize("broken", [False, True])
def test_rung0_runs_the_repo_tests_on_a_working_copy(ws, tmp_path, broken):
    work = ws.materialize(ws.base_ref("pw"), tmp_path / "r0")
    target = work / "policy_wrapper" / "seeding.py"
    text = target.read_text()
    target.write_text(text + ("\ndef broken(:\n" if broken else "\n# reviewed\n"))
    diff = f"--- a/policy_wrapper/seeding.py\n+++ b/policy_wrapper/seeding.py\n@@ -1 +1,2 @@\n+x\n"
    res = run_rung0(_Env(work, diff), RUNG0)
    assert res.passed is (not broken)
    tests = next(c for c in res.checks if c.name == "tests")
    assert tests.status == ("fail" if broken else "pass")


def test_rung0_rejects_a_patch_to_the_instrumentation_and_accepts_one_elsewhere(ws, tmp_path):
    """The evidence is protected: the pilot's Rung 0 forbids policy_wrapper/instrument.py."""
    from vla_isaac.workspace import rung0_config

    cfg = rung0_config(TASK)
    assert "policy_wrapper/instrument.py" in cfg.forbidden_prefixes
    work = ws.materialize(ws.base_ref("pw"), tmp_path / "w")
    for path, ok in (("policy_wrapper/instrument.py", False), ("policy_wrapper/policy/gr00t_core.py", True)):
        diff = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n+x\n"
        res = run_rung0(_Env(work, diff), cfg)
        assert res.passed is ok, res.feedback
        if not ok:
            assert res.checks[0].name == "diff" and "forbidden" in res.feedback
