"""Phase 4: ``extract_training_spec`` from FIXTURE checkpoint files and Arena's real config files.

The checkpoint and dataset files are FIXTUREs (``tests/fixtures/README.md``); Arena's closed-loop YAML,
data config, ``modality.json`` and converter YAML are read from the Arena checkout. The end-to-end test
runs the router's config contract on the spec against policy-wrapper's ``eval.yaml`` through the
eval-config loader (IMPL_PLAN Q3), which shares the norm-stats function with the extraction.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ladder.workspace import WorkspaceManager
from router.schemas import ActorKind, ActorRef, Incident, Mode, SceneSpec, TrainingSpec
from router.stage1.cfg_contract import ConfigContractCheck

from vla_isaac.config import ConfigError, load_task_config, task_config_from_dict
from vla_isaac.specs.arena_data_config import DataConfigError, read_data_config
from vla_isaac.specs.extract_training_spec import (SPEC_FIELDS, SpecError, extract_for_backend, extract_training_spec,
                                                    main, section_from_dict)
from vla_isaac.specs.norm_stats import training_norm_stats
from vla_isaac.workspace import config_mapping, eval_config_loader

from tests.helpers import PILOT_TASK, REPO

ARENA = REPO.parent / "external" / "IsaacLab-Arena"
PW = Path(os.environ.get("POLICY_WRAPPER_ROOT", REPO.parent / "policy-wrapper"))
FIX = Path(__file__).resolve().parent / "fixtures"
CKPT, DATASET = FIX / "checkpoint_gr1", FIX / "dataset_gr1"
TASK = load_task_config(PILOT_TASK)
SECTION = TASK.training_spec
GROUPS = TASK.eval_config.norm_stats_groups
EVAL_INSTRUCTION = "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."
pytestmark = pytest.mark.skipif(not (ARENA / "isaaclab_arena_gr00t").is_dir(), reason=f"no Arena checkout at {ARENA}")


def extract(ckpt=CKPT, dataset=DATASET, section=SECTION) -> TrainingSpec:
    return extract_training_spec(section, checkpoint_dir=ckpt, arena_root=ARENA, fallback_groups=GROUPS,
                                 dataset_dir=dataset)


def edited_checkpoint(tmp_path, edit) -> Path:
    ckpt = tmp_path / "ckpt"
    shutil.copytree(CKPT, ckpt)
    path = ckpt / "processor_config.json"
    d = json.loads(path.read_text())
    edit(d["processor_kwargs"]["modality_configs"]["gr1"])
    path.write_text(json.dumps(d))
    return ckpt


# ---- every field filled or explicitly missing --------------------------------------------------

def test_the_nominal_spec_has_the_pilot_values_and_says_where_each_came_from():
    spec = extract()
    assert (spec.policy_name, spec.action_horizon, spec.replan_steps, spec.denoising_steps) == (
        "gr00t_n1d6_gr1_ranch_bottle_into_fridge", 16, 16, 4)
    assert spec.control_hz == 50.0 and spec.image_size == (512, 512) and spec.channel_order == "RGB"
    assert spec.render_settings == {} and spec.instruction_templates == {"task_0": EVAL_INSTRUCTION}
    layout = {"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 20], "right_hand": [20, 26]}
    assert spec.state_layout == layout and spec.action_layout == layout
    assert len(spec.norm_stats["action"]["min"]) == 26 and set(spec.norm_stats) == {"state", "action"}
    x = spec.extra
    assert set(x["sources"]) | set(x["missing"]) == set(SPEC_FIELDS) and not set(x["sources"]) & set(x["missing"])
    assert set(x["missing"]) == {"camera_poses"}
    assert x["sources"]["action_horizon"].startswith("checkpoint:processor_config.json")
    assert x["sources"]["denoising_steps"] == "checkpoint:config.json num_inference_timesteps"
    assert x["sources"]["instruction_templates"] == "dataset:meta/tasks.jsonl"
    assert x["sources"]["control_hz"] == "dataset:meta/info.json fps"
    assert x["disagreements"] == []                       # the fixture checkpoint agrees with Arena
    assert (x["embodiment_tag"], x["embodiment_id"], x["model"]["tune_top_llm_layers"]) == ("gr1", 24, 4)
    assert x["arena_instructions"]["arena_converter"].startswith("Place the ranch dressing bottle")


def test_the_spec_round_trips_through_json_and_the_cli_writes_it(tmp_path):
    spec = extract()
    assert TrainingSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    backend = {"arena_root": str(ARENA), "gr00t_root": str(tmp_path), "arena_python": "py", "server_python": "py",
               "output_root": str(tmp_path / "out"), "task_config": str(PILOT_TASK),
               "workspace_store": str(tmp_path / "store"), "workspace_bases": {"pw": str(tmp_path)},
               "checkpoints": {SECTION.checkpoint: str(CKPT)}, "datasets": {SECTION.dataset: str(DATASET)}}
    (tmp_path / "backend.yaml").write_text(yaml.safe_dump(backend))
    assert extract_for_backend(tmp_path / "backend.yaml") == spec
    assert main(["--backend", str(tmp_path / "backend.yaml"), "--output", str(tmp_path / "spec.json")]) == 0
    assert TrainingSpec.from_dict(json.loads((tmp_path / "spec.json").read_text())) == spec


# ---- the checkpoint wins over Arena's data config -----------------------------------------------

def test_the_checkpoint_modality_config_wins_and_every_difference_lists_both_values(tmp_path):
    def edit(m):
        m["action"]["delta_indices"] = list(range(8))
        m["state"]["sin_cos_embedding_keys"] = None
    spec = extract(edited_checkpoint(tmp_path, edit))
    assert spec.action_horizon == 8
    got = {d["field"]: d for d in spec.extra["disagreements"]}
    assert set(got) == {"modality.action.delta_indices", "modality.state.sin_cos_embedding_keys", "action_horizon"}
    assert got["modality.action.delta_indices"] == {"field": "modality.action.delta_indices", "used": "checkpoint",
                                                    "values": {"checkpoint": list(range(8)),
                                                               "arena_data_config": list(range(16))}}
    assert got["action_horizon"]["values"] == {"checkpoint": 8, "arena_data_config": 16, "arena_closedloop": 16}
    assert got["modality.state.sin_cos_embedding_keys"]["values"] == {
        "checkpoint": None, "arena_data_config": ["left_arm", "right_arm", "left_hand", "right_hand"]}


def test_the_checkpoint_group_order_sets_the_layout_and_both_sides_of_the_norm_stats(tmp_path):
    order = ["right_arm", "left_arm", "right_hand", "left_hand"]

    def edit(m):
        m["state"]["modality_keys"] = order
        m["action"]["modality_keys"] = order
    ckpt = edited_checkpoint(tmp_path, edit)
    spec = extract(ckpt)
    assert list(spec.action_layout) == order and spec.action_layout["right_arm"] == [0, 7]
    effective = training_norm_stats(ckpt, "gr1", GROUPS)        # what the eval-config loader computes
    assert effective.groups_source == "checkpoint:processor_config.json" and spec.norm_stats == effective.stats
    fields = {d["field"] for d in spec.extra["disagreements"]}
    assert {"norm_stats_groups", "action_layout", "state_layout", "modality.action.modality_keys"} <= fields


# ---- the instruction comes from the dataset or is missing ---------------------------------------

def test_without_a_local_dataset_the_instruction_is_missing_never_the_eval_string():
    spec = extract(dataset=None)
    assert spec.instruction_templates is None
    assert "V-train-instr" in spec.extra["missing"]["instruction_templates"]
    assert spec.extra["sources"]["control_hz"] == "arena_converter:fps" and spec.control_hz == 50.0
    assert spec.extra["sources"]["image_size"] == "arena_converter:target_image_size" and spec.image_size == (512, 512)
    assert EVAL_INSTRUCTION not in json.dumps({k: v for k, v in spec.to_dict().items() if k != "extra"})


def test_a_dataset_without_tasks_jsonl_leaves_the_instruction_missing(tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(DATASET, ds)
    (ds / "meta" / "tasks.jsonl").unlink()
    spec = extract(dataset=ds)
    assert spec.instruction_templates is None and "tasks.jsonl" in spec.extra["missing"]["instruction_templates"]


# ---- the config contract end to end -------------------------------------------------------------

def _contract(spec: TrainingSpec, eval_edit=None, tmp_path=None) -> list:
    if not (PW / ".git").exists():
        pytest.skip(f"no policy-wrapper checkout at {PW}")
    ws = WorkspaceManager(tmp_path / "store", {"pw": PW}, config_loader=eval_config_loader(TASK, {SECTION.checkpoint: CKPT}))
    code = ws.base_ref("pw")
    if eval_edit is not None:
        work = ws.materialize(code, tmp_path / "edit")
        path = work / "config" / "eval.yaml"
        cfg = yaml.safe_load(path.read_text())
        eval_edit(cfg)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        code = ws.adopt(work)
    inc = Incident("pw", SceneSpec(TASK.nominal_scene), spec, code, ActorRef(ActorKind.POLICY, "ckpt"), (0,), Mode.CLEAN)
    return ConfigContractCheck(config_mapping(TASK)).run(inc)


def test_the_nominal_pilot_gives_zero_config_contract_violations_end_to_end(tmp_path):
    spec = extract()
    assert set(config_mapping(TASK).keys) <= set(spec.extra["sources"])   # every mapped field is compared
    assert _contract(spec, tmp_path=tmp_path) == []


def test_the_spec_and_the_loader_give_the_same_norm_stats_structure(tmp_path):
    if not (PW / ".git").exists():
        pytest.skip(f"no policy-wrapper checkout at {PW}")
    spec = extract()
    ws = WorkspaceManager(tmp_path / "store", {"pw": PW}, config_loader=eval_config_loader(TASK, {SECTION.checkpoint: CKPT}))
    effective = ws.base_ref("pw").eval_config["server"]["effective_norm_stats"]
    assert effective == spec.norm_stats


def test_a_training_string_that_differs_from_the_eval_string_is_one_violation(tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(DATASET, ds)
    converter = "Place the ranch dressing bottle on the top shelf of the fridge, and close the fridge door."
    (ds / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": converter}) + "\n")
    found = _contract(extract(dataset=ds), tmp_path=tmp_path)
    assert [e.id for e in found] == ["s1.cfg.instruction_templates"]


def test_an_edited_denoising_step_count_is_still_caught_against_the_extracted_spec(tmp_path):
    found = _contract(extract(), lambda cfg: cfg["server"].update(denoising_steps=8), tmp_path)
    assert [e.id for e in found] == ["s1.cfg.denoising_steps"]


# ---- inputs that cannot be used -----------------------------------------------------------------

def test_a_missing_checkpoint_file_or_an_unknown_embodiment_is_an_error(tmp_path):
    ckpt = tmp_path / "ckpt"
    shutil.copytree(CKPT, ckpt)
    (ckpt / "embodiment_id.json").unlink()
    with pytest.raises(SpecError, match="embodiment_id.json"):
        extract(ckpt)
    with pytest.raises(SpecError, match="no embodiment 'g1'"):
        extract(section=replace(SECTION, embodiment_tag="G1"))


def test_the_section_is_strict_and_only_unrecorded_fields_may_be_declared():
    d = yaml.safe_load(PILOT_TASK.read_text())["training_spec"]
    assert section_from_dict(d) == SECTION
    with pytest.raises(ConfigError, match="unknown keys"):
        section_from_dict({**d, "instruction": "x"})
    with pytest.raises(ConfigError, match="may only name"):
        section_from_dict({**d, "declared": {"instruction_templates": {"a": "b"}}})


# ---- Arena's data config, read without importing it -------------------------------------------

def test_arena_data_config_is_read_as_data_like_a_checkpoint_writes_it():
    d = read_data_config(ARENA / SECTION.arena_data_config)
    assert set(d) == {"video", "state", "action", "language"}
    assert d["action"]["delta_indices"] == list(range(16)) and d["video"]["modality_keys"] == ["ego_view"]
    assert d["action"]["action_configs"][0] == {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT",
                                                "state_key": None}
    assert d["video"]["sin_cos_embedding_keys"] is None


def test_a_data_config_the_reader_cannot_read_is_an_error(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("N = 16\ncfg = {'action': ModalityConfig(delta_indices=list(range(N)), modality_keys=['a'])}\n"
                   "register_modality_config(cfg, embodiment_tag=None)\n")
    with pytest.raises(DataConfigError, match="cannot read"):
        read_data_config(bad)
