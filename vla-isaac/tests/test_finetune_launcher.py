"""Phase 6: the GR00T fine-tune command (both presets, 1 and 8 GPUs), the tune_top_llm_layers refusal,
and the run manifest, before and after a (FIXTURE) run."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from vla_isaac.data.config import MixConfig, PipelineConfig
from vla_isaac.data.heldout import split_conditions
from vla_isaac.data.mixer import LeRobotSource, select_mix, write_mixed_dataset
from vla_isaac.data.rung_a import KEEPS_CHECKPOINT
from vla_isaac.runner.subprocess_runner import ArenaRunner, RunInvocation
from vla_isaac.train.finetune_launcher import (MANIFEST_FILE, LauncherError, parse_parameter_counts, plan_finetune,
                                               preset_args, record_run, run_finetune)

from tests.helpers import backend_config
from tests.test_data_pipeline import write_lerobot

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT.parent / "external" / "IsaacLab-Arena"
PIPE = PipelineConfig.load(ROOT / "config" / "finetune_gr1_fridge.yaml")
CKPT_ID = PIPE.finetune.base_checkpoint
FIXTURE_CKPT = ROOT / "tests" / "fixtures" / "checkpoint_gr1"
FIXTURE_RUN = ROOT / "tests" / "fixtures" / "finetune_run"


def setup(tmp_path, *, config_json=None):
    ckpt = tmp_path / "ckpt"
    shutil.copytree(FIXTURE_CKPT, ckpt)
    if config_json is not None:
        if config_json is False:
            (ckpt / "config.json").unlink()
        else:
            (ckpt / "config.json").write_text(config_json)
    cfg = backend_config(tmp_path, tmp_path, checkpoints={CKPT_ID: ckpt})
    return cfg, ckpt


JITTER = "--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"
DEFAULT_FLAGS = "--no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model"
VISUAL_FLAGS = "--no-tune-llm --tune-visual --tune-projector --tune-diffusion-model"


@pytest.mark.parametrize("preset, gpus, expected", [
    ("default", 1, "CUDA_VISIBLE_DEVICES=0 gr00t-python gr00t/experiment/launch_finetune.py --dataset-path {D} "
                   "--output-dir {O} --modality-config-path {A}/isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py "
                   "--global-batch-size 16 --max-steps 30000 --num-gpus 1 --save-steps 5000 --save-total-limit 5 "
                   "--base-model-path {C} " + DEFAULT_FLAGS + " --dataloader-num-workers 4 --embodiment-tag GR1 " + JITTER),
    ("tune_visual", 1, "CUDA_VISIBLE_DEVICES=0 gr00t-python gr00t/experiment/launch_finetune.py --dataset-path {D} "
                       "--output-dir {O} --modality-config-path {A}/isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py "
                       "--global-batch-size 16 --max-steps 30000 --num-gpus 1 --save-steps 5000 --save-total-limit 5 "
                       "--base-model-path {C} " + VISUAL_FLAGS + " --dataloader-num-workers 4 --embodiment-tag GR1 " + JITTER),
    ("default", 8, "gr00t-python -m torch.distributed.run --nproc_per_node=8 --standalone gr00t/experiment/launch_finetune.py "
                   "--dataset-path {D} --output-dir {O} --modality-config-path "
                   "{A}/isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py --global-batch-size 96 "
                   "--max-steps 20000 --num-gpus 8 --save-steps 5000 --save-total-limit 5 --base-model-path {C} "
                   + DEFAULT_FLAGS + " --dataloader-num-workers 16 --embodiment-tag GR1 " + JITTER),
    ("tune_visual", 8, "gr00t-python -m torch.distributed.run --nproc_per_node=8 --standalone gr00t/experiment/launch_finetune.py "
                       "--dataset-path {D} --output-dir {O} --modality-config-path "
                       "{A}/isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py --global-batch-size 96 "
                       "--max-steps 20000 --num-gpus 8 --save-steps 5000 --save-total-limit 5 --base-model-path {C} "
                       + VISUAL_FLAGS + " --dataloader-num-workers 16 --embodiment-tag GR1 " + JITTER),
])
def test_exact_command_strings(tmp_path, preset, gpus, expected):
    cfg, ckpt = setup(tmp_path)
    plan = plan_finetune(cfg, PIPE, dataset_path=tmp_path / "mix", output_dir=tmp_path / "ft", preset=preset, num_gpus=gpus)
    assert plan.command == expected.format(D=tmp_path / "mix", O=tmp_path / "ft", A=cfg.arena_root, C=ckpt)
    assert plan.cwd == cfg.gr00t_root and plan.manifest["command"] == plan.command
    assert plan.env == ({"CUDA_VISIBLE_DEVICES": "0"} if gpus == 1 else {})
    again = plan_finetune(cfg, PIPE, dataset_path=tmp_path / "mix", output_dir=tmp_path / "ft", preset=preset, num_gpus=gpus)
    assert again.command == plan.command


def test_the_ablation_preset_and_jitter_are_arenas_documented_recipe():
    doc = (ARENA / "docs/pages/example_workflows/sequential_static_manipulation/step_4_policy_training.rst").read_text()
    lines = " ".join(line.strip().rstrip("\\").strip() for line in doc.splitlines())
    assert VISUAL_FLAGS in lines and JITTER in lines
    assert " ".join(preset_args(PIPE.finetune.presets["tune_visual"])) == VISUAL_FLAGS
    assert " ".join(preset_args(PIPE.finetune.presets["default"])) == DEFAULT_FLAGS


def test_the_manifest_records_presets_top_layers_checkpoint_mix_heldout_seeds_and_command(tmp_path):
    cfg, ckpt = setup(tmp_path)
    nom = LeRobotSource("nominal", write_lerobot(tmp_path / "nom", 6), {})
    rb = LeRobotSource("rb", write_lerobot(tmp_path / "rb", 3), {"lighting.intensity": "100.0"})
    sel = select_mix(nom.episodes(), rb.episodes(), MixConfig(seed=5, nominal_dataset="nominal"))
    mix = write_mixed_dataset(sel, [nom, rb], tmp_path / "mix")
    held = split_conditions(PIPE.rung_b.conditions, PIPE.heldout).to_dict()
    plan = plan_finetune(cfg, PIPE, dataset_path=mix, output_dir=tmp_path / "ft", heldout=held,
                         seeds={"mix": 5, "heldout": PIPE.heldout.seed, "rung_b": None})
    m = json.loads(json.dumps(plan.manifest))                         # JSON-serialisable as written
    assert m["preset"] == "default"
    assert m["presets"]["default"]["flags"] == dict(PIPE.finetune.presets["default"])
    assert m["presets"]["tune_visual"]["args"] == VISUAL_FLAGS.split()
    top = m["tune_top_llm_layers"]
    assert (top["value"], top["readable"], top["allowed_unknown"], top["allow_unknown_top_llm_layers"]) == (4, True, False, False)
    assert top["source"] == str(ckpt / "config.json") and "--no-tune-llm" in top["note"]
    assert m["checkpoint"] == {"id": CKPT_ID, "revision": "TODO_PIN", "revision_pinned": False, "local_dir": str(ckpt)}
    assert m["data_mix"]["mixed"] is True and m["data_mix"]["ratio"] == 0.5 and m["data_mix"]["seed"] == 5
    assert (m["data_mix"]["nominal_count"], m["data_mix"]["shifted_count"]) == (3, 3)
    assert m["heldout"]["heldout"] == held["heldout"] and m["heldout"]["keyed_by"] == "condition"
    assert m["seeds"]["mix"] == 5 and m["seeds"]["training"]["value"] == 42 and m["seeds"]["training"]["on_cli"] is False
    assert m["rung_a"]["fields"]["random_rotation_angle"]["note"] == KEEPS_CHECKPOINT
    assert m["command"] == plan.command and m["argv"] == list(plan.argv) and m["result"] is None
    plain = plan_finetune(cfg, PIPE, dataset_path=tmp_path / "nom", output_dir=tmp_path / "ft2")
    assert plain.manifest["data_mix"] == {"dataset_path": str(tmp_path / "nom"), "mixed": False}


@pytest.mark.parametrize("config_json, reason", [
    (False, "cannot read"),
    ('{"model_type": "Gr00tN1d6"}', "has no tune_top_llm_layers"),
    ('{"tune_top_llm_layers": "four"}', "not an integer"),
    ("not json", "is not JSON"),
])
def test_an_unreadable_tune_top_llm_layers_is_refused_unless_allowed(tmp_path, config_json, reason):
    cfg, _ = setup(tmp_path, config_json=config_json)
    with pytest.raises(LauncherError, match="tune_top_llm_layers cannot be read"):
        plan_finetune(cfg, PIPE, dataset_path=tmp_path / "d", output_dir=tmp_path / "o")
    by_arg = plan_finetune(cfg, PIPE, dataset_path=tmp_path / "d", output_dir=tmp_path / "o",
                           allow_unknown_top_llm_layers=True)
    allowed = replace(PIPE, finetune=replace(PIPE.finetune, allow_unknown_top_llm_layers=True))
    by_cfg = plan_finetune(cfg, allowed, dataset_path=tmp_path / "d", output_dir=tmp_path / "o")
    for plan in (by_arg, by_cfg):
        top = plan.manifest["tune_top_llm_layers"]
        assert top["value"] is None and top["readable"] is False and top["allowed_unknown"] is True
        assert top["allow_unknown_top_llm_layers"] is True and reason in top["reason"]
    assert by_arg.command == by_cfg.command


def test_an_unmapped_checkpoint_is_refused_even_when_unknown_layers_are_allowed(tmp_path):
    cfg = backend_config(tmp_path, tmp_path)
    with pytest.raises(LauncherError, match="never downloaded"):
        plan_finetune(cfg, PIPE, dataset_path=tmp_path / "d", output_dir=tmp_path / "o", allow_unknown_top_llm_layers=True)
    cfg, _ = setup(tmp_path)
    with pytest.raises(LauncherError, match="preset"):
        plan_finetune(cfg, PIPE, dataset_path=tmp_path / "d", output_dir=tmp_path / "o", preset="lora")
    with pytest.raises(LauncherError, match="4 GPUs"):
        plan_finetune(cfg, PIPE, dataset_path=tmp_path / "d", output_dir=tmp_path / "o", num_gpus=4)


# ---- after the run -----------------------------------------------------------------------------------

def test_the_parameter_counts_and_final_config_go_into_the_manifest(tmp_path):
    cfg, _ = setup(tmp_path)
    out = tmp_path / "ft"
    shutil.copytree(FIXTURE_RUN, out)
    plan = plan_finetune(cfg, PIPE, dataset_path=tmp_path / "d", output_dir=out)
    done = record_run(plan.manifest, out / "finetune.log", out)
    r = done["result"]
    assert r["parameters"] == {"trainable": 1_019_203_584, "trainable_percent": 31.0, "total": 3_288_135_680}
    assert r["final_model_config"]["tune_top_llm_layers"] == 4 and r["final_model_config"]["tune_vlln"] is True
    assert r["tune_top_llm_layers_final"] == 4 and r["tune_top_llm_layers_consistent"] is True
    assert r["complete"] is True and r["missing"] == []
    assert done["command"] == plan.command and plan.manifest["result"] is None          # the plan is not mutated


def test_a_run_without_its_outputs_is_recorded_as_incomplete(tmp_path):
    cfg, _ = setup(tmp_path)
    plan = plan_finetune(cfg, PIPE, dataset_path=tmp_path / "d", output_dir=tmp_path / "ft")
    (tmp_path / "ft").mkdir()
    (tmp_path / "ft" / "finetune.log").write_text("Model Config: ...\n")
    r = record_run(plan.manifest, tmp_path / "ft" / "finetune.log", tmp_path / "ft")["result"]
    assert r["complete"] is False and r["parameters"]["trainable"] is None
    assert any("Trainable parameters" in m for m in r["missing"]) and any("final_model_config" in m for m in r["missing"])
    assert r["tune_top_llm_layers_consistent"] is None


def test_ranks_reporting_different_counts_are_an_error():
    log = "Trainable parameters: 10 (1.00%)\nTrainable parameters: 11 (1.10%)\n"
    with pytest.raises(ValueError, match="different parameter counts"):
        parse_parameter_counts(log)
    same = "Total parameters: 1,000\n" * 2 + "Trainable parameters: 10 (1.00%)\n" * 2
    assert parse_parameter_counts(same) == {"trainable": 10, "trainable_percent": 1.0, "total": 1000}


class FakeFinetune(ArenaRunner):
    def __init__(self):
        self.calls = []

    def run(self, inv: RunInvocation) -> None:
        self.calls.append(inv)
        assert (inv.output_dir / MANIFEST_FILE).is_file()                 # written before the run starts
        shutil.copyfile(FIXTURE_RUN / "finetune.log", inv.log_path)
        (inv.output_dir / "experiment_cfg").mkdir()
        shutil.copyfile(FIXTURE_RUN / "experiment_cfg" / "final_model_config.json",
                        inv.output_dir / "experiment_cfg" / "final_model_config.json")


def test_run_finetune_writes_the_manifest_before_and_after(tmp_path):
    cfg, _ = setup(tmp_path)
    plan = plan_finetune(cfg, PIPE, dataset_path=tmp_path / "d", output_dir=tmp_path / "ft")
    runner = FakeFinetune()
    done = run_finetune(plan, runner, timeout_s=PIPE.finetune.timeout_s)
    inv = runner.calls[0]
    assert list(inv.argv) == list(plan.argv) and inv.cwd == cfg.gr00t_root and inv.env == {"CUDA_VISIBLE_DEVICES": "0"}
    assert inv.timeout_s == 172800
    saved = json.loads((tmp_path / "ft" / MANIFEST_FILE).read_text())
    assert saved == json.loads(json.dumps(done)) and saved["result"]["parameters"]["trainable"] == 1_019_203_584
