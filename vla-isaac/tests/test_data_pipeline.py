"""Phase 6: the pipeline config, Rung A arguments, the held-out split and the mixer."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from vla_isaac.config import ConfigError, read_yaml
from vla_isaac.data.config import HeldoutConfig, MixConfig, PipelineConfig, RungAConfig, condition_key
from vla_isaac.data.heldout import EpisodeRef, SplitError, split_conditions, split_episodes
from vla_isaac.data.mixer import (MIX_FILE, LeRobotSource, MixError, mix_counts, select_mix,
                                  write_mixed_dataset)
from vla_isaac.data.rung_a import KEEPS_CHECKPOINT, rung_a_args, rung_a_manifest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "config" / "finetune_gr1_fridge.yaml"
CKPT = ROOT / "tests" / "fixtures" / "checkpoint_gr1"


def pipeline_dict() -> dict:
    return read_yaml(PIPELINE)


# ---- config ------------------------------------------------------------------------------------------

def test_the_pilot_pipeline_config_loads():
    p = PipelineConfig.load(PIPELINE)
    assert p.mix.ratio == 0.5 and p.rung_b.converter.action_name_sim == "processed_actions"
    assert p.rung_b.converter.template == ROOT / "config" / "arena" / "gr1_manip_ranch_bottle_config.yaml"
    assert set(p.finetune.presets) == {"default", "tune_visual"} and set(p.finetune.hardware) == {1, 8}
    assert 50 <= p.rung_b.episodes <= 100


def test_the_mix_ratio_defaults_to_one_half():
    d = pipeline_dict()
    del d["mix"]["ratio"]
    assert PipelineConfig.from_dict(d, PIPELINE.parent).mix.ratio == 0.5
    assert MixConfig(seed=0, nominal_dataset="x").ratio == 0.5


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(extra=1), "unknown keys"),
    (lambda d: d["mix"].update(ratio=1.0), "mix.ratio"),
    (lambda d: d["mix"].update(ratio=0.0), "mix.ratio"),
    (lambda d: d["finetune"].update(preset="lora"), "preset"),
    (lambda d: d["finetune"].update(num_gpus=4), "hardware"),
    (lambda d: d["finetune"]["presets"]["default"].pop("tune_llm"), "missing keys"),
    (lambda d: d["finetune"].update(allow_unknown_top_llm_layers="yes"), "allow_unknown"),
    (lambda d: d["rung_a"]["color_jitter_params"].update(blur=1.0), "color_jitter_params"),
    (lambda d: d["rung_a"].update(random_rotation_angle=-5), "random_rotation_angle"),
    (lambda d: d["rung_b"].update(episodes=120), "rung_b"),
    (lambda d: d["rung_b"]["conditions"].append({}), "shifted"),
    (lambda d: d["heldout"].update(fraction=1.0), "heldout.fraction"),
])
def test_bad_pipeline_configs_are_refused(edit, match):
    d = pipeline_dict()
    edit(d)
    with pytest.raises(ConfigError, match=match):
        PipelineConfig.from_dict(d, PIPELINE.parent)


# ---- Rung A ------------------------------------------------------------------------------------------

def test_rung_a_gives_arenas_jitter_flags_in_config_order():
    p = PipelineConfig.load(PIPELINE)
    assert rung_a_args(p.rung_a) == ["--color-jitter-params", "brightness", "0.3", "contrast", "0.4",
                                     "saturation", "0.5", "hue", "0.08"]
    both = RungAConfig({"hue": 0.1, "brightness": 0.2}, 15)
    assert rung_a_args(both) == ["--color-jitter-params", "hue", "0.1", "brightness", "0.2",
                                 "--random-rotation-angle", "15"]


def test_a_none_value_passes_no_flag_and_the_manifest_says_the_checkpoint_setting_is_kept():
    m = rung_a_manifest(RungAConfig({"brightness": 0.3}, None), CKPT)
    rot = m["fields"]["random_rotation_angle"]
    assert "--random-rotation-angle" not in m["args"] and rot["passed"] is False
    assert rot["note"] == KEEPS_CHECKPOINT and "keeps the checkpoint's own setting" in rot["note"]
    assert rot["checkpoint_value"] is None                       # the FIXTURE processor_config.json has null
    assert m["fields"]["color_jitter_params"] == {"value": {"brightness": 0.3}, "flag": "--color-jitter-params",
                                                  "passed": True}
    none = rung_a_manifest(RungAConfig(None, None))
    assert none["args"] == [] and none["fields"]["color_jitter_params"]["checkpoint_value"] == "unread"
    assert none["fields"]["color_jitter_params"]["note"] == KEEPS_CHECKPOINT


# ---- held-out split ----------------------------------------------------------------------------------

SHIFTED = [{"lighting.intensity": "100.0"}, {"lighting.intensity": "300.0"}, {"lighting.hdr": "office"},
           {"lighting.intensity": "100.0", "lighting.hdr": "office"}]
HELD = HeldoutConfig(seed=0, fraction=0.5)


def test_the_split_is_by_condition_deterministic_and_order_free():
    a = split_conditions([{}] + SHIFTED, HELD)
    b = split_conditions(list(reversed(SHIFTED)) + [{}], HELD)
    assert a == b
    assert len(a.train) == 2 and len(a.heldout) == 2 and not set(a.train) & set(a.heldout)
    assert set(a.train) | set(a.heldout) == {condition_key(c) for c in SHIFTED}
    assert a.regression == ("nominal",) and "nominal" not in a.train + a.heldout
    assert a.to_dict()["keyed_by"] == "condition"


def test_the_seed_drives_the_split():
    splits = {split_conditions(SHIFTED, HeldoutConfig(seed=s, fraction=0.5)).heldout for s in range(20)}
    assert len(splits) > 1
    assert split_conditions(SHIFTED, HeldoutConfig(seed=7, fraction=0.5)) == \
        split_conditions(SHIFTED, HeldoutConfig(seed=7, fraction=0.5))


@pytest.mark.parametrize("fraction, held", [(0.1, 1), (0.5, 2), (0.6, 2), (0.9, 3)])
def test_at_least_one_condition_on_each_side(fraction, held):
    assert len(split_conditions(SHIFTED, HeldoutConfig(0, fraction)).heldout) == held


def test_too_few_or_duplicate_conditions_are_refused():
    with pytest.raises(SplitError, match="two shifted"):
        split_conditions([{}, SHIFTED[0]], HELD)
    with pytest.raises(SplitError, match="duplicate"):
        split_conditions([SHIFTED[0], dict(SHIFTED[0])], HELD)


def episodes_under(conditions, per=5, dataset_of=lambda i: f"rb{i}"):
    return [EpisodeRef(dataset_of(i), k, c) for i, c in enumerate(conditions) for k in range(per)]


def test_no_episode_or_condition_is_on_both_sides_and_nominal_is_only_in_the_regression_set():
    eps = episodes_under([{}] + SHIFTED)
    random.Random(3).shuffle(eps)
    s = split_episodes(eps, HELD)
    train, held, reg = ({e.id for e in x} for x in (s.train, s.heldout, s.regression))
    assert not train & held and not train & reg and not held & reg
    assert train | held | reg == {e.id for e in eps}
    assert {e.condition_key for e in s.train}.isdisjoint({e.condition_key for e in s.heldout})
    assert {e.condition_key for e in s.regression} == {"nominal"} and len(s.regression) == 5
    assert all(e.condition for e in s.heldout) and all(e.condition for e in s.train)
    assert s.conditions == split_conditions([{}] + SHIFTED, HELD)          # the episode split follows the conditions
    assert split_episodes(list(reversed(eps)), HELD) == s
    d = s.to_dict()
    assert set(d["heldout_episodes"]) == held and set(d["regression_episodes"]) == reg


def test_one_episode_under_two_conditions_is_refused():
    eps = episodes_under(SHIFTED, dataset_of=lambda i: "same")
    with pytest.raises(SplitError, match="two conditions"):
        split_episodes(eps, HELD)


# ---- mixer: counts and determinism -------------------------------------------------------------------

@pytest.mark.parametrize("n_nom, n_shift, ratio, expect", [
    (1000, 100, 0.5, (100, 100)),
    (1000, 100, 0.25, (300, 100)),
    (30, 100, 0.5, (30, 30)),
    (100, 100, 0.75, (33, 100)),
    (10, 100, 0.75, (10, 30)),
    (100, 7, 0.5, (7, 7)),
    (1000, 3, 0.4, (5, 3)),               # 3 * 0.6 / 0.4 = 4.5 -> 5 (half up, exact rationals)
])
def test_mix_counts_are_exact(n_nom, n_shift, ratio, expect):
    assert mix_counts(n_nom, n_shift, ratio) == expect


def pool(name, n, condition=None):
    return [EpisodeRef(name, k, condition or {}) for k in range(n)]


DARK_C = {"lighting.intensity": "100.0"}


def test_the_mix_is_deterministic_for_a_seed_and_has_the_exact_counts():
    nominal, shifted = pool("nominal", 400), pool("rb", 120, DARK_C)
    cfg = MixConfig(seed=11, nominal_dataset="nominal", ratio=0.5)
    a = select_mix(nominal, shifted, cfg)
    assert len(a.nominal) == 120 and len(a.shifted) == 120 and a.to_dict()["achieved_ratio"] == 0.5
    assert {e.dataset for e in a.nominal} == {"nominal"} and {e.dataset for e in a.shifted} == {"rb"}
    assert a == select_mix(list(reversed(nominal)), list(reversed(shifted)), cfg)     # input order is irrelevant
    assert a.nominal != select_mix(nominal, shifted, replace(cfg, seed=12)).nominal
    q = select_mix(nominal, shifted, replace(cfg, ratio=0.25))
    assert (len(q.nominal), len(q.shifted)) == (360, 120)


@pytest.mark.parametrize("nominal, shifted", [([], pool("rb", 5)), (pool("nominal", 5), [])])
def test_an_empty_source_is_refused(nominal, shifted):
    with pytest.raises(MixError, match="both sources"):
        select_mix(nominal, shifted, MixConfig(seed=0, nominal_dataset="nominal"))


def test_an_episode_in_both_sources_is_refused():
    with pytest.raises(MixError, match="both sources"):
        select_mix(pool("x", 3), pool("x", 3), MixConfig(seed=0, nominal_dataset="x"))


# ---- mixer: the merged LeRobot directory -------------------------------------------------------------

def write_lerobot(root: Path, n: int, *, task="Place the bottle.", tag="n", modality=None) -> Path:
    """A FIXTURE GR00T LeRobot v2 directory: meta files plus byte-stamped parquet and mp4 files."""
    info = {"codebase_version": "v2.0", "robot_type": "gr1", "chunks_size": 2, "fps": 50,
            "total_episodes": n, "total_frames": 10 * n, "total_videos": n, "total_chunks": (n - 1) // 2 + 1,
            "splits": {"train": f"0:{n}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"observation.state": {"dtype": "float64"}, "action": {"dtype": "float64"},
                         "observation.images.ego_view": {"dtype": "video"}}}
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text("".join(json.dumps({"episode_index": k, "tasks": [task], "length": 10 + k}) + "\n"
                                                 for k in range(n)))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    (meta / "modality.json").write_text(json.dumps(modality or {"state": {"left_arm": {"start": 0, "end": 7}}}))
    (meta / "stats.json").write_text("{}")
    for k in range(n):
        c = k // 2
        p = root / f"data/chunk-{c:03d}/episode_{k:06d}.parquet"
        v = root / f"videos/chunk-{c:03d}/observation.images.ego_view/episode_{k:06d}.mp4"
        for f in (p, v):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(f"{tag}{k}{f.suffix}".encode())
    return root


def test_the_mixed_dataset_renumbers_files_and_meta_and_drops_the_stats(tmp_path):
    nom = LeRobotSource("nominal", write_lerobot(tmp_path / "nom", 6, tag="n"), {})
    rb = LeRobotSource("rb", write_lerobot(tmp_path / "rb", 3, tag="s"), DARK_C)
    sel = select_mix(nom.episodes(), rb.episodes(), MixConfig(seed=0, nominal_dataset="nominal"))
    out = write_mixed_dataset(sel, [nom, rb], tmp_path / "mix")
    info = json.loads((out / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_videos"], info["total_chunks"], info["splits"]) == (6, 6, 3, {"train": "0:6"})
    lines = [json.loads(x) for x in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [x["episode_index"] for x in lines] == list(range(6))
    assert info["total_frames"] == sum(x["length"] for x in lines)
    mapping = json.loads((out / "meta" / MIX_FILE).read_text())
    assert mapping["nominal_count"] == 3 and mapping["shifted_count"] == 3 and mapping["seed"] == 0
    for row in mapping["episodes"]:                                   # every file is the source episode's
        new, old = row["episode_index"], row["source_episode_index"]
        tag = "n" if row["source"] == "nominal" else "s"
        assert row["role"] == ("nominal" if tag == "n" else "shifted")
        assert (out / f"data/chunk-{new // 2:03d}/episode_{new:06d}.parquet").read_bytes() == f"{tag}{old}.parquet".encode()
        assert (out / f"videos/chunk-{new // 2:03d}/observation.images.ego_view/episode_{new:06d}.mp4").read_bytes() == \
            f"{tag}{old}.mp4".encode()
        assert lines[new]["length"] == 10 + old
    assert [r["condition"] for r in mapping["episodes"] if r["role"] == "shifted"] == [DARK_C] * 3
    assert not (out / "meta" / "stats.json").exists()                 # GR00T regenerates it; a stale copy would stick
    assert (out / "meta" / "tasks.jsonl").read_text() == (nom.path / "meta" / "tasks.jsonl").read_text()
    with pytest.raises(MixError, match="exists"):
        write_mixed_dataset(sel, [nom, rb], out)


def test_sources_with_different_instructions_or_modality_are_refused(tmp_path):
    nom = LeRobotSource("nominal", write_lerobot(tmp_path / "nom", 2), {})
    other_task = LeRobotSource("rb", write_lerobot(tmp_path / "rb", 2, task="Place the sauce bottle."), DARK_C)
    other_mod = LeRobotSource("rb2", write_lerobot(tmp_path / "rb2", 2, modality={"state": {}}), DARK_C)
    for src, match in ((other_task, "tasks.jsonl"), (other_mod, "modality.json")):
        sel = select_mix(nom.episodes(), src.episodes(), MixConfig(seed=0, nominal_dataset="nominal"))
        with pytest.raises(MixError, match=match):
            write_mixed_dataset(sel, [nom, src], tmp_path / f"mix_{src.id}")
        assert not (tmp_path / f"mix_{src.id}").exists()
