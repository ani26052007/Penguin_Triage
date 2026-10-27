"""Phase 5: strip one factor's key from an Arena recorded-variation JSONL so that Arena replays it with
that factor disabled. The FIXTURE records follow ``A:isaaclab_arena/recording/common_terms.py``; the
stripped output is also checked with Arena's own validator (pure Python, run in a subprocess so Arena
never lands on this process's ``sys.path``)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from router.schemas import Factor, SceneSpec

from vla_isaac.config import load_task_config
from vla_isaac.replay.recorded_variations import (RecordedVariationError, enabled_keys, main, read_records,
                                                  strip_factor, strip_key)
from vla_isaac.scene.factors import FactorTable

from tests.helpers import PILOT_TASK, REPO

ARENA = REPO.parent / "external" / "IsaacLab-Arena"
TABLE = FactorTable.load(load_task_config(PILOT_TASK).factor_table)
SCENE = SceneSpec("gr1_fridge_ranch", (Factor("lighting.intensity", "100.0"), Factor("lighting.hdr", "home_office_robolab"),
                                       Factor("object.mass", "0.4")))
INTENSITY, HDR, MASS = "light.intensity", "light.hdr_image", "ranch_dressing_hope_robolab.mass"


def record(i: int, **variations) -> dict:
    """FIXTURE episode record (VERIFY(V-var-record): the recorded value format)."""
    return {"seed": 0, "env_id": 0, "episode_in_env": i, "success": i % 2 == 0, "episode_length": 300,
            "variations": variations, "placement": {"ranch_dressing_hope_robolab": [0.1 * i, 0.2, 0.9]}}


def write(path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def recording(tmp_path):
    src = tmp_path / "episode_results_rebuild0.jsonl"
    write(src, [record(i, **{INTENSITY: [100.0], HDR: ["home_office_robolab"], MASS: [0.4 + 0.01 * i]})
                for i in range(3)])
    return src


def test_strip_removes_exactly_one_key_and_keeps_everything_else(recording, tmp_path):
    out = strip_key(recording, tmp_path / "out.jsonl", INTENSITY, {HDR, MASS})
    before, after = read_records(recording), read_records(out)
    assert len(after) == 3
    for b, a in zip(before, after):
        assert INTENSITY not in a["variations"]
        assert a["variations"] == {k: v for k, v in b["variations"].items() if k != INTENSITY}
        assert {k: v for k, v in a.items() if k != "variations"} == {k: v for k, v in b.items() if k != "variations"}


def test_a_remaining_key_without_an_enabled_variation_is_refused_and_named(recording, tmp_path):
    dst = tmp_path / "out.jsonl"
    with pytest.raises(RecordedVariationError, match=r"'ranch_dressing_hope_robolab.mass' \(first in record 0\)"):
        strip_key(recording, dst, INTENSITY, {HDR})                     # mass is recorded but not enabled
    assert not dst.exists()                                             # nothing written
    write(recording, [record(0, **{INTENSITY: [100.0]}), record(1, **{INTENSITY: [100.0], "light.color": [1, 0, 0]})])
    with pytest.raises(RecordedVariationError, match=r"'light.color' \(first in record 1\)") as exc:
        strip_key(recording, dst, INTENSITY, set())
    assert "Arena asserts" in str(exc.value)


def test_nothing_to_strip_or_a_still_enabled_key_is_refused(recording, tmp_path):
    with pytest.raises(RecordedVariationError, match="no record holds variation 'light.color'"):
        strip_key(recording, tmp_path / "o.jsonl", "light.color", {INTENSITY, HDR, MASS})
    with pytest.raises(RecordedVariationError, match="still enabled"):
        strip_key(recording, tmp_path / "o.jsonl", INTENSITY, {INTENSITY, HDR, MASS})
    with pytest.raises(RecordedVariationError, match="must not overwrite"):
        strip_key(recording, recording, INTENSITY, {HDR, MASS})


def test_records_are_read_as_arena_reads_them(tmp_path):
    bad = tmp_path / "dup.jsonl"
    bad.write_text('{"variations": {"light.intensity": [1.0], "light.intensity": [2.0]}}\n')
    with pytest.raises(RecordedVariationError, match="duplicate key"):
        read_records(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(RecordedVariationError, match="no episode records"):
        read_records(empty)


def test_strip_factor_uses_the_factor_table_and_the_restored_scene(recording, tmp_path):
    assert enabled_keys(TABLE, SCENE) == {INTENSITY, HDR, MASS}
    out = strip_factor(recording, tmp_path / "out.jsonl", TABLE, SCENE, "lighting.intensity")
    assert all(set(r["variations"]) == {HDR, MASS} for r in read_records(out))
    with pytest.raises(RecordedVariationError, match="mass"):           # the replay scene lacks object.mass
        strip_factor(recording, tmp_path / "o2.jsonl", TABLE, SCENE.restore("object.mass"), "lighting.intensity")
    with pytest.raises(KeyError):
        strip_factor(recording, tmp_path / "o3.jsonl", TABLE, SCENE, "lighting.color")


ARENA_CHECK = """
import sys
sys.path.insert(0, sys.argv[1])
from isaaclab_arena.variations.recorded_variation_samples import (load_rebuild_variation_record,
                                                                  validate_recorded_variation_sample_keys)
enabled, build_time = set(sys.argv[3].split(",")) - {""}, set(sys.argv[4].split(",")) - {""}
rec = load_rebuild_variation_record(sys.argv[2], build_time_variation_keys=build_time)
validate_recorded_variation_sample_keys(rec, enabled)
print("accepted")
"""


def arena_accepts(path, enabled, build_time) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", "-c", ARENA_CHECK, str(ARENA), str(path), ",".join(sorted(enabled)),
                           ",".join(sorted(build_time))], capture_output=True, text=True)


@pytest.mark.skipif(not (ARENA / "isaaclab_arena" / "variations" / "recorded_variation_samples.py").is_file(),
                    reason="Arena checkout not present")
def test_arena_rejects_the_original_and_accepts_the_stripped_recording(recording, tmp_path):
    enabled = enabled_keys(TABLE, SCENE.restore("lighting.intensity"))
    build_time = {TABLE.spec(f.name).path for f in SCENE.factors if TABLE.spec(f.name).kind == "build_time"} & enabled
    rejected = arena_accepts(recording, enabled, build_time)
    assert rejected.returncode != 0 and "no enabled variation" in rejected.stderr and INTENSITY in rejected.stderr
    out = strip_factor(recording, tmp_path / "out.jsonl", TABLE, SCENE, "lighting.intensity")
    accepted = arena_accepts(out, enabled, build_time)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == "accepted"


def test_the_cli(recording, tmp_path, capsys):
    assert main(["--input", str(recording), "--output", str(tmp_path / "c.jsonl"), "--strip", INTENSITY,
                 "--enabled", HDR, MASS]) == 0
    assert all(INTENSITY not in r["variations"] for r in read_records(tmp_path / "c.jsonl"))
