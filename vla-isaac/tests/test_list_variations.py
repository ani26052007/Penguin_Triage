"""--list_variations output -> factor table (FIXTURE catalogue in the code's print format)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vla_isaac.config import load_task_config
from vla_isaac.scene.factors import FactorTable
from vla_isaac.scene.list_variations import catalog_to_table, main, parse_catalog

from tests.helpers import PILOT_TASK

FIXTURE = Path(__file__).parent / "fixtures" / "list_variations_gr1_fridge.txt"
NAMES = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "factor_names_gr1_fridge.yaml").read_text())


def test_parse_reads_assets_variations_kinds_and_fields_and_ignores_log_lines():
    entries = parse_catalog(FIXTURE.read_text())
    assert [e.path for e in entries] == ["light.hdr_image", "light.intensity", "light.color", "light.color_temperature",
                                         "ranch_dressing_hope_robolab.mass", "ranch_dressing_hope_robolab.disappear",
                                         "ranch_dressing_hope_robolab.pose"]
    by = {e.path: e for e in entries}
    assert by["light.intensity"].kind == "build_time" and by["ranch_dressing_hope_robolab.mass"].kind == "run_time"
    assert by["light.intensity"].fields == {"sampler_cfg.high": [2000.0], "sampler_cfg.low": [100.0]}
    assert by["light.hdr_image"].fields == {"hdr_names": [], "sampler_cfg": {}}
    assert by["light.intensity"].enabled_default == "False"


def test_inferred_table_regenerates_the_committed_pilot_table():
    table, unsupported = catalog_to_table(parse_catalog(FIXTURE.read_text()), NAMES)
    committed = FactorTable.load(load_task_config(PILOT_TASK).factor_table)
    assert {n: s for n, s in table.specs.items() if n in committed.specs} == committed.specs
    assert table.spec("ranch_dressing_hope_robolab.disappear").sampler == "bernoulli"
    assert unsupported == ["ranch_dressing_hope_robolab.pose"]          # never guessed


def test_cli_writes_a_loadable_table(tmp_path):
    names = tmp_path / "names.yaml"
    names.write_text(yaml.safe_dump(NAMES))
    out = tmp_path / "table.yaml"
    assert main(["--input", str(FIXTURE), "--output", str(out), "--names", str(names)]) == 0
    assert FactorTable.load(out).spec("lighting.color").dims == 3


def test_errors():
    with pytest.raises(ValueError, match="header"):
        parse_catalog("nothing printed\n")
    with pytest.raises(ValueError, match="does not have"):
        catalog_to_table(parse_catalog(FIXTURE.read_text()), {"light.direction": "lighting.direction"})
