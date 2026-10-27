"""Factor table: total, injective, unknown factors raise, restore(i) removes exactly one factor."""

from __future__ import annotations

import pytest

from router.schemas import Factor, SceneSpec

from vla_isaac.config import load_task_config
from vla_isaac.scene.factors import FactorSpec, FactorTable, FactorTableError, UnknownFactor, hydra_override

from tests.helpers import PILOT_TASK

TABLE = FactorTable.load(load_task_config(PILOT_TASK).factor_table)
VALUES = {"lighting.intensity": "150.0", "lighting.hdr": "home_office_robolab", "lighting.color": "1.0,0.5,0.2",
          "lighting.color_temperature": "2500", "object.mass": "0.8"}
FULL = SceneSpec("gr1_fridge_ranch", tuple(Factor(n, v) for n, v in VALUES.items()))


def test_the_pilot_table_is_total_over_its_factors_and_every_value_maps():
    assert set(TABLE.specs) == set(VALUES)
    for name, value in VALUES.items():
        a = TABLE.assignments(SceneSpec("s", (Factor(name, value),)))
        assert a[f"{TABLE.spec(name).path}.enabled"] is True and len(a) >= 2


def test_mapping_is_injective_no_two_factors_share_a_variation_or_an_override_path():
    paths = [TABLE.spec(n).path for n in VALUES]
    assert len(paths) == len(set(paths))
    per_factor = [set(TABLE.assignments(SceneSpec("s", (Factor(n, v),)))) for n, v in VALUES.items()]
    for i, a in enumerate(per_factor):
        for b in per_factor[i + 1:]:
            assert not a & b
    with pytest.raises(FactorTableError, match="injective"):
        FactorTable({"a": FactorSpec("a", "light", "intensity", "build_time", "uniform"),
                     "b": FactorSpec("b", "light", "intensity", "build_time", "uniform")})


def test_an_unknown_factor_raises_never_falls_back_to_nominal():
    with pytest.raises(UnknownFactor, match="material"):
        TABLE.overrides(SceneSpec("s", (Factor("lighting.intensity", "100"), Factor("material(mug)", "glossy"))))


def test_fixed_values_hold_the_sampler_at_one_point():
    a = TABLE.assignments(FULL)
    assert a["light.intensity.sampler_cfg.low"] == a["light.intensity.sampler_cfg.high"] == [150.0]
    assert a["light.color.sampler_cfg.low"] == [1.0, 0.5, 0.2]
    assert a["light.hdr_image.hdr_names"] == ["home_office_robolab"]
    assert a["ranch_dressing_hope_robolab.mass.sampler_cfg.high"] == [0.8]
    assert TABLE.overrides(SceneSpec("s", (Factor("lighting.intensity", "100"),))) == [
        "light.intensity.enabled=true", "light.intensity.sampler_cfg.low=[100.0]",
        "light.intensity.sampler_cfg.high=[100.0]"]
    assert hydra_override("light.hdr_image.hdr_names", ["x"]) == 'light.hdr_image.hdr_names=["x"]'


@pytest.mark.parametrize("name", sorted(VALUES))
def test_restore_removes_exactly_that_factors_overrides(name):
    before = TABLE.overrides(FULL)
    after = TABLE.overrides(FULL.restore(name))
    own = TABLE.overrides(SceneSpec("s", (Factor(name, VALUES[name]),)))
    assert set(before) - set(after) == set(own) and set(after) <= set(before)
    assert len(before) - len(after) == len(own)
    assert TABLE.overrides(FULL.restore_all()) == [] and TABLE.overrides(SceneSpec("s")) == []


@pytest.mark.parametrize("name,bad", [("lighting.intensity", "dark"), ("lighting.color", "1,2"),
                                      ("object.mass", "1,2"), ("lighting.hdr", "")])
def test_malformed_values_raise(name, bad):
    with pytest.raises(ValueError):
        TABLE.assignments(SceneSpec("s", (Factor(name, bad),)))


def test_table_validation():
    with pytest.raises(FactorTableError):
        FactorTable.from_dict({"factors": {"x": {"asset": "light", "variation": "intensity", "kind": "sometimes",
                                                 "sampler": "uniform"}}})
    with pytest.raises(FactorTableError):
        FactorTable.from_dict({"factors": {"x": {"asset": "light", "variation": "hdr_image", "kind": "build_time",
                                                 "sampler": "choice"}}})          # no values_field
    with pytest.raises(FactorTableError):
        FactorTable.from_dict({"factors": {"x": {"asset": "light", "variation": "intensity", "kind": "build_time",
                                                 "sampler": "uniform", "typo": 1}}})
    assert FactorTable.from_dict(TABLE.to_dict()).to_dict() == TABLE.to_dict()
    assert TABLE.build_time_factors(FULL) == ("lighting.color", "lighting.color_temperature", "lighting.hdr",
                                              "lighting.intensity")
