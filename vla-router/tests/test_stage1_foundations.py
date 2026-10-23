"""Stage 1 foundations: NominalStats, trace records, TraceSource (plan.md 3.2.3)."""

import json

import pytest

from router.prereq.trace import (EpisodeTrace, FrameStats, InMemoryTraceSource, StepPair, TraceUnavailable)
from router.schemas import Mode
from router.stage1.base import nrmse, pearson, saturating
from router.stage1.nominal import MetricNominal, MissingNominal, NominalStats
from router.stage1.synthetic import Faults, make_incident, make_nominal, make_trace, make_traces


# ---- strength / nominal ---------------------------------------------------- #

def test_saturating_is_zero_half_and_below_one():
    assert saturating(0) == 0.0
    assert saturating(1) == pytest.approx(0.5)
    assert saturating(3) == pytest.approx(0.9)
    assert saturating(1e300) == 1.0
    assert saturating(-5) == 0.0
    xs = [saturating(r / 10) for r in range(0, 100)]
    assert xs == sorted(xs) and max(xs) < 1.0


def _stats(fire_z=4.0):
    return NominalStats({"m": MetricNominal(mean=10.0, std=1.0, n=30)}, fire_z=fire_z)


def test_deviation_fires_exactly_at_fire_z_and_strength_is_half_there():
    n = _stats()
    d = n.deviation("m", 14.0, "high")
    assert d.fires and d.z == pytest.approx(4.0) and d.strength == pytest.approx(0.5)
    d = n.deviation("m", 13.9, "high")
    assert not d.fires and d.strength < 0.5


def test_deviation_sides():
    n = _stats()
    assert n.deviation("m", 16.0, "high").fires and not n.deviation("m", 16.0, "low").fires
    assert n.deviation("m", 4.0, "low").fires and not n.deviation("m", 4.0, "high").fires
    assert n.deviation("m", 4.0, "both").fires and n.deviation("m", 16.0, "both").fires
    wrong_side = n.deviation("m", 4.0, "high")
    assert wrong_side.z < 0 and wrong_side.strength == 0.0
    with pytest.raises(ValueError):
        n.deviation("m", 1.0, "sideways")
    with pytest.raises(ValueError):
        n.deviation("m", float("nan"))


def test_fire_z_override_and_floor():
    n = _stats()
    assert not n.deviation("m", 13.0, "high").fires
    assert n.deviation("m", 13.0, "high", fire_z=2.0).fires
    flat = NominalStats({"c": MetricNominal(mean=0.0, std=0.0, n=30, min_std=0.25)})
    assert flat.deviation("c", 0.9, "high").z == pytest.approx(3.6)       # the floor sets the scale
    assert flat.deviation("c", 1.0, "high").fires


def test_from_samples_and_json_round_trip():
    n = NominalStats.from_samples({"a": [1.0, 2.0, 3.0]}, vectors={"v": [0.1, 0.2]}, fire_z=5.0, min_std={"a": 0.5})
    assert n.metric("a").mean == pytest.approx(2.0) and n.metric("a").std == pytest.approx(1.0)
    assert n.metric("a").min_std == 0.5 and n.fire_z == 5.0
    again = NominalStats.from_dict(json.loads(json.dumps(n.to_dict())))
    assert again.to_dict() == n.to_dict()
    with pytest.raises(ValueError):
        NominalStats.from_samples({"a": [1.0]})


def test_missing_nominal_is_reported_in_one_error():
    n = _stats()
    with pytest.raises(MissingNominal) as e:
        n.require(["m", "x", "y"], ["v"])
    assert "'x'" in str(e.value) and "'y'" in str(e.value) and "'v'" in str(e.value) and "'m'" not in str(e.value)


def test_pearson_and_nrmse_edge_cases():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) is None and pearson([1], [1]) is None and pearson([1, 2], [1]) is None
    assert nrmse([0, 1, 0, 1], [0, 1, 0, 1]) == 0.0
    assert nrmse([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)
    assert nrmse([1, 1, 1], [1, 1, 1]) is None


# ---- trace records ---------------------------------------------------------- #

def test_trace_round_trips_through_json():
    tr = make_trace(3, Mode.INCIDENT, Faults(server_delay_max=4))
    again = EpisodeTrace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert again == tr


def test_trace_validation():
    with pytest.raises(ValueError):
        FrameStats((10, 10, 3), (0.1, 0.2), 0.0, 1.0)                 # 2 means for 3 channels
    with pytest.raises(ValueError):
        StepPair(obs_step=5, applied_step=6, chunk_offset=2)         # applied before the chunk allows
    with pytest.raises(ValueError):
        EpisodeTrace(1, Mode.CLEAN, joint_cmd=[[0.0]], joint_pos=[])
    assert StepPair(10, 14, 3).extra_delay == 1


def test_in_memory_source_resolves_log_refs():
    inc = make_incident()
    src = InMemoryTraceSource({"clean": make_traces(2, Mode.CLEAN), "incident": make_traces(2, Mode.INCIDENT)})
    assert [t.mode for t in src.load(inc, Mode.CLEAN)] == [Mode.CLEAN] * 2
    assert [t.mode for t in src.load(inc, Mode.INCIDENT)] == [Mode.INCIDENT] * 2


def test_in_memory_source_raises_instead_of_returning_nothing():
    with pytest.raises(TraceUnavailable):
        InMemoryTraceSource({}).load(make_incident(log_refs={}), Mode.CLEAN)              # no ref
    with pytest.raises(TraceUnavailable):
        InMemoryTraceSource({}).load(make_incident(), Mode.CLEAN)                         # ref points nowhere
    with pytest.raises(ValueError):
        InMemoryTraceSource({"clean": make_traces(1, Mode.INCIDENT)}).load(make_incident(), Mode.CLEAN)


def test_synthetic_nominal_covers_every_metric_and_is_deterministic():
    a, b = make_nominal(), make_nominal()
    assert a.to_dict() == b.to_dict()
    assert make_trace(1, Mode.CLEAN) == make_trace(1, Mode.CLEAN)
    assert make_trace(1, Mode.CLEAN) != make_trace(2, Mode.CLEAN)
