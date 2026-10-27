"""Router change 5: ``SimHealth`` fields a backend does not measure are None, and the sanity
check says "not measured" instead of reading them as healthy."""

from dataclasses import replace

import pytest

from router import vocab as V
from router.prereq.trace import EpisodeTrace, InMemoryTraceSource, SimHealth
from router.schemas import Mode
from router.stage1.registry import run_stage1
from router.stage1.sanity import M_PENETRATION, SanityCheck, not_measured, sanity_metrics
from router.stage1.synthetic import Faults, make_incident, make_nominal, make_traces

NOMINAL = make_nominal()
INCIDENT = make_incident()


def unmeasured(traces, every: int = 1):
    """``traces`` with ``max_penetration`` None in every ``every``-th episode."""
    return [replace(t, sim=replace(t.sim, max_penetration=None)) if i % every == 0 else t
            for i, t in enumerate(traces)]


def check(traces):
    return SanityCheck(NOMINAL, InMemoryTraceSource({"clean": traces}))


def test_none_means_not_measured_and_round_trips():
    h = SimHealth(nan_fraction=0.1, max_penetration=None, object_initial_pos=(1, 2, 3))
    assert h.to_dict()["max_penetration"] is None
    assert SimHealth.from_dict(h.to_dict()) == h
    assert SimHealth.from_dict({"nan_fraction": 0.0}).max_penetration == 0.0      # the old format: unchanged
    assert SimHealth.from_dict({"max_penetration": 0.02}).max_penetration == 0.02
    t = make_traces(1, Mode.CLEAN)[0]
    assert EpisodeTrace.from_dict(t.to_dict()) == t


def test_existing_traces_with_numbers_are_unchanged():
    traces = make_traces(6, Mode.CLEAN, Faults(interpenetration=True))
    assert all(isinstance(t.sim.max_penetration, float) for t in traces)
    assert not_measured(traces) == {}
    c = check(traces)
    found = c.run(INCIDENT)
    assert [e.id for e in found] == ["s1.sanity.interpenetration"]
    assert "not measured" not in found[0].detail and c.notes(INCIDENT) == []


def test_unmeasured_penetration_is_skipped_never_scored_as_healthy():
    traces = unmeasured(make_traces(6, Mode.CLEAN, Faults(interpenetration=True)))
    assert M_PENETRATION not in sanity_metrics(traces)
    assert not_measured(traces) == {M_PENETRATION: 6}
    c = check(traces)
    assert c.run(INCIDENT) == []
    assert c.notes(INCIDENT) == ["sanity: sanity/max_penetration not measured (6 of 6 episodes)"]


def test_evidence_from_a_measured_metric_says_what_was_not_measured():
    measured = make_traces(6, Mode.CLEAN, Faults(nan_state=True))
    before = check(measured).run(INCIDENT)
    after = check(unmeasured(measured)).run(INCIDENT)
    assert [e.id for e in after] == [e.id for e in before] == ["s1.sanity.nan_state"]
    b, a = before[0], after[0]
    assert (a.signal, a.strength, a.location) == (b.signal, b.strength, b.location) == (
        V.SIG_RENDER_PHYSICS_BROKEN, b.strength, "sim.nan_state")
    assert a.detail == b.detail + "; sanity/max_penetration not measured (6 of 6 episodes)"


def test_partly_measured_penetration_is_scored_on_the_measured_episodes():
    traces = make_traces(6, Mode.CLEAN, Faults(interpenetration=True))
    mixed = unmeasured(traces, every=3)                     # episodes 0 and 3 not measured
    assert sanity_metrics(mixed)[M_PENETRATION] == [t.sim.max_penetration for i, t in enumerate(traces) if i % 3]
    found = check(mixed).run(INCIDENT)
    assert [e.id for e in found] == ["s1.sanity.interpenetration"]
    assert found[0].detail.endswith("; sanity/max_penetration not measured (2 of 6 episodes)")
    assert "median over 4 episodes" in found[0].detail


def test_not_measured_is_a_note_not_a_skip():
    res = run_stage1(INCIDENT, [check(unmeasured(make_traces(6, Mode.CLEAN)))])
    assert res.sources_run == ("sanity",) and res.partial == () and res.evidence == ()
    assert res.notes == ("sanity: sanity/max_penetration not measured (6 of 6 episodes)",)
    assert "sanity" not in {s for s, _ in res.skipped}


@pytest.mark.parametrize("value", [0.0, 0.001, 0.03])
def test_a_measured_value_is_kept_as_a_float(value):
    assert SimHealth(max_penetration=value).max_penetration == value
    assert SimHealth.from_dict(SimHealth(max_penetration=value).to_dict()).max_penetration == value
