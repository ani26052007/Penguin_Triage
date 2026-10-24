"""Stage 1 check 4: staleness probe (plan.md 3.2.4)."""

from dataclasses import replace

import pytest

from router import vocab as V
from router.prereq.trace import TraceUnavailable
from router.schemas import Mode
from router.stage1.nominal import MissingNominal, NominalStats
from router.stage1.staleness import (StalenessProbe, delay_summary, estimate_extra_delay, extra_delays, mode_gap)
from router.stage1.synthetic import BASE_LATENCY, Faults, make_incident, make_nominal, make_source, make_traces

NOMINAL = make_nominal()
INCIDENT = make_incident()


def run(clean=Faults(), incident=None, **kw):
    return StalenessProbe(NOMINAL, make_source(clean, incident)).run(make_incident(**kw))


def ids(found):
    return sorted(e.id for e in found)


def test_clean_traces_have_no_evidence():
    assert run() == []


def test_chunking_sawtooth_is_not_counted_as_delay():
    traces = make_traces(4, Mode.CLEAN)
    assert all(d == BASE_LATENCY for t in traces for d in extra_delays(t))
    assert delay_summary(traces).mean == pytest.approx(BASE_LATENCY)


def test_known_constant_delay_is_recovered_exactly():
    clean = make_traces(6, Mode.CLEAN, Faults(extra_delay=3))
    assert estimate_extra_delay(clean, NOMINAL) == pytest.approx(3.0)
    found = run(Faults(extra_delay=3))
    e = next(e for e in found if e.id == "s1.staleness.constant")
    assert e.source == "staleness" and e.signal == V.SIG_STALENESS_CONSTANT and e.location == "controller.latency"
    assert "3.00 steps" in e.detail and "constant" in e.detail
    assert e.strength == pytest.approx(0.9)               # z = 3 / 0.25 floor = 12 = 3x fire_z; r^2/(1+r^2) = 0.9
    assert "s1.staleness.variable" not in ids(found)          # constant latency is not server staleness


def test_server_staleness_shows_only_in_incident_mode_and_is_named_variable():
    found = run(clean=Faults(), incident=Faults(server_delay_max=6))
    assert ids(found) == ["s1.staleness.variable"]
    e = found[0]
    assert e.signal == V.SIG_STALENESS_VARIABLE and e.location == "policy_server" and e.strength > 0.5
    assert "steps staler in incident mode" in e.detail


def test_constant_delay_in_both_modes_is_code_not_server():
    found = run(clean=Faults(extra_delay=3), incident=Faults(extra_delay=3))
    assert ids(found) == ["s1.staleness.constant"]            # same delay in both modes: no server gap


def test_both_can_be_present():
    found = run(clean=Faults(extra_delay=3), incident=Faults(extra_delay=3, server_delay_max=6))
    assert ids(found) == ["s1.staleness.constant", "s1.staleness.variable"]


def test_mode_gap_is_signed_and_zero_for_equal_modes():
    a, b = make_traces(6, Mode.CLEAN), make_traces(6, Mode.INCIDENT)
    assert mode_gap(a, b) == pytest.approx(0.0)
    assert mode_gap(a, make_traces(6, Mode.INCIDENT, Faults(extra_delay=2))) == pytest.approx(2.0)


def test_incident_traces_are_only_read_when_the_incident_was_observed_there():
    # server staleness planted in incident traces, but the failure was observed in clean mode
    assert run(clean=Faults(), incident=Faults(server_delay_max=6), observed_mode=Mode.CLEAN) == []


def test_missing_incident_traces_degrade_to_the_clean_half(caplog):
    from router.prereq.trace import InMemoryTraceSource
    src = InMemoryTraceSource({"clean": make_traces(6, Mode.CLEAN, Faults(extra_delay=3))})
    with caplog.at_level("WARNING", logger="router.stage1"):
        found = StalenessProbe(NOMINAL, src).run(INCIDENT)
    assert ids(found) == ["s1.staleness.constant"] and "staleness_variable) not checked" in caplog.text


def test_missing_incident_traces_are_a_partial_skip_not_just_a_warning():
    """Tier 5 (Fix F1): the clean half still reports, and the missing incident
    half is returned as a partial-skip reason."""
    from router.prereq.trace import InMemoryTraceSource
    src = InMemoryTraceSource({"clean": make_traces(6, Mode.CLEAN, Faults(extra_delay=3))})
    found, blind = StalenessProbe(NOMINAL, src).run_partial(INCIDENT)
    assert ids(found) == ["s1.staleness.constant"]
    assert len(blind) == 1 and blind[0].signals == {V.SIG_STALENESS_VARIABLE} and "incident" in blind[0].reason
    # incident traces present but without step pairs: also partial
    empty = InMemoryTraceSource({"clean": make_traces(6, Mode.CLEAN),
                                 "incident": [replace(t, step_pairs=()) for t in make_traces(6, Mode.INCIDENT)]})
    assert StalenessProbe(NOMINAL, empty).run_partial(INCIDENT)[1]
    # observed in clean mode: the incident half does not apply, nothing is skipped
    assert StalenessProbe(NOMINAL, src).run_partial(make_incident(observed_mode=Mode.CLEAN))[1] == []


def test_missing_clean_traces_raise():
    with pytest.raises(TraceUnavailable):
        run(log_refs={})


def test_missing_nominal_fails_at_construction():
    with pytest.raises(MissingNominal):
        StalenessProbe(NominalStats({}), make_source())
